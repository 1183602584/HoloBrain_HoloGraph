import argparse
import os
import logging
import random
import time
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from accelerate import Accelerator, utils
from torch.distributed.nn.functional import all_gather
from source.utils import LinearWarmupScheduler, compute_weighted_metrics, logger
from source.data.create_dataset import create_dataset
from source.brick import BRICK  
from ema_pytorch import EMA  

"""
代码流程：
读取数据:feature adj label
brick:
    计算控制信号y   x放入gst进行扩展    x扩展后提取高维特征
    x,y放入kuramoto模型中进行演化:
        演化时,控制信号y不变,omega每个时间步都需要变化。(这里与论文描述相反)
        演化结束后,x通过f_phi函数得到y,y放入卷积模型进行分类任务
"""




def train_one_epoch(model, ema, optimizer, scheduler, train_loader, epoch, device, accelerator, log):
    # 进行一次训练
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()


    for batch_idx, (features, adj, targets) in enumerate(train_loader):
        features = features.to(device)
        adj = adj.to(device)

        # unsqueeze： 如果输入的数据是两维的就插入一个维度变成三维。
        if features.dim() == 2:
            features = features.unsqueeze(0)

        targets = targets.to(device)
        # 修改标签维度，这个损失函数只能融入一维的标签。
        targets = targets.squeeze(1) if targets.dim() == 2 else targets

        optimizer.zero_grad()
        outputs, x_features, y_features = model(features, adj)
        # logit = outputs.argmax(dim=1)
        # # 在多GPU上训练时，将多个GPU上的数据可以互相观测到
        # if accelerator.num_processes > 1:
        #     outputs = torch.cat(all_gather(outputs), dim=0)
        #     targets = torch.cat(all_gather(targets), dim=0)

        # 输出与标签进行比较 output [B, hidden*N] 
        loss = criterion(outputs, targets)
        # accelerator:混合多卡时比较适用
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        # ema是什么
        ema.update() 

    avg_loss = total_loss / len(train_loader)
    if accelerator.is_main_process :
        log.info(f"[Epoch {epoch+1}] Training Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(model, accelerator, test_loader, device, log):
    model.eval()
    all_preds = []
    all_targets = []
    all_x_feats = []
    all_y_feats = []
    all_inputs = []
    
    with torch.no_grad():
        for batch_idx, (features, adj, targets) in enumerate(test_loader):
            features = features.to(device)
            adj = adj.to(device)
            if features.dim() == 2:
                features = features.unsqueeze(0)
            targets = targets.to(device)
            targets = targets.squeeze(1) if targets.dim() == 2 else targets

            outputs, x_features, y_features = model(features, adj)
            # Kuramoto returns a list of x snapshots; stack them into a tensor.
            if isinstance(x_features, list):
                x_features = torch.cat(x_features, dim=1)       # (B, steps, hidden, node)
                x_features = x_features.transpose(2, 3)         # (B, steps, node, hidden)
            if accelerator.num_processes > 1:
                outputs = torch.cat(all_gather(outputs), dim=0)
                targets = torch.cat(all_gather(targets), dim=0)
                x_features = torch.cat(all_gather(x_features), dim=0)
                y_features = torch.cat(all_gather(y_features), dim=0)
                features = torch.cat(all_gather(features), dim=0)

            preds = outputs.argmax(dim=-1)
            all_preds.append(preds)
            all_targets.append(targets)
            all_x_feats.append(x_features)
            all_y_feats.append(y_features)
            all_inputs.append(features)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_feats = torch.cat([torch.unsqueeze(torch.transpose(torch.cat(all_y_feats, dim=0), 1, 2), 1),
                           torch.cat(all_x_feats, dim=0)], dim=1)
    metrics = compute_weighted_metrics(all_preds, all_targets)
    return metrics, all_feats, torch.cat(all_inputs, dim=0), all_targets


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--gpu", type=str, default="0", help="GPU id to use")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay factor")

    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--warmup_iters", type=int, default=10)
    parser.add_argument("--batchsize", type=int, default=256)  
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--data", type=str, default="ABIDE", help="Dataset name")
    parser.add_argument("--num_nodes", type=int, default=116, help="Number of nodes")
    parser.add_argument("--feature_dim", type=int, default=175, help="Input feature dimension")
    parser.add_argument("--num_class", type=int, default=2, help="Number of classes")
    parser.add_argument("--L", type=int, default=1, help="Number of Kuramoto solvers")
    parser.add_argument("--h", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--T", type=int, default=8, help="Number of times steps")
    parser.add_argument("--N", type=int, default=4, help="oscillator dimensions")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta for Kuramoto solver")

    parser.add_argument("--use_pe", action="store_true", help="Use positional encoding")
    parser.add_argument("--node_cls", action="store_false", help="Node classification mode")
    parser.add_argument("--y_type", type=str, default="linear", choices=["conv", "linear"], help="y computation type ")
    parser.add_argument("--mapping_type", type=str, default="conv", choices=["conv", "gconv"], help="Mapping type for y")
    parser.add_argument("--parcellation", action="store_false", help="Implement parcellation or not")
    
    args = parser.parse_args()
    utils.set_seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    accelerator = Accelerator()
    device = accelerator.device


    log = logger()
    # 这里读取数据
    dataset = create_dataset(args.data)

    splits = 5
    kfold = KFold(n_splits=splits, shuffle=True, random_state=args.seed)
    all_fold_acc = []
    all_fold_pre = []
    all_fold_f1 = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(dataset)):

        log.info(f"Fold {fold_idx}:")

        # sbuset: 拆分数据按照train_idx
        # 这里可以靠索引读取数据，假设输入维度是B T N
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        if accelerator.is_main_process:
            log.info(f"Train samples: {len(train_subset):,}, Test samples: {len(test_subset):,}")
        
        train_loader = DataLoader(
            train_subset,
            batch_size=args.batchsize // accelerator.num_processes,
            shuffle=True,
            num_workers=args.num_workers,
        )
        test_loader = DataLoader(
            test_subset,
            batch_size=args.batchsize // accelerator.num_processes,
            shuffle=False,
            num_workers=args.num_workers,
        )
        # 初始化模型
        # N是振子的维度，默认为4 h是隐藏层维度。
        model = BRICK(
            N=args.N,
            hidden_dim=args.h,
            L=args.L,   # kuramoto的次数
            T=args.T,   # 时间步的数量，时间步是什么？？
            num_classes=args.num_class,   # 分类数
            beta=args.beta, # ？？？
            feature_dim=args.feature_dim,
            num_nodes=args.num_nodes,
            use_pe=args.use_pe,
            # 节点分类也设为默认关闭
            node_cls=False,
            y_type=args.y_type,
            mapping_type=args.mapping_type,
            # 这里有修改， parcellation是脑区分割的意思
            parcellation=False,
        ).to(device)


        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"Total trainable parameters: {total_params:,}")

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
        # 训练的参数梯度表
        scheduler = LinearWarmupScheduler(optimizer, warmup_iters=args.warmup_iters)
        # ema是什么
        ema = EMA(model, beta=args.ema_decay, update_every=10, update_after_step=200)

        if accelerator.is_main_process:
            log.info(f"Starting training for {args.epochs} epochs...")

        best_test_acc, best_pre, best_f1 = 0, 0, 0

        # 开始训练epochs次
        for epoch in range(args.epochs):    
            epoch_loss = train_one_epoch(model, ema, optimizer, scheduler, train_loader, epoch, device, accelerator, log)
            start_time = time.time()
            metrics, features, inputs_data, gt = evaluate(model, accelerator, test_loader, device, log)
            elapsed_ms = (time.time() - start_time) * 1000 / len(gt)    # 计算总用时
            test_acc, pre, rec, f1 = metrics
            log.info(
                    f"Epoch {epoch+1:03d} | Acc {test_acc:.4f} | Pre {pre:.4f} | Rec {rec:.4f} | "
                    f"F1 {f1:.4f} | Avg {elapsed_ms:7.2f} ms"
                    )

            
            # 更新准确值
            if test_acc > best_test_acc:
                best_test_acc, best_pre, best_f1 = test_acc, pre, f1
                np.save(os.path.join(".", f"fold_{fold_idx}_features.npy"), features.cpu().detach().numpy())
                np.save(os.path.join(".", f"fold_{fold_idx}_inputs.npy"), inputs_data.cpu().detach().numpy())
                np.save(os.path.join(".", f"fold_{fold_idx}_gt.npy"), gt.cpu().detach().numpy())

        if accelerator.is_main_process:
            torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(".", "model.pth"))
            torch.save(ema.state_dict(), os.path.join(".", "ema_model.pth"))

        log.info(f"Fold {fold_idx}: Best Test Acc: {best_test_acc:.4f}, Precision: {best_pre:.4f}, F1: {best_f1:.4f}")
        all_fold_acc.append(best_test_acc)
        all_fold_pre.append(best_pre)
        all_fold_f1.append(best_f1)

    avg_acc = np.mean(all_fold_acc)
    avg_pre = np.mean(all_fold_pre)
    avg_f1 = np.mean(all_fold_f1)
    log.info(f"Final Results -- Average Test Acc: {avg_acc:.4f}, Precision: {avg_pre:.4f}, F1: {avg_f1:.4f}")
    log.info(f"All folds Accuracies: {all_fold_acc}")
    log.info(f"All folds Precisions: {all_fold_pre}")
    log.info(f"All folds F1 scores: {all_fold_f1}")


if __name__ == "__main__":
    main()
