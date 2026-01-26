import math
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from source.modules.kuramoto_solver import Kuramoto_Solver
from source.modules.GST import Wavelet 
"""
输入信息,输出预测结果。是整体的框架
"""
class BRICK(nn.Module):
    def __init__(self,
                 N=4,
                 L=1,
                 T=8,
                 hidden_dim=256, 
                 num_classes=4,
                 beta=2.5,
                 feature_dim=39,
                 num_nodes=116,
                 use_pe=True,
                 node_cls=False,
                 y_type='linear',
                 mapping_type='conv',
                 parcellation=False,
                 ):

        super(BRICK, self).__init__()
        self.node_classification = node_cls
        self.y_type = y_type
        self.conv_y = nn.Sequential(
            GCNConv(feature_dim, hidden_dim),
            nn.ReLU(),
            GCNConv(hidden_dim, hidden_dim),
        )
        self.linear_y = nn.Linear(feature_dim, hidden_dim)

        self.gst = Wavelet(wavelet=[0, 1, 2], level=1)
        self.x_processor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.use_pe = use_pe
        self.register_buffer('pe_y', self.positional_encoding(hidden_dim, num_nodes))
        self.register_buffer('pe_x', self.positional_encoding(hidden_dim, num_nodes))

        self.kuramoto_solver = Kuramoto_Solver(
            N=N,
            hidden_dim=hidden_dim,
            beta=beta,
            T=T,
            L=L,
            mapping_type=mapping_type,
            num_modes=num_nodes
        )

        self.out_pred = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, num_classes),
        )
        # 输出预测节点是什么？？？
        self.out_pred_node = nn.Sequential( 
            nn.Linear(hidden_dim, 4 * hidden_dim),  
            nn.ReLU(), 
            nn.Linear(4 * hidden_dim, num_classes)  
        )
        
        self.parcellation = parcellation
        
    def positional_encoding(self, d_model, length):
        pe = torch.zeros(d_model, length)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pe[0::2, :] = torch.sin(position * div_term).T
        pe[1::2, :] = torch.cos(position * div_term).T
        return pe


    def forward(self, features, adj):
        # 计算控制y
        # 这里y的维度是B T N
        if self.y_type == "linear":
            y = self.linear_y(features.transpose(1, 2)).transpose(1, 2)
        else:
            y = self.conv_y(features.squeeze().T, torch.nonzero(adj, as_tuple=False).T).T.unsqueeze(0)
        saved_y = y.clone()

        # 用邻接矩阵放入gst中计算，只对时间维度进行处理，windowed模式
        # 为什么这里feature传入gst的时候维度就是BNT了
        x = self.gst(features, adj)
        # flatten后x的维度变为B N T*dim
        x = torch.flatten(x, start_dim=2)
        x = self.x_processor(x).transpose(1, 2)

        # pe是什么
        if self.use_pe:
            y = y + self.pe_y[None, :, :]
            x = x + self.pe_x[None, :, :]
        # 放入 kuramoto 中进行物理演化
        x, y, saved_x = self.kuramoto_solver(x, y, adj)
        
        # 对x进行分类任务
        if self.parcellation:
            w_matrix = saved_y.reshape(saved_y.shape[0], -1)
            output = y.transpose(1, 2).reshape(y.shape[0], -1)
            return output, w_matrix, saved_x, saved_y
        
        if self.node_classification:
            output = self.out_pred_node(y.transpose(1, 2))
        else:
            output = self.out_pred(y)

        return output, saved_x, saved_y
