
import torch 
import sys
import os

# 获取 test.py 所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 策略1：如果你觉得 modules 文件夹和 test.py 在同一级
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 策略2：如果 modules 文件夹在 test.py 的上一级 (通常是这种情况导致报错)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from source.brick import BRICK

model = BRICK() 
X = torch.Tensor(torch.randn(5, 175, 116))
adj = torch.Tensor(torch.randn(5, 116, 116))
y_hat = model(X, adj)