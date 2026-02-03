# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from torch_geometric.nn import GCNConv

# class OmegaModule(nn.Module):
#     # 与节点固有频率有关
#     def __init__(self, hidden_dim):
#         super(OmegaModule, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.omega_param = nn.Parameter((1.0 / np.sqrt(2)) * torch.ones(hidden_dim // 2, 2))

#     def forward(self, x):
#         B, T, C = x.shape
#         x_reshaped = x.transpose(1, 2).unflatten(1, (C // 2, 2))  # (B,128,2,T)

#         omega = torch.linalg.norm(self.omega_param, dim=1).view(1, -1, 1, 1)  # (1,128,1,1)

#         # 保持 4D： (B,128,1,T)
#         x1 = x_reshaped[:, :, 1:2, :]
#         x0 = x_reshaped[:, :, 0:1, :]

#         omega_x = torch.cat([omega * x1, -omega * x0], dim=2)  # (B,128,2,T)
#         omega_x = omega_x.flatten(1, 2).transpose(1, 2)        # (B,T,256)
#         return omega_x
# """
# 这里的写法会有维数对不齐的问题
# """
#     # def forward(self, x):
#     #     B, T, C = x.shape
#     #     x_reshaped = x.transpose(1, 2).unflatten(1, (C // 2, 2))
#     #     omega = torch.linalg.norm(self.omega_param, dim=1)  
#     #     omega = omega.unsqueeze(0)  
#     #     while omega.ndim < x_reshaped.ndim:
#     #         omega = omega.unsqueeze(-1)
#     #     omega_x = torch.stack([omega * x_reshaped[:, :, 1], -omega * x_reshaped[:, :, 0]], dim=2)
#     #     omega_x = omega_x.flatten(1, 2).transpose(1, 2)
#     #     return omega_x

# class SyncModule(nn.Module):
#     # 计算矩阵K？？这里的adj是拉普拉斯矩阵还是邻接矩阵？？
#     def __init__(self, num_nodes):
#         super(SyncModule, self).__init__()
#         self.param = nn.Parameter(torch.empty(num_nodes, num_nodes))
#         nn.init.xavier_uniform_(self.param)

#     def forward(self, adj, x):
#         sym_param = (self.param + self.param.t()) / 2
#         P_new = sym_param * adj  
#         output = F.relu(torch.matmul(P_new, x))
#         return output

# class f_phi(nn.Module):
#     def __init__(self, in_channels,  mapping_type, N):
#         super(f_phi, self).__init__()
#         self.mapping_type = mapping_type
#         self.N = N
#         if self.mapping_type == 'conv':
#             self.mapping_conv = nn.Conv1d(in_channels, in_channels * N, kernel_size=1)
#         elif self.mapping_type == 'gconv':
#             self.mapping_gconv = GCNConv(in_channels, in_channels * N)
#         self.bias = nn.Parameter(torch.zeros(in_channels))
        
#     def forward(self, x, adj):
#         print(f"f_phi-----------------------")
#         print(f"x.shape{x.shape}")
#         # x进入时是 B N T*dim
#         # mappint_type默认是 conv
#         if self.mapping_type == 'conv':
#             x = x.permute(0, 2, 1)  
#             print(f"x.shape{x.shape}")
#             x = self.mapping_conv(x)  
#             print(f"x.shape{x.shape}")
#         elif self.mapping_type == 'gconv':
#             edge_index = self._get_edge_index(adj.squeeze())
#             x = self.mapping_gconv(x.squeeze(0), edge_index).T.unsqueeze(0)
#         x = x.unflatten(1, (self.N, -1))
#         print(f"x.shape{x.shape}")
#         x = x.mean(dim=1)
#         #x = torch.linalg.norm(x, dim=2)
#         print(f"x.shape{x.shape}")
#         # 这一行会报错
#         x = x + self.bias.unsqueeze(0).unsqueeze(-1)

#         print(f"x.shape{x.shape}")
#         return x
    
# # class f_phi(nn.Module):
# #     def __init__(self, in_channels,  mapping_type, N):
# #         super(f_phi, self).__init__()
# #         self.mapping_type = mapping_type
# #         self.N = N
# #         if self.mapping_type == 'conv':
# #             self.mapping_conv = nn.Conv1d(in_channels, in_channels * N, kernel_size=1)
# #         elif self.mapping_type == 'gconv':
# #             self.mapping_gconv = GCNConv(in_channels, in_channels * N)
# #         self.bias = nn.Parameter(torch.zeros(in_channels))
        
# #     def forward(self, x, adj):
# #         if self.mapping_type == 'conv':
# #             x = x.permute(0, 2, 1)  
# #             x = self.mapping_conv(x)  
# #         elif self.mapping_type == 'gconv':
# #             edge_index = self._get_edge_index(adj.squeeze())
# #             x = self.mapping_gconv(x.squeeze(0), edge_index).T.unsqueeze(0)
# #         x = x.unflatten(1, (self.N, -1))


# #         bias = self.bias.view(1, 1, -1, 1)        # ✅ (1,1,256,1)
# #         print(f"bias:{bias.shape}       x:{x.shape}")
# #         x = x + bias

# #         x = torch.linalg.norm(x, dim=2)
# #         # x = x + self.bias.unsqueeze(0).unsqueeze(-1)
# #         return x
    
#     def _get_edge_index(self, adj):
#         edge_index = torch.nonzero(adj, as_tuple=False).T
#         return edge_index

# class Kuramoto_Solver(nn.Module):  
#     # 进行动力学演化
#     def __init__(self, N, hidden_dim, beta, T, L, mapping_type='conv', num_modes=116):
#         super().__init__()
#         self.N = N
#         self.T = T
#         self.L = L
#         self.hidden_dim = hidden_dim
#         self.beta = beta
#         self.mapping_type = mapping_type
#         self.num_modes = num_modes
        
#         self.omega_module = OmegaModule(hidden_dim)
#         self.sync_module = SyncModule(num_modes)
#         self.norm_y = nn.GroupNorm(hidden_dim // N, hidden_dim, affine=True)
#         # 这个f_phi有问题，传输的参数不对
#         # self.f_phi = f_phi(in_channels=hidden_dim, out_channels=hidden_dim, mapping_type=mapping_type, N=N)
#         self.f_phi = f_phi(in_channels=hidden_dim, mapping_type=mapping_type, N=N)

#     # 对节点进行状态更新
#     def surrounding_osc(self, x: torch.Tensor, y: torch.Tensor, adj: torch.Tensor, memory_level=1):
#         wx = self.sync_module(adj, x)
#         z = wx + memory_level * y
#         return z
    
#     def project_osc(self, x, z):
#         # 完成投影操作
#         B, T, C = x.shape
#         # 将x，z都转为B T N D四维
#         x = x.transpose(1, 2).unflatten(1, (self.N, C // self.N))
#         z = z.transpose(1, 2).unflatten(1, (self.N, C // self.N))
#         # 这是什么意思？
#         phi_z = z - torch.sum(x * z, dim=-1, keepdim=True) * x
#         phi_z = phi_z.flatten(1, 2).transpose(1, 2)
#         return phi_z
    
#     def update_osc(self, omega_x, phi_z):
#         # 这里的维度也对不齐
#         print(f"omega:{omega_x.shape}     phi_z:{phi_z.shape}")
#         # 这里已经对其，直接加即可
#         # delta_x = omega_x + self.beta * phi_z.flatten(1, 2).transpose(1, 2)
#         delta_x = omega_x + self.beta * phi_z
#         return delta_x
    
#     def map_to_sphere(self, x):
#         x = x.transpose(1, 2).unflatten(1, (-1, self.N))
#         x = F.normalize(x, dim=2)   # 归一化
#         x = x.flatten(1, 2).transpose(1, 2)
#         return x

    
#     def forward(self, x, y, adj):
#         x_L = []
#         print(f"y-----------:{y.shape}")
#         print(f"x----------{x.shape}")
#         for _ in range(self.L):
#             y = self.norm_y(y)

#             y = y.transpose(1, 2) if y.shape[1]==self.hidden_dim else y #[B, T, C]
#             x = x.transpose(1, 2) if x.shape[1]==self.hidden_dim else x #[B, T, C]
#             print(f"y.shape:{y.shape}   x.shape:{x.shape}")
#             x = self.map_to_sphere(x)
#             print(f"x----------{x.shape}")
#             for _ in range(self.T):
#                 # 这里完成对x的多次更新
#                 # 为什么每次更新都要重新计算x的固定频率
#                 # 这里面的y一只不动，每次变动的只有固有频率omega
#                 omega = self.omega_module(x)
#                 Z = self.surrounding_osc(x, y, adj)
#                 Phi_Z = self.project_osc(x, Z)
#                 Delta_X = self.update_osc(omega, Phi_Z)
#                 x = self.map_to_sphere(Delta_X) 
#                 # 这一步是保存x的快照
#                 x_L.append(x.unsqueeze(1))
                
#             y = self.f_phi(x, adj)
#         return x, y, x_L






import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv

class OmegaModule(nn.Module):
    def __init__(self, hidden_dim):
        super(OmegaModule, self).__init__()
        self.hidden_dim = hidden_dim
        # 这里构造了一个可学习的参数矩阵[128, 2] 每行表示一个复数a+bi 表示在二维球面的旋转
        # 复数值都默认为 1/√2  
        self.omega_param = nn.Parameter((1.0 / np.sqrt(2)) * torch.ones(hidden_dim // 2, 2))
    def forward(self, x):
        # TODO 不知到为什么，这里x 的输入维度是 B N hidden

        B, hidden, node = x.shape
        # 既然前面构造了一个复数矩阵，那么hidden//2就是在二维圆上进行旋转。
        # 通过旋转90度来构造出垂直的向量
        # TODO 这里有所修改，没有使用转置
        x_reshaped = x.unflatten(1, (hidden // 2, 2))   
        omega = torch.linalg.norm(self.omega_param, dim=1)      # 这里是按行求模长 每行由二维变成了一维标量
        omega = omega.unsqueeze(0)      # 目前维度是 1 hidden //2 
        while omega.dim() < x_reshaped.dim():     
            # 这里while运行完，维度变为 1 hidden//2 1 1 同时x_reshaped 维度是 B hidden//2 2 node
            omega = omega.unsqueeze(-1)    
        omega = omega.squeeze(-1)      #  把omega变成三维


        omega_x = torch.stack([omega * x_reshaped[:,:,1], -omega * x_reshaped[:,:,0]], dim=2)   
        omega_x = omega_x.flatten(1, 2)     # 这里的omeag_x维度是 B, hidden, node
        return omega_x
    # def forward(self, x):
    #     B, T, C = x.shape
    #     x_reshaped = x.transpose(1, 2).unflatten(1, (C // 2, 2))
    #     omega = torch.linalg.norm(self.omega_param, dim=1)  
    #     omega = omega.unsqueeze(0)  
    #     while omega.ndim < x_reshaped.ndim:
    #         omega = omega.unsqueeze(-1)
    #     omega_x = torch.stack([omega * x_reshaped[:, :, 1], -omega * x_reshaped[:, :, 0]], dim=2)
    #     omega_x = omega_x.flatten(1, 2).transpose(1, 2)
    #     return omega_x

class SyncModule(nn.Module):
    def __init__(self, num_nodes):
        super(SyncModule, self).__init__()
        self.param = nn.Parameter(torch.empty(num_nodes, num_nodes))
        nn.init.xavier_uniform_(self.param)

    def forward(self, adj, x):
        
        # 计算矩阵A
        sym_param = (self.param + self.param.T) / 2
        # A * adj，这里为什么不是矩阵乘法
        P_new = sym_param * adj     # 这里x的维度是B hidden node
        # 自己加的
        x = x.transpose(1, 2)
        
        output = F.relu(torch.matmul(P_new, x)) # 计算K @ x
        output = output.transpose(1, 2)      # 这里也要将维度变回去
        return output
    
# class f_phi(nn.Module):
#     def __init__(self, in_channels,  mapping_type, N):
#         super(f_phi, self).__init__()
#         self.mapping_type = mapping_type
#         self.N = N
#         if self.mapping_type == 'conv':
#             self.mapping_conv = nn.Conv1d(in_channels, in_channels * N, kernel_size=1)
#         elif self.mapping_type == 'gconv':
#             self.mapping_gconv = GCNConv(in_channels, in_channels * N)
#         self.bias = nn.Parameter(torch.zeros(in_channels))
        
#     def forward(self, x, adj):
#         if self.mapping_type == 'conv':
#             x = x.transpose(1, 2)
#             # mapping_conv 期望输入的是in channels 输出的是 in channels * wavelet
#             x = self.mapping_conv(x)  
#         elif self.mapping_type == 'gconv':
#             edge_index = self._get_edge_index(adj.squeeze())
#             x = self.mapping_gconv(x.squeeze(0), edge_index).T.unsqueeze(0)
#         x = x.unflatten(1, (self.N, -1))
#         x = torch.linalg.norm(x, dim=2)
#         x = x + self.bias.unsqueeze(0).unsqueeze(-1)
#         return x
    
#     def _get_edge_index(self, adj):
#         edge_index = torch.nonzero(adj, as_tuple=False).T
#         return edge_index

class f_phi(nn.Module):
    def __init__(self, in_channels, mapping_type, N):
        super(f_phi, self).__init__()
        self.mapping_type = mapping_type
        
        # 不需要 * N，保持特征维度一致
        if self.mapping_type == 'conv':
            # 点对点的变换 (1x1 Conv)
            self.mapping = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        elif self.mapping_type == 'gconv':
            # 利用图结构的变换
            self.mapping = GCNConv(in_channels, in_channels)
        
        # Bias 维度应该是 [in_channels, 1] 或者 [1, in_channels, 1]
        self.bias = nn.Parameter(torch.zeros(in_channels, 1)) 

    def forward(self, x, adj):
        # 假设输入 x: [B, hidden, N]
        
        if self.mapping_type == 'conv':
            y = self.mapping(x) # [B, hidden, N]
            
        elif self.mapping_type == 'gconv':
            # GCN 处理需要转置
            # x: [B, C, N] -> [N, C] (假设 batch=1 或需 reshape)
            x_in = x.squeeze(0).T 
            edge_index = self._get_edge_index(adj.squeeze())
            
            y = self.mapping(x_in, edge_index) # [N, C]
            y = y.T.unsqueeze(0) # [1, C, N]
            
        # 加入 Bias
        y = y + self.bias
        
        # 关键：论文 Source 916 要求 y 也在球面上
        # Normalize along the channel dimension (dim=1)
        y = F.normalize(y, p=2, dim=1)
        
        return y

    def _get_edge_index(self, adj):
        edge_index = torch.nonzero(adj, as_tuple=False).T
        return edge_index

class Kuramoto_Solver(nn.Module):  
    def __init__(self, N, hidden_dim, beta, T, L, mapping_type='conv', num_modes=116):
        super().__init__()
        self.N = N
        self.T = T
        self.L = L
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.mapping_type = mapping_type
        self.num_modes = num_modes
        
        self.omega_module = OmegaModule(hidden_dim)
        self.sync_module = SyncModule(num_modes)
        
        """
        这里的N是4 先整除 得到每个小波滤波器的张量 然后归一化再返回dim的维度。
        affine=true意思是归一化后还有可学习的参数 y_new = /scale * y_norm + /bias
        """
        self.norm_y = nn.GroupNorm(hidden_dim // N, hidden_dim, affine=True)
        self.f_phi = f_phi(in_channels=hidden_dim,  mapping_type=mapping_type, N=N)
        
    def surrounding_osc(self, x: torch.Tensor, y: torch.Tensor, adj: torch.Tensor, memory_level=1):
        wx = self.sync_module(adj, x)       # 这里wx的维度是B hidden node
        z = wx + memory_level * y   # z = relu(kx) + memory * y
        return z
    
    def project_osc(self, x, z):
        # 这里x和z的输入维度都是 B hidden dim 
        # B, T, C = x.shape
        # x = x.transpose(1, 2).unflatten(1, (self.N, C // self.N))
        # z = z.transpose(1, 2).unflatten(1, (self.N, C // self.N))
        # phi_z = z - torch.sum(x * z, dim=-1, keepdim=True) * x
        # phi_z = phi_z.flatten(1, 2).transpose(1, 2)

        # 这里是给z进行投影，是proj的过程
        B, hidden, dim = x.shape
        x = x.unflatten(1, (self.N, hidden//self.N))        # 这里x的维度是 B wavelet hiddem//4, node
        z = z.unflatten(1, (self.N, hidden//self.N))
        phi_z = z - torch.sum(x * z, dim = -1, keepdim=True) * x
        phi_z = phi_z.flatten(1, 2)
        return phi_z
    
    def update_osc(self, omega_x, phi_z, x):
        x_new = omega_x + self.beta * phi_z + x
        return x_new
    
    def map_to_sphere(self, x):
        """
            对这里进行了修改 x输入到维度是 B hidden N 
            投影到球面的应该是 hidden // wavelet 
            作者原代码表意不明
        """
        # x = x.transpose(1,2).unflatten(1, (-1, self.N))
        # x = F.normalize(x, dim=2)
        # x = x.flatten(1, 2).transpose(1, 2)
        x = x.unflatten(1, (self.N, -1))    # 这里的维度是 B wavelet dim node
        x = F.normalize(x, dim=2)
        x = x.flatten(1, 2)     # 这里flatten处理后维度是 B hidden node.  作者还进行了transpose 但是感觉不对
        return x

    
    def forward(self, x, y, adj):
        x_L = []
        for _ in range(self.L):
            y = self.norm_y(y)
            # 这里把维度搞反了，反而没能充当一个保险
            # y = y.transpose(1, 2) if y.shape[1]==self.hidden_dim else y #[B, T, C]
            # x = x.transpose(1, 2) if x.shape[1]==self.hidden_dim else x #[B, T, C]
            #print(f"x_{x.shape}====y{y.shape}")
            omega = self.omega_module(x)    # 这里omega 的维度是B hidden node
            x = self.map_to_sphere(x)       # 这里x的维度也是 B hidden node
            for _ in range(self.T):
                
                Z = self.surrounding_osc(x, y, adj)     # 这里Z的维度是 B hidden node
                Phi_Z = self.project_osc(x, Z)  
                x_new = self.update_osc(omega, Phi_Z, x)
                # 这里将Delta_X 投影到球面上，但是并没有对x得值进行更新
                x_new = self.map_to_sphere(x_new) 
                x = x_new
                x_L.append(x.unsqueeze(1))      # 这一句话莫名其妙
                
            y = self.f_phi(x, adj)
        return x, y, x_L