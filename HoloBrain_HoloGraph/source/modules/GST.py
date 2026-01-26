import torch

class Wavelet(torch.nn.Module):
    # 导入module父类，为了以后可以将gst过程变成可学习的
    def __init__(self, wavelet=[0, 1, 2], level=2):
        super(Wavelet, self).__init__()
        self.wavelet = wavelet
        self.level = level


    # 返回的小波滤波器组，和低通滤波器
    def construct_wavelet(self, adj):
        adj = self.no_zero_adj(adj)
        wavelets = []
        degree = torch.sum(adj, dim=-1)
        if (degree == 0).any():
            raise ValueError("The adj has isolated nodes (degree=0).")
        D = torch.diag_embed(degree)
        D = D.to(adj.device)
        # 这里实际上是构造了拉普拉斯矩阵
        adj = D - adj
        D_inverse = torch.inverse(D)
        D_inverse[D_inverse == float("inf")] = 0.0
        I_n = torch.eye(adj.size(-1)).unsqueeze(0).repeat(adj.size(0), 1, 1).float()
        I_n = I_n.to(adj.device)
        # 这里与论文不一致，这里是I - L @ D_inv
        adj = 0.5 * (I_n + torch.bmm(adj, D_inverse))
        adj_sct = adj.float()
        adj_power = adj_sct.clone()
        for order in self.wavelet:
            if order == 0:
                wavelets.append(I_n - adj_sct)
                continue
            if order > 1:
                adj_power = torch.bmm(adj_power, adj_power)
            # S^n(S^n-I)
            # 这一项等价于 P^{2^k} - P^{2^{k+1}}（因为 A(I-A)=A-A^2），
            # 所以它是在构造 \Psi_h 的那一族差分滤波器（只是索引/幂次的对应关系由 wavelet=[0,1,2] 决定）。
            adj_int = torch.bmm(adj_power, I_n - adj_power)
            wavelets.append(adj_int)
            # print(adj_int.shape)
        # low_pass 是低通滤波
        low_pass = torch.bmm(adj_power, adj_power)  # t^(2^j)
        low_pass = torch.bmm(low_pass, low_pass)  # t^(2^(j+1))
        return wavelets, low_pass


# 临界矩阵度为零的点的处理办法。
    def no_zero_adj(self, adj):
        batch_size, n, _ = adj.shape
        adj_fixed_batch = adj.clone()
        
        for b in range(batch_size):
            adj = adj_fixed_batch[b]
            zero_row_mask = (adj.sum(dim=-1) == 0)
            zero_row_indices = torch.where(zero_row_mask)[0]
            if len(zero_row_indices) == 0:
                continue

            for zero_row in zero_row_indices:
                prev_row = zero_row - 1
                while prev_row >= 0 and zero_row_mask[prev_row]:
                    prev_row -= 1
                
                if prev_row >= 0: 
                    adj[zero_row, :] = adj[prev_row, :]
                else:  
                    next_row = zero_row + 1
                    while next_row < n and zero_row_mask[next_row]:
                        next_row += 1
                    if next_row < n:
                        adj[zero_row, :] = adj[next_row, :]

            adj_symmetric = (adj + adj.T) / 2
            adj_fixed_batch[b] = adj_symmetric

        return adj_fixed_batch

# 将原始信号转为小波滤波器处理后的
    def windowed(self, x, adj):
        # y: B x N x T x dim
        wavelets, low_pass = self.construct_wavelet(adj)
        outputs = [[x.transpose(1, 2)]]
        for layer in range(self.level):
            layer_output = []
            # 拿出时间维度
            for input in outputs[-1]:
                # 小波滤波器组也是N*N的
                for wavelet in wavelets:
                    # 感觉这个乘法的数据并未对齐
                    out = torch.matmul(wavelet, input)
                    out = torch.abs(out)
                    layer_output.append(out)
            outputs.append(layer_output)

        basis = torch.cat([torch.stack(layer, dim=-1) for layer in outputs], dim=-1)

        basis_shape = basis.shape
        basis = basis.view(basis.shape[0], basis.shape[1], -1)
        scattering_coeff = torch.matmul(low_pass, basis)
        scattering_coeff = scattering_coeff.view(basis_shape)
        # 这里返回的BNTdim，不然矩阵乘法会失败
        # B x N x T x dim
        return scattering_coeff


# 这块感觉比较重要
    def nonwindowed(self, x, adj):
        wavelets, low_pass = self.construct_wavelet(adj)
        # 这里不太理解？？
        outputs = [[x.transpose(1, 2)]]
        for layer in range(self.level):
            layer_output = []
            for input in outputs[-1]:
                # 拿出一个用例，乘以H个小波滤波器
                for wavelet in wavelets:
                    out = torch.matmul(wavelet, input)
                    out = torch.abs(out)
                    layer_output.append(out)
            outputs.append(layer_output)

        # B x N x T x dim
        basis = torch.cat([torch.stack(layer, dim=-1) for layer in outputs], dim=-1)
        Q = 2
        gst = []
        for q in range(1, Q + 1):
            q_st = basis**q
            q_st = torch.mean(q_st, dim=1)
            gst.append(q_st)
        gst = torch.stack(gst, dim=-1)  # B x T x Q*dim
        return basis

    def forward(self, x, adj, windowed=True):
        if windowed:
            return self.windowed(x, adj)
        return self.nonwindowed(x, adj)
