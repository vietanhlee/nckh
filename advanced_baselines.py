import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================
# 1. Graph WaveNet (Wu et al., IJCAI 2019)
# ============================================================
class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        # x: (B, C, N, T), A: (N, N)
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class GraphWaveNet(nn.Module):
    def __init__(self, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=4, out_dim=1, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512, kernel_size=2, blocks=4, layers=2, horizon=6):
        super(GraphWaveNet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.horizon = horizon

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)
                self.supports_len += 1
            else:
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation))
                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=residual_channels, kernel_size=(1, 1)))
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=skip_channels, kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim * horizon, kernel_size=(1, 1), bias=True)
        self.receptive_field = receptive_field
        self.out_dim = out_dim

    def forward(self, input):
        # input: (B, T, N, F) -> (B, F, N, T)
        input = input.permute(0, 3, 2, 1)
        
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input

        x = self.start_conv(x)
        skip = 0

        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]
        elif self.gcn_bool and self.addaptadj and self.supports is None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = [adp]
        elif self.gcn_bool and self.supports is not None:
            new_supports = self.supports

        for i in range(self.blocks * self.layers):
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and new_supports is not None:
                x = self.gconv[i](x, new_supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip[:, :, :, -1:])
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x) # (B, horizon*out_dim, N, 1)
        x = x.squeeze(-1) # (B, horizon*out_dim, N)
        
        B, HO, N = x.shape
        x = x.view(B, self.horizon, self.out_dim, N)
        # return: (B, Horizon, N, F)
        x = x.permute(0, 1, 3, 2)
        return x


# ============================================================
# 2. ASTGCN (Guo et al., AAAI 2019) (Simplified block structure)
# ============================================================
class SpatialAttention(nn.Module):
    def __init__(self, in_channels, num_nodes, num_timesteps):
        super(SpatialAttention, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_timesteps))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_timesteps))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_nodes, num_nodes))
        self.Vs = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.W1, -0.1, 0.1)
        nn.init.uniform_(self.W2, -0.1, 0.1)
        nn.init.uniform_(self.W3, -0.1, 0.1)
        nn.init.uniform_(self.bs, -0.1, 0.1)
        nn.init.uniform_(self.Vs, -0.1, 0.1)

    def forward(self, x):
        # x: (B, N, C, T)
        lhs = torch.einsum('bnct,t,ct->bnt', x, self.W1, self.W2) # (B, N, T)
        rhs = torch.einsum('c,bnct->btn', self.W3, x) # (B, T, N)
        product = torch.matmul(lhs, rhs) # (B, N, N)
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs)) # (B, N, N)
        S = F.softmax(S, dim=-1)
        return S

class TemporalAttention(nn.Module):
    def __init__(self, in_channels, num_nodes, num_timesteps):
        super(TemporalAttention, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_nodes))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_nodes))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels))
        self.be = nn.Parameter(torch.FloatTensor(1, num_timesteps, num_timesteps))
        self.Ve = nn.Parameter(torch.FloatTensor(num_timesteps, num_timesteps))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.U1, -0.1, 0.1)
        nn.init.uniform_(self.U2, -0.1, 0.1)
        nn.init.uniform_(self.U3, -0.1, 0.1)
        nn.init.uniform_(self.be, -0.1, 0.1)
        nn.init.uniform_(self.Ve, -0.1, 0.1)

    def forward(self, x):
        # x: (B, N, C, T)
        lhs = torch.einsum('bnct,n,cn->btn', x, self.U1, self.U2) # (B, T, N)
        rhs = torch.einsum('c,bnct->bnt', self.U3, x) # (B, N, T)
        product = torch.matmul(lhs, rhs) # (B, T, T)
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be)) # (B, T, T)
        E = F.softmax(E, dim=-1)
        return E

class ChebConvASTGCN(nn.Module):
    def __init__(self, K, in_channels, out_channels):
        super(ChebConvASTGCN, self).__init__()
        self.K = K
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels)) for _ in range(K)])
        self.reset_parameters()

    def reset_parameters(self):
        for k in range(self.K):
            nn.init.xavier_normal_(self.Theta[k])

    def forward(self, x, spatial_attention, cheb_polynomials):
        # x: (B, N, C)
        B, N, C = x.shape
        outputs = []
        for k in range(self.K):
            T_k = cheb_polynomials[k] # (N, N)
            T_k_with_at = T_k.unsqueeze(0) * spatial_attention # (B, N, N)
            rhs = torch.bmm(T_k_with_at, x) # (B, N, C)
            output = torch.matmul(rhs, self.Theta[k]) # (B, N, C_out)
            outputs.append(output)
        return F.relu(torch.sum(torch.stack(outputs), dim=0))

class ASTGCNBlock(nn.Module):
    def __init__(self, num_nodes, in_channels, K, num_timesteps, out_channels):
        super(ASTGCNBlock, self).__init__()
        self.SAt = SpatialAttention(in_channels, num_nodes, num_timesteps)
        self.TAt = TemporalAttention(in_channels, num_nodes, num_timesteps)
        self.cheb_conv = ChebConvASTGCN(K, in_channels, out_channels)
        self.time_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(out_channels)

    def forward(self, x, cheb_polynomials):
        # x: (B, N, C, T)
        B, N, C, T = x.shape
        spatial_at = self.SAt(x)
        temporal_at = self.TAt(x)
        x_TAt = torch.matmul(x.reshape(B, -1, T), temporal_at).reshape(B, N, C, T)
        
        spatial_gcn = []
        for t in range(T):
            out = self.cheb_conv(x_TAt[:, :, :, t], spatial_at, cheb_polynomials)
            spatial_gcn.append(out)
        spatial_gcn = torch.stack(spatial_gcn, dim=-1) # (B, N, C_out, T)
        
        time_conv_output = self.time_conv(spatial_gcn.transpose(1, 2)).transpose(1, 2) # (B, N, C_out, T)
        x_residual = self.residual_conv(x.transpose(1, 2)).transpose(1, 2)
        x_out = self.ln((time_conv_output + x_residual).transpose(2, 3)).transpose(2, 3)
        return F.relu(x_out)

class ASTGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, K, num_blocks, T_in, horizon, block_channels=64, L_tilde=None):
        super(ASTGCN, self).__init__()
        self.blocks = nn.ModuleList([ASTGCNBlock(num_nodes, in_channels if i==0 else block_channels, K, T_in, block_channels) for i in range(num_blocks)])
        self.final_conv = nn.Conv2d(T_in, horizon, kernel_size=(1, block_channels))
        self.cheb_polynomials = []
        
        if L_tilde is not None:
            L_tilde = torch.FloatTensor(L_tilde)
            self.cheb_polynomials.append(torch.eye(num_nodes))
            if K > 1:
                self.cheb_polynomials.append(L_tilde)
            for k in range(2, K):
                self.cheb_polynomials.append(2 * torch.matmul(L_tilde, self.cheb_polynomials[-1]) - self.cheb_polynomials[-2])
        else:
            for k in range(K):
                self.cheb_polynomials.append(torch.eye(num_nodes))
                
        self.cheb_polynomials = nn.ParameterList([nn.Parameter(T_k, requires_grad=False) for T_k in self.cheb_polynomials])

    def forward(self, x):
        # x: (B, T, N, F) -> (B, N, F, T)
        x = x.permute(0, 2, 3, 1)
        cheb_poly = [p.to(x.device) for p in self.cheb_polynomials]
        for block in self.blocks:
            x = block(x, cheb_poly)
        # x: (B, N, C_out, T) -> permute to (B, T, N, C_out)
        x = x.permute(0, 3, 1, 2) # (B, T, N, C_out)
        out = self.final_conv(x).squeeze(-1) # (B, horizon, N)
        return out.unsqueeze(-1) # (B, horizon, N, 1)


# ============================================================
# 3. GMAN (Zheng et al., AAAI 2020) (Simplified structure)
# ============================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        
    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out

class GMANBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super(GMANBlock, self).__init__()
        self.spatial_attention = MultiHeadAttention(embed_size, heads)
        self.temporal_attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.ReLU(),
            nn.Linear(embed_size * 2, embed_size)
        )
        
    def forward(self, x):
        # x: (B, T, N, E)
        B, T, N, E = x.shape
        # Spatial Attention: loop over time steps T to save 24x peak GPU memory
        out_s_list = []
        for t in range(T):
            x_st = x[:, t, :, :] # (B, N, E)
            out_st = self.spatial_attention(x_st, x_st, x_st) # (B, N, E)
            out_s_list.append(out_st)
        out_s = torch.stack(out_s_list, dim=1) # (B, T, N, E)
        x = self.norm1(x + out_s)
        
        # Temporal Attention
        x_t = x.transpose(1, 2).reshape(B*N, T, E)
        out_t = self.temporal_attention(x_t, x_t, x_t)
        x = self.norm2(x + out_t.reshape(B, N, T, E).transpose(1, 2))
        
        # FF
        x = x + self.ff(x)
        return x

class GMAN(nn.Module):
    def __init__(self, num_nodes, in_channels, T_in, horizon, embed_size=32, heads=4, num_blocks=1):
        super(GMAN, self).__init__()
        self.fc_in = nn.Linear(in_channels, embed_size)
        self.blocks = nn.ModuleList([GMANBlock(embed_size, heads) for _ in range(num_blocks)])
        # Transform Attention from history to future
        self.transform_attention = MultiHeadAttention(embed_size, heads)
        self.fc_out = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, 1)
        )
        
        # Time embeddings
        self.query_emb = nn.Parameter(torch.randn(1, horizon, 1, embed_size))
        
    def forward(self, x):
        # x: (B, T, N, F)
        x = self.fc_in(x)
        for block in self.blocks:
            x = block(x) # (B, T, N, E)
            
        B, T, N, E = x.shape
        # x: (B*N, T, E)
        x_flat = x.transpose(1, 2).reshape(B*N, T, E)
        
        # Create queries for future steps
        queries = self.query_emb.expand(B, -1, N, -1).transpose(1, 2).reshape(B*N, -1, E)
        
        # Transform attention
        future = self.transform_attention(x_flat, x_flat, queries) # (B*N, Horizon, E)
        future = future.reshape(B, N, -1, E).transpose(1, 2) # (B, Horizon, N, E)
        
        out = self.fc_out(future) # (B, Horizon, N, 1)
        return out
