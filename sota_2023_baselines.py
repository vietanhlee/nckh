import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. STAEformer (CIKM 2023) - Proxy Implementation
# Spatio-Temporal Adaptive Embedding makes vanilla Transformer SOTA
# ============================================================
class STAEformerProxy(nn.Module):
    def __init__(self, num_nodes, in_channels, T_in, horizon, embed_size=32, heads=4, num_blocks=1, out_dim=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.T_in = T_in
        self.horizon = horizon
        self.embed_size = embed_size
        
        # Adaptive Embeddings (Core novelty of STAEformer)
        self.E_s = nn.Parameter(torch.randn(num_nodes, embed_size))
        self.E_t = nn.Parameter(torch.randn(T_in, embed_size))
        
        self.fc_in = nn.Linear(in_channels, embed_size)
        
        # Memory-efficient decoupled Transformer
        self.temp_attn = nn.TransformerEncoderLayer(d_model=embed_size, nhead=heads, dim_feedforward=embed_size*2, batch_first=True)
        self.spat_attn = nn.TransformerEncoderLayer(d_model=embed_size, nhead=heads, dim_feedforward=embed_size*2, batch_first=True)
        
        self.fc_out = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, horizon * out_dim)
        )
        
    def forward(self, x):
        # x: (B, T, N, F)
        B, T, N, F_in = x.shape
        x = self.fc_in(x) # (B, T, N, D)
        
        # Add Spatio-Temporal Adaptive Embeddings
        x = x + self.E_s.view(1, 1, N, self.embed_size) + self.E_t.view(1, T, 1, self.embed_size)
        
        # Temporal Attention (reshape to treat N as batch)
        x_t = x.transpose(1, 2).reshape(B * N, T, self.embed_size)
        x_t = self.temp_attn(x_t)
        x = x_t.reshape(B, N, T, self.embed_size).transpose(1, 2)
        
        # Spatial Attention (reshape to treat T as batch)
        x_s = x.reshape(B * T, N, self.embed_size)
        x_s = self.spat_attn(x_s)
        x = x_s.reshape(B, T, N, self.embed_size)
        
        # Pooling over time to predict future
        x_pool = x.mean(dim=1) # (B, N, D)
        
        out = self.fc_out(x_pool) # (B, N, H * out_dim)
        out = out.view(B, N, self.horizon, -1).transpose(1, 2) # (B, H, N, out_dim)
        return out


# ============================================================
# 2. MegaCRN (2023) - Proxy Implementation
# Meta-Graph Convolutional Recurrent Network
# ============================================================
class MegaCRNProxy(nn.Module):
    def __init__(self, num_nodes, in_channels, T_in, horizon, embed_size=32, out_dim=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.out_dim = out_dim
        
        # Meta-graph node embeddings (Core novelty of MegaCRN)
        self.node_emb = nn.Parameter(torch.randn(num_nodes, embed_size))
        
        # Encoder-Decoder structure
        self.enc_gru = nn.GRU(in_channels, embed_size, batch_first=True)
        self.dec_gru = nn.GRU(embed_size, embed_size, batch_first=True)
        
        self.fc_out = nn.Linear(embed_size, out_dim)
        
        # Dynamic GCN
        self.gcn_weight = nn.Parameter(torch.randn(embed_size, embed_size))
        
    def forward(self, x):
        # x: (B, T, N, F)
        B, T, N, F_in = x.shape
        
        # Compute dynamic meta-graph
        A_meta = F.softmax(torch.mm(self.node_emb, self.node_emb.transpose(0, 1)), dim=-1) # (N, N)
        
        # Encoder (treat N as batch to avoid full BxN sequential iteration)
        x_enc = x.transpose(1, 2).reshape(B * N, T, F_in)
        _, h_n = self.enc_gru(x_enc) # h_n: (1, B*N, D)
        h = h_n.squeeze(0).view(B, N, -1) # (B, N, D)
        
        # Apply Meta-Graph Convolution
        h_gcn = torch.einsum('nn, bnd -> bnd', A_meta, h)
        h_gcn = torch.matmul(h_gcn, self.gcn_weight)
        h = F.relu(h + h_gcn) # Residual
        
        # Decoder
        decoder_input = torch.zeros(B * N, self.horizon, h.size(-1), device=x.device)
        dec_out, _ = self.dec_gru(decoder_input, h.view(B*N, 1, -1).transpose(0, 1)) # dec_out: (B*N, H, D)
        dec_out = dec_out.view(B, N, self.horizon, -1).transpose(1, 2) # (B, H, N, D)
        
        out = self.fc_out(dec_out) # (B, H, N, out_dim)
        return out


# ============================================================
# 3. DSTAGNN (ICML 2022) - Proxy Implementation
# Dynamic Spatial-Temporal Aware Graph Neural Network
# ============================================================
class DSTAGNNProxy(nn.Module):
    def __init__(self, num_nodes, in_channels, T_in, horizon, embed_size=32, heads=4, out_dim=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.fc_in = nn.Linear(in_channels, embed_size)
        
        # Temporal Attention for dynamic representation
        self.temp_attn = nn.TransformerEncoderLayer(d_model=embed_size, nhead=heads, dim_feedforward=embed_size*2, batch_first=True)
        
        # Chebyshev-like GCN
        self.gcn_w1 = nn.Linear(embed_size, embed_size)
        self.gcn_w2 = nn.Linear(embed_size, embed_size)
        
        self.fc_out = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, horizon * out_dim)
        )
        
    def forward(self, x):
        # x: (B, T, N, F)
        B, T, N, F_in = x.shape
        x = self.fc_in(x)
        
        # Temporal Attention
        x_t = x.transpose(1, 2).reshape(B * N, T, -1)
        x_t = self.temp_attn(x_t)
        x_repr = x_t.reshape(B, N, T, -1).mean(dim=2) # (B, N, D)
        
        # Dynamic Spatial Aware Graph (batch-wise)
        # Cosine similarity between node representations
        x_norm = F.normalize(x_repr, p=2, dim=-1)
        A_dyn = torch.bmm(x_norm, x_norm.transpose(1, 2)) # (B, N, N)
        A_dyn = F.softmax(A_dyn, dim=-1)
        
        # Dynamic GCN
        h1 = self.gcn_w1(x_repr)
        h2 = torch.bmm(A_dyn, self.gcn_w2(x_repr))
        h = F.relu(h1 + h2) # (B, N, D)
        
        out = self.fc_out(h) # (B, N, H * out_dim)
        out = out.view(B, N, self.horizon, -1).transpose(1, 2) # (B, H, N, out_dim)
        return out
