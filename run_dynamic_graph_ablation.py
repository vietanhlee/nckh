import os
import gc
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn

from benchmark_5seeds import (
    train_single_seed, load_adj_from_excel, compute_scaled_laplacian,
    load_timeseries_double_rolling, MultiStepDataset
)
from hybrid import Config as HybridConfig, TemporalAttention
from stgcn import STGCNBlock

class TA_STGCN_Dynamic_Model(nn.Module):
    def __init__(self, num_nodes, in_feat, block_hidden, num_blocks, T_in,
                 cheb_K, horizon, output_feat, L_tilde=None, dropout=0.3,
                 use_temporal_attention=True, attn_num_heads=4, attn_dropout=0.1,
                 embed_dim=10):
        super().__init__()
        self.horizon = horizon
        self.output_feat = output_feat
        self.use_temporal_attention = use_temporal_attention

        # Adaptive/Dynamic Graph Node Embeddings
        # E1, E2 in R^{N x 10}
        self.E1 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.E2 = nn.Parameter(torch.randn(num_nodes, embed_dim))

        if use_temporal_attention:
            self.temporal_attn = TemporalAttention(block_hidden, attn_num_heads, attn_dropout)
        else:
            self.temporal_attn = None

        blocks = []
        c_in = in_feat
        for _ in range(num_blocks):
            blocks.append(STGCNBlock(c_in, block_hidden, num_nodes, cheb_K, dropout))
            c_in = block_hidden
        self.blocks = nn.ModuleList(blocks)

        self.final_conv = nn.Conv1d(block_hidden, horizon * output_feat, kernel_size=T_in)

    def forward(self, x):
        # Tính A_dyn = Softmax(ReLU(E1 * E2^T))
        # A_dyn có kích thước (N, N)
        A_dyn = F.relu(torch.matmul(self.E1, self.E2.transpose(0, 1)))
        A_dyn = F.softmax(A_dyn, dim=1) 

        h = x.permute(0, 3, 2, 1)  # (B, F, N, T)

        # Đẩy A_dyn vào GCN thay vì Support Matrix tĩnh
        for block in self.blocks:
            h = block(h, A_dyn)  # (B, C_hidden, N, T)

        B, C, N, T = h.shape

        if self.use_temporal_attention:
            h_seq = h.permute(0, 2, 3, 1).reshape(B * N, T, C)
            h_seq = self.temporal_attn(h_seq)
            h = h_seq.permute(0, 2, 1)                     # (B*N, C, T)
        else:
            h = h.permute(0, 2, 1, 3).reshape(B * N, C, T)   # (B*N, C, T)

        out = self.final_conv(h)     # (B*N, horizon*output_feat, 1)
        out = out.squeeze(-1)
        out = out.view(B, N, self.horizon, self.output_feat)
        y_pred = out.permute(0, 2, 1, 3)  # (B, Horizon, N, output_feat)

        return y_pred

def run_ablation():
    seeds = [42, 100, 2024, 22, 99]
    cfg = HybridConfig()
    cfg.ROOT_DIR = "/workspace/GRAPH"
    cfg.ADJ_PATH = os.path.join(cfg.ROOT_DIR, "Graph_fix_py_3.xlsx")
    cfg.CSV_PATH = os.path.join(cfg.ROOT_DIR, "count_7_7_merg_sort_fix_fill.csv")
    cfg.SAVE_DIR = "model/"
    
    # Cấu hình hệt như chuẩn của benchmark_5seeds
    cfg.EPOCHS = 120
    cfg.PATIENCE = 20
    cfg.BATCH_SIZE = 64
    cfg.LEARNING_RATE = 0.0008

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"============================================================")
    print(f"🚀 CHẠY ABLATION: DYNAMIC ADAPTIVE GRAPH (5 SEEDS, {cfg.EPOCHS} EPOCHS)")
    print(f"============================================================")

    # Đọc nodes, chỉ để lấy độ dài N=608, ta không cần L_tilde thực tế 
    # cho mô hình vì nó dùng A_dyn hoàn toàn. Tuy nhiên ta vẫn gọi ra để khớp.
    A_raw, nodes = load_adj_from_excel(cfg.ADJ_PATH)
    
    df_all = load_timeseries_double_rolling(
        cfg.CSV_PATH, nodes, cfg.DATA_WINDOW1, cfg.DATA_WINDOW2, cfg.TIME_STEP_MINUTES
    )

    n_total = len(df_all)
    n_train = int(n_total * 0.8)
    n_val   = int(n_total * 0.1)

    df_train = df_all.iloc[:n_train]
    df_val   = df_all.iloc[n_train:n_train + n_val]
    df_test  = df_all.iloc[n_train + n_val:]

    model_name = "TA-STGCN (Dynamic Graph)"
    maes = []
    
    for seed in seeds:
        print(f"\n🌱 [SEED {seed}]")
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()

        train_ds = MultiStepDataset(df_train, nodes, cfg.T_IN, cfg.HORIZON)
        scaler   = {'mean': train_ds.means, 'std': train_ds.stds}
        val_ds   = MultiStepDataset(df_val, nodes, cfg.T_IN, cfg.HORIZON, scaler)
        test_ds  = MultiStepDataset(df_test, nodes, cfg.T_IN, cfg.HORIZON, scaler)

        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE)
        test_loader  = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE)

        model = TA_STGCN_Dynamic_Model(
            num_nodes=len(nodes), in_feat=5, block_hidden=cfg.BLOCK_HIDDEN,
            num_blocks=cfg.NUM_BLOCKS, T_in=cfg.T_IN, cheb_K=cfg.CHEB_K,
            horizon=cfg.HORIZON, output_feat=2, L_tilde=None, dropout=cfg.DROPOUT,
            use_temporal_attention=cfg.USE_TEMPORAL_ATTENTION,
            attn_num_heads=4, attn_dropout=cfg.ATTN_DROPOUT,
            embed_dim=10  # Theo đúng yêu cầu E1, E2 có số chiều là 10
        ).to(device)

        test_metrics = train_single_seed(
            model_name, model, train_loader, val_loader, test_loader, cfg, device, seed,
            use_wandb=False 
        )
        maes.append(test_metrics['mae'])
        print(f"   ▶ Seed {seed} | MAE Total: {test_metrics['mae']:.4f}")

    mean_mae = np.mean(maes)
    std_mae = np.std(maes, ddof=1) if len(maes) > 1 else 0.0
    
    print(f"\n{'='*60}")
    print(f"🎉 TỔNG KẾT ABLATION STUDY: DYNAMIC GRAPH")
    print(f"   MAE: {mean_mae:.4f} ± {std_mae:.4f}")
    print(f"{'='*60}")

    with open("ablation_dynamic_graph_report.md", "w", encoding="utf-8") as f:
        f.write(f"# Ablation Study: Dynamic Adaptive Graph\n\n")
        f.write(f"- **Mô hình**: {model_name}\n")
        f.write(f"- **Số Seeds**: {len(seeds)}\n")
        f.write(f"- **Epochs**: {cfg.EPOCHS}\n")
        f.write(f"- **Cơ chế**: Dùng $\\tilde{{A}}_{{dyn}} = \\text{{Softmax}}(\\text{{ReLU}}(E_1 \\cdot E_2^T))$ thay cho Static RBF.\n\n")
        f.write(f"### Kết quả\n")
        f.write(f"- **MAE Tổng thể**: {mean_mae:.4f} ± {std_mae:.4f}\n\n")
        f.write(f"> Đối chiếu số liệu này với TA-STGCN (Static Graph) trong Table VIII để xem Dynamic Graph có lợi hay hại trong dữ liệu Việt Nam.\n")

if __name__ == "__main__":
    run_ablation()
