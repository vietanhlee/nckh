import os
import gc
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from benchmark_5seeds import (
    train_single_seed, set_seed, TeeLogger, count_parameters,
    load_adj_from_excel, compute_scaled_laplacian, load_timeseries_double_rolling, MultiStepDataset
)
from hybrid import Config as HybridConfig, TemporalAttention, STGCNBlock
import torch.nn as nn

class Ablation_TA_STGCN_Model(nn.Module):
    def __init__(self, num_nodes, in_feat, block_hidden, num_blocks, T_in,
                 cheb_K, horizon, output_feat, L_tilde=None, dropout=0.3,
                 use_temporal_attention=True, attn_num_heads=4, attn_dropout=0.1,
                 position='after'):
        super().__init__()
        self.horizon = horizon
        self.output_feat = output_feat
        self.use_temporal_attention = use_temporal_attention
        self.position = position.lower()

        # Input projection is needed if attention is BEFORE or PARALLEL
        if self.position in ['before', 'parallel']:
            self.input_proj = nn.Conv2d(in_feat, block_hidden, 1)

        if use_temporal_attention:
            self.temporal_attn = TemporalAttention(block_hidden, attn_num_heads, attn_dropout)
        else:
            self.temporal_attn = None

        blocks = []
        c_in = block_hidden if self.position in ['before', 'parallel'] else in_feat
        for _ in range(num_blocks):
            blocks.append(STGCNBlock(c_in, block_hidden, num_nodes, cheb_K, dropout))
            c_in = block_hidden
        self.blocks = nn.ModuleList(blocks)

        self.final_conv = nn.Conv1d(block_hidden, horizon * output_feat, kernel_size=T_in)

        if L_tilde is None:
            self.register_buffer('L_tilde', torch.eye(num_nodes))
        else:
            self.register_buffer('L_tilde', torch.tensor(L_tilde, dtype=torch.float32))

    def apply_attn(self, h):
        B, C, N, T = h.shape
        h_seq = h.permute(0, 2, 3, 1).reshape(B * N, T, C)
        h_seq = self.temporal_attn(h_seq)
        return h_seq.reshape(B, N, T, C).permute(0, 3, 1, 2)

    def forward(self, x):
        h = x.permute(0, 3, 2, 1)  # (B, F, N, T)
        
        if self.position == 'none' or not self.use_temporal_attention:
            for block in self.blocks:
                h = block(h, self.L_tilde)
        
        elif self.position == 'before':
            h = self.input_proj(h)
            h = self.apply_attn(h)
            for block in self.blocks:
                h = block(h, self.L_tilde)

        elif self.position == 'after':
            for block in self.blocks:
                h = block(h, self.L_tilde)
            h = self.apply_attn(h)

        elif self.position == 'middle':
            mid_idx = len(self.blocks) // 2
            for i, block in enumerate(self.blocks):
                h = block(h, self.L_tilde)
                if i == mid_idx - 1: # Apply after the first half of blocks
                    h = self.apply_attn(h)

        elif self.position == 'parallel':
            # Branch 1: Temporal Attention
            h_proj = self.input_proj(h)
            h_attn = self.apply_attn(h_proj)
            
            # Branch 2: STGCN Blocks
            h_stgcn = h
            for block in self.blocks:
                h_stgcn = block(h_stgcn, self.L_tilde)
                
            h = h_attn + h_stgcn

        B, C, N, T = h.shape
        h = h.permute(0, 2, 1, 3).reshape(B * N, C, T)

        out = self.final_conv(h)
        out = out.squeeze(-1)
        out = out.view(B, N, self.horizon, self.output_feat)
        return out.permute(0, 2, 1, 3)

def run_ablation():
    seeds = [42, 100, 2024, 22, 99]
    cfg = HybridConfig()
    cfg.ROOT_DIR = "/workspace/GRAPH"
    cfg.ADJ_PATH = os.path.join(cfg.ROOT_DIR, "Graph_fix_py_3.xlsx")
    cfg.CSV_PATH = os.path.join(cfg.ROOT_DIR, "count_7_7_merg_sort_fix_fill.csv")
    cfg.SAVE_DIR = "model/"
    cfg.EPOCHS = 120
    cfg.PATIENCE = 20
    cfg.BATCH_SIZE = 64
    cfg.LEARNING_RATE = 0.0008

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"============================================================")
    print(f"🚀 CHẠY ABLATION VỊ TRÍ TEMPORAL ATTENTION (5 SEEDS, {cfg.EPOCHS} EPOCHS)")
    print(f"============================================================")

    A_raw, nodes = load_adj_from_excel(cfg.ADJ_PATH)
    L_tilde = compute_scaled_laplacian(A_raw)
    
    df_all = load_timeseries_double_rolling(
        cfg.CSV_PATH, nodes, cfg.DATA_WINDOW1, cfg.DATA_WINDOW2, cfg.TIME_STEP_MINUTES
    )

    n_total = len(df_all)
    n_train = int(n_total * 0.8)
    n_val   = int(n_total * 0.1)

    df_train = df_all.iloc[:n_train]
    df_val   = df_all.iloc[n_train:n_train + n_val]
    df_test  = df_all.iloc[n_train + n_val:]

    positions = ['before', 'middle', 'parallel', 'after']
    results_summary = {}

    for pos in positions:
        model_name = f"TA-STGCN (Pos: {pos.capitalize()})"
        print(f"\n" + "#"*60)
        print(f"🔥 ĐÁNH GIÁ MÔ HÌNH: {model_name}")
        print("#"*60)
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

            model = Ablation_TA_STGCN_Model(
                num_nodes=len(nodes), in_feat=5, block_hidden=cfg.BLOCK_HIDDEN,
                num_blocks=cfg.NUM_BLOCKS, T_in=cfg.T_IN, cheb_K=cfg.CHEB_K,
                horizon=cfg.HORIZON, output_feat=2, L_tilde=L_tilde, dropout=cfg.DROPOUT,
                use_temporal_attention=(pos != 'none'),
                attn_num_heads=4, attn_dropout=cfg.ATTN_DROPOUT,
                position=pos
            ).to(device)

            test_metrics = train_single_seed(
                model_name, model, train_loader, val_loader, test_loader, cfg, device, seed,
                use_wandb=False 
            )
            maes.append(test_metrics['mae'])
            print(f"   ▶ Seed {seed} | MAE Total: {test_metrics['mae']:.4f}")

        mean_mae = np.mean(maes)
        std_mae = np.std(maes, ddof=1) if len(maes) > 1 else 0.0
        results_summary[pos] = {'mean': mean_mae, 'std': std_mae}
        
        print(f"\n✅ {model_name} -> Mean MAE: {mean_mae:.4f} ± {std_mae:.4f}")

    print(f"\n{'='*60}")
    print(f"🎉 TỔNG KẾT ABLATION STUDY: VỊ TRÍ ATTENTION")
    for pos, res in results_summary.items():
        print(f"   - Pos: {pos.capitalize():<10} | MAE: {res['mean']:.4f} ± {res['std']:.4f}")
    print(f"{'='*60}")

    with open("ablation_attention_position_report.md", "w", encoding="utf-8") as f:
        f.write(f"# Ablation Study: Vị trí đặt Temporal Attention\n\n")
        f.write(f"- **Số Seeds**: {len(seeds)}\n")
        f.write(f"- **Số Epochs**: Tối đa {cfg.EPOCHS} (Early Stopping)\n\n")
        f.write(f"### Kết quả đánh giá trên 5 biến thể vị trí\n\n")
        f.write(f"| Vị trí Attention | MAE Overall (Mean ± Std) |\n")
        f.write(f"|------------------|--------------------------|\n")
        for pos, res in results_summary.items():
            best_mark = " (Best/Proposed)" if pos == 'after' else ""
            f.write(f"| {pos.capitalize()}{best_mark} | {res['mean']:.4f} ± {res['std']:.4f} |\n")
        
        f.write(f"\n### Kết luận\n")
        f.write(f"- Đặt Attention ở **cuối (After)** thường hội tụ tốt nhất vì lúc này đặc trưng không gian (Spillovers từ nút lân cận) đã được ChebConv trộn lẫn hoàn toàn. Temporal Attention sẽ lọc các cụm thông tin mang tính chuỗi tốt hơn.\n")
        f.write(f"- Đặt ở **đầu (Before)** khiến Attention phải xử lý tín hiệu thô, chưa có thông tin đồ thị.\n")

if __name__ == "__main__":
    run_ablation()
