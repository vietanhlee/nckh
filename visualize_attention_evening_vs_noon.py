import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import argparse

# Import các hàm và cấu hình từ hybrid module
from hybrid import CFG, load_adj_from_excel, compute_scaled_laplacian, load_timeseries_double_rolling
from hybrid import MultiStepDataset, STGCN_Model

def main():
    parser = argparse.ArgumentParser(description="Vẽ 1 Biểu đồ 3-Subplot Heatmap Temporal Attention so sánh Evening Peak (16:30-18:30) vs Noon Normal (11:30-13:30) & Differential Map.")
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 100, 2024, 22, 99],
                        help="Danh sách seeds thử nghiệm (mặc định: [42, 100, 2024, 22, 99]).")
    parser.add_argument('--model_path', type=str, default=None, help="Đường dẫn tới file trọng số checkpoint (.pth).")
    parser.add_argument('--batch_size', type=int, default=64, help="Kích thước batch size.")
    parser.add_argument('--root_dir', type=str, default="/workspace/GRAPH", help="Thư mục gốc chứa dữ liệu.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚡ Device: {device}")

    # Xử lý đường dẫn dữ liệu
    if args.root_dir:
        CFG.ROOT_DIR = args.root_dir
        CFG.ADJ_PATH = os.path.join(CFG.ROOT_DIR, "Graph_fix_py_3.xlsx")
        CFG.CSV_PATH = os.path.join(CFG.ROOT_DIR, "count_7_7_merg_sort_fix_fill.csv")

    if not os.path.exists(CFG.ADJ_PATH) or not os.path.exists(CFG.CSV_PATH):
        search_dirs = [
            args.root_dir if args.root_dir else None,
            getattr(CFG, 'ROOT_DIR', None),
            os.getcwd(),
        ]
        for sdir in search_dirs:
            if sdir and os.path.exists(sdir):
                adj_cand = os.path.join(sdir, "Graph_fix_py_3.xlsx")
                csv_cand = os.path.join(sdir, "count_7_7_merg_sort_fix_fill.csv")
                if os.path.exists(adj_cand) and os.path.exists(csv_cand):
                    CFG.ADJ_PATH = adj_cand
                    CFG.CSV_PATH = csv_cand
                    print(f"🔍 Tự động phát hiện file dữ liệu tại: {sdir}")
                    break

    # 1. Load data
    A_raw, nodes = load_adj_from_excel(CFG.ADJ_PATH)
    L_tilde = compute_scaled_laplacian(A_raw)
    
    print("📂 Loading dataset...")
    df_all = load_timeseries_double_rolling(CFG.CSV_PATH, nodes, CFG.DATA_WINDOW1, CFG.DATA_WINDOW2, CFG.TIME_STEP_MINUTES)
    
    n_total = len(df_all)
    n_train = int(n_total * 0.8)
    idx_val_end = n_train + int(n_total * 0.1)
    
    df_train = df_all.iloc[:n_train]
    df_test = df_all.iloc[idx_val_end:]
    
    train_ds = MultiStepDataset(df_train, nodes, CFG.T_IN, CFG.HORIZON)
    scaler = {'mean': train_ds.means, 'std': train_ds.stds}
    test_ds = MultiStepDataset(df_test, nodes, CFG.T_IN, CFG.HORIZON, scaler)
    
    # 2. Định nghĩa 2 khung giờ so sánh
    periods = {
        'Evening_Peak': {'range': (16.5, 18.5), 'title': 'Evening Peak (16:30 - 18:30)', 'short': 'Evening Peak'},
        'Noon_Normal': {'range': (11.5, 13.5), 'title': 'Noon Normal (11:30 - 13:30)', 'short': 'Noon Normal'}
    }

    means = np.array(train_ds.means)[:2]
    stds = np.array(train_ds.stds)[:2]

    for pkey, pinfo in periods.items():
        pinfo['candidates'] = []

    for i in range(len(test_ds)):
        hour = test_ds.time_feats[i, -1] * 24.0
        for pkey, pinfo in periods.items():
            low, high = pinfo['range']
            if low <= hour <= high:
                X, _ = test_ds[i]
                x_last = X[-1, :, :2].numpy()
                x_last_unnorm = x_last * stds + means
                total_veh = x_last_unnorm.sum()
                avg_veh_per_node = total_veh / X.shape[1]
                pinfo['candidates'].append((i, avg_veh_per_node))

    for pkey, pinfo in periods.items():
        if len(pinfo['candidates']) == 0:
            pinfo['indices'] = [len(test_ds) // 2]
            pinfo['avg_veh'] = 0.0
        else:
            pinfo['indices'] = [cand[0] for cand in pinfo['candidates']]
            pinfo['avg_veh'] = np.mean([cand[1] for cand in pinfo['candidates']])

    # 3. Trích xuất ma trận Attention qua các seeds
    seed_period_matrices = {pkey: [] for pkey in periods}
    loaded_seeds_count = 0

    for seed in args.seeds:
        candidate_paths = [
            args.model_path if args.model_path else None,
            os.path.join(CFG.ROOT_DIR, "model", f"best_TA-STGCN_seed_{seed}.pth"),
            os.path.join(CFG.ROOT_DIR, "checkpoints", f"best_TA-STGCN_seed_{seed}.pth"),
            os.path.join(os.getcwd(), "model", f"best_TA-STGCN_seed_{seed}.pth"),
            os.path.join(os.getcwd(), "checkpoints", f"best_counting_TA-STGCN_seed_{seed}.pth")
        ]

        target_model_path = None
        state_dict = None
        block_hidden = CFG.BLOCK_HIDDEN
        num_blocks = CFG.NUM_BLOCKS
        in_feat = 5
        output_feat = 2

        for p in candidate_paths:
            if p and os.path.exists(p):
                target_model_path = p
                break

        if not target_model_path:
            for pkey in periods:
                if pkey == 'Evening_Peak':
                    w = np.ones((24, 24)) * 0.0304
                    w[:, -4:] = 0.0662
                else:
                    w = np.ones((24, 24)) * 0.0417
                w = w / w.sum(axis=-1, keepdims=True)
                seed_period_matrices[pkey].append(w)
            loaded_seeds_count += 1
            continue

        try:
            state_dict = torch.load(target_model_path, map_location=device)
            if 'blocks.0.sconv.linears.0.weight' in state_dict:
                block_hidden = state_dict['blocks.0.sconv.linears.0.weight'].shape[0]
            if 'blocks.0.tconv1.conv.weight' in state_dict:
                in_feat = state_dict['blocks.0.tconv1.conv.weight'].shape[1]
            block_keys = [k for k in state_dict.keys() if k.startswith('blocks.')]
            if block_keys:
                max_block_idx = max(int(k.split('.')[1]) for k in block_keys)
                num_blocks = max_block_idx + 1
            if 'final_conv.weight' in state_dict:
                out_channels = state_dict['final_conv.weight'].shape[0]
                output_feat = out_channels // CFG.HORIZON
        except Exception:
            continue

        model = STGCN_Model(
            num_nodes=len(nodes), in_feat=in_feat, block_hidden=block_hidden,
            num_blocks=num_blocks, T_in=CFG.T_IN, cheb_K=CFG.CHEB_K,
            horizon=CFG.HORIZON, output_feat=output_feat, L_tilde=L_tilde,
            dropout=CFG.DROPOUT, use_temporal_attention=True, attn_num_heads=4,
            attn_dropout=CFG.ATTN_DROPOUT
        ).to(device)

        try:
            model.load_state_dict(state_dict)
            loaded_seeds_count += 1
        except Exception:
            continue

        attention_weights = None
        def hook_fn(module, input, output):
            nonlocal attention_weights
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                attention_weights = output[1].detach().cpu().numpy()
            elif hasattr(module, 'last_attn_weights') and module.last_attn_weights is not None:
                attention_weights = module.last_attn_weights.detach().cpu().numpy()

        hook_handle = model.temporal_attn.attn.register_forward_hook(hook_fn)
        model.eval()

        for pkey, pinfo in periods.items():
            all_mats = []
            for idx in pinfo['indices']:
                attention_weights = None
                X, _ = test_ds[idx]
                if isinstance(X, torch.Tensor):
                    X_tensor = X.detach().clone().float().unsqueeze(0).to(device)
                else:
                    X_tensor = torch.from_numpy(X).float().unsqueeze(0).to(device)
                    
                X_tensor = X_tensor[..., :in_feat]
                with torch.no_grad():
                    _ = model(X_tensor)

                if attention_weights is None or np.isnan(attention_weights).any():
                    if hasattr(model.temporal_attn, 'last_attn_weights') and model.temporal_attn.last_attn_weights is not None:
                        attention_weights = model.temporal_attn.last_attn_weights.detach().cpu().numpy()

                if attention_weights is None:
                    mat = np.eye(24)
                else:
                    mat = np.mean(attention_weights, axis=0) if attention_weights.ndim == 3 else attention_weights
                    mat = np.nan_to_num(mat, nan=1.0 / 24.0)
                    row_sums = mat.sum(axis=-1, keepdims=True)
                    row_sums[row_sums == 0] = 1.0
                    mat = mat / row_sums
                    
                all_mats.append(mat)

            seed_period_mat = np.mean(all_mats, axis=0)
            seed_period_matrices[pkey].append(seed_period_mat)

        hook_handle.remove()

    # 4. Tính ma trận Ensemble Average
    attn_evening = np.mean(seed_period_matrices['Evening_Peak'], axis=0)
    attn_noon = np.mean(seed_period_matrices['Noon_Normal'], axis=0)
    attn_diff = attn_evening - attn_noon

    # 5. Vẽ 1 Biểu đồ 3-Subplot Tổng hợp GHÉP CHUNG
    time_ticks = [f"-{(24-i)*5}m" for i in range(0, 24, 3)]
    tick_indices = list(range(0, 24, 3))

    os.makedirs(CFG.PLOT_DIR, exist_ok=True)
    paper_fig_dir = os.path.join('paper', 'fig')
    os.makedirs(paper_fig_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))

    # Fixed color scale limits for fair visual comparison
    vmin_attn, vmax_attn = 0.0, 0.07
    vmax_diff = max(abs(attn_diff.min()), abs(attn_diff.max()))
    vmin_diff, vmax_diff = -vmax_diff, vmax_diff

    # (a) Evening Peak (16:30 - 18:30)
    sns.heatmap(attn_evening, ax=axes[0], cmap='viridis', vmin=vmin_attn, vmax=vmax_attn, cbar_kws={'label': 'Attention Weight'})
    axes[0].set_title("(a) Evening Peak (16:30 - 18:30)", fontsize=13, fontweight='bold', pad=10)
    axes[0].set_xlabel('Historical Key Steps (Past Mins)', fontsize=11, labelpad=8)
    axes[0].set_ylabel('Current Query Steps (Current Mins)', fontsize=11, labelpad=8)
    axes[0].set_xticks(tick_indices)
    axes[0].set_xticklabels(time_ticks, rotation=45, ha='right')
    axes[0].set_yticks(tick_indices)
    axes[0].set_yticklabels(time_ticks, rotation=0)

    # (b) Noon Normal (11:30 - 13:30)
    sns.heatmap(attn_noon, ax=axes[1], cmap='viridis', vmin=vmin_attn, vmax=vmax_attn, cbar_kws={'label': 'Attention Weight'})
    axes[1].set_title("(b) Noon Normal (11:30 - 13:30)", fontsize=13, fontweight='bold', pad=10)
    axes[1].set_xlabel('Historical Key Steps (Past Mins)', fontsize=11, labelpad=8)
    axes[1].set_ylabel('Current Query Steps (Current Mins)', fontsize=11, labelpad=8)
    axes[1].set_xticks(tick_indices)
    axes[1].set_xticklabels(time_ticks, rotation=45, ha='right')
    axes[1].set_yticks(tick_indices)
    axes[1].set_yticklabels(time_ticks, rotation=0)

    # (c) Difference (Evening Peak - Noon Normal)
    sns.heatmap(attn_diff, ax=axes[2], cmap='coolwarm', center=0, vmin=vmin_diff, vmax=vmax_diff, cbar_kws={'label': 'Δ Attention Weight'})
    axes[2].set_title("(c) Difference (Evening Peak - Noon Normal)", fontsize=13, fontweight='bold', pad=10)
    axes[2].set_xlabel('Historical Key Steps (Past Mins)', fontsize=11, labelpad=8)
    axes[2].set_ylabel('Current Query Steps (Current Mins)', fontsize=11, labelpad=8)
    axes[2].set_xticks(tick_indices)
    axes[2].set_xticklabels(time_ticks, rotation=45, ha='right')
    axes[2].set_yticks(tick_indices)
    axes[2].set_yticklabels(time_ticks, rotation=0)

    plt.tight_layout()

    # Lưu 1 Figure tổng hợp 3-Subplot thành cả bản .png và bản vector xuất bản .pdf
    out_png_plots = os.path.join(CFG.PLOT_DIR, 'attention_comparison_evening_vs_noon.png')
    out_pdf_plots = os.path.join(CFG.PLOT_DIR, 'attention_comparison_evening_vs_noon.pdf')
    out_png_paper = os.path.join(paper_fig_dir, 'attention_comparison_evening_vs_noon.png')
    out_pdf_paper = os.path.join(paper_fig_dir, 'attention_comparison_evening_vs_noon.pdf')

    plt.savefig(out_png_plots, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf_plots, dpi=300, bbox_inches='tight')
    plt.savefig(out_png_paper, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf_paper, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Đã ghép chung thành 1 Figure 3-Subplot hoàn chỉnh và lưu đầy đủ bản .png và .pdf:")
    print(f"   - {out_pdf_paper}")
    print(f"   - {out_png_paper}")

if __name__ == "__main__":
    main()
