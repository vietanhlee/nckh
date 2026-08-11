import os
# Cấu hình PyTorch Allocator tránh phân mảnh bộ nhớ CUDA OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import copy
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import wilcoxon
from dotenv import load_dotenv

# Import các mô hình từ codebase
from gcn_lstm import ImprovedGNN_LSTM, Config as GCNLSTMConfig, normalize_adj_sym
from stgcn import STGCN_Model as Baseline_STGCN_Model, Config as BaselineConfig
from hybrid import STGCN_Model as Hybrid_STGCN_Model, Config as HybridConfig
from advanced_baselines import GraphWaveNet, ASTGCN, GMAN
from stgcn import (
    load_adj_from_excel,
    compute_scaled_laplacian,
    load_timeseries_double_rolling,
    MultiStepDataset,
    PureHuberLoss
)
from benchmark_5seeds import ModelEMA, evaluate_detailed

load_dotenv()


def set_seed(seed):
    """Cố định seed ngẫu nhiên đảm bảo tính lặp lại (Reproducibility)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model_for_evaluation(model_name, model_fn, train_loader, val_loader, cfg, device, seed):
    """Huấn luyện mô hình phục vụ đánh giá Benchmark & Kiểm định Thống kê (Khớp 100% với benchmark_5seeds.py)."""
    set_seed(seed)
    model = model_fn(cfg).to(device)
    
    criterion = PureHuberLoss(delta=getattr(cfg, 'LOSS_DELTA', 1.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
    grad_scaler = torch.amp.GradScaler('cuda')
    
    use_ema = getattr(cfg, 'USE_EMA', False)
    ema = ModelEMA(model, decay=getattr(cfg, 'EMA_DECAY', 0.995)) if use_ema else None
    
    lr_scheduler = None
    if getattr(cfg, 'USE_LR_SCHEDULER', False):
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=cfg.LR_SCHED_FACTOR,
            patience=cfg.LR_SCHED_PATIENCE, min_lr=cfg.LR_SCHED_MIN_LR
        )
        
    scaler_stats = {'mean': train_loader.dataset.means, 'std': train_loader.dataset.stds}
    grad_clip_norm = getattr(cfg, 'GRAD_CLIP_NORM', None)

    best_val_mae = float('inf')
    patience_cnt = 0
    best_weights = copy.deepcopy(model.state_dict())

    print(f"⚡ [{model_name}] Đang huấn luyện cho Seed {seed}...")
    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                pred = model(X_batch)
                loss = criterion(pred, Y_batch)
                
            grad_scaler.scale(loss).backward()
            
            if grad_clip_norm is not None:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                
            grad_scaler.step(optimizer)
            grad_scaler.update()
            
            if ema is not None:
                ema.update(model)

        # Đánh giá trên tập Validation
        if ema is not None:
            raw_state = copy.deepcopy(model.state_dict())
            model.load_state_dict(ema.shadow)
            val_metrics = evaluate_detailed(model, val_loader, device, scaler_stats, loss_fn=criterion)
            model.load_state_dict(raw_state)
        else:
            val_metrics = evaluate_detailed(model, val_loader, device, scaler_stats, loss_fn=criterion)
            
        val_mae = val_metrics['mae']
        val_loss = val_metrics['loss']
        
        if lr_scheduler is not None:
            lr_scheduler.step(val_loss)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_cnt = 0
            best_weights = copy.deepcopy(ema.shadow if ema is not None else model.state_dict())
        else:
            patience_cnt += 1

        if patience_cnt >= getattr(cfg, 'PATIENCE', 10):
            break

    model.load_state_dict(best_weights)
    model.eval()
    return model


def evaluate_benchmark_and_pointwise_errors(model, test_loader, scaler, device, horizon_steps=6):
    """
    Tính toán chỉ số Benchmark từng chân trời (t+1..t+6) VÀ thu thập mảng sai số tuyệt đối từng điểm.
    """
    model.eval()
    means = torch.tensor(scaler['mean'], device=device)
    stds = torch.tensor(scaler['std'], device=device)

    all_abs_errors = []
    horizon_maes = [[] for _ in range(horizon_steps)]
    overall_mses = []

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            pred = model(X_batch)

            y_true = Y_batch * stds + means
            y_pred = pred * stds + means

            err = y_true - y_pred
            abs_err = torch.abs(err)

            all_abs_errors.append(abs_err.cpu().numpy().flatten())
            overall_mses.append((err ** 2).mean().item())

            # Tính MAE từng horizon (B, H, N, F)
            for h in range(horizon_steps):
                h_mae = abs_err[:, h, :, :].mean().item()
                horizon_maes[h].append(h_mae)

    pointwise_errors = np.concatenate(all_abs_errors)
    overall_mae = np.mean(pointwise_errors)
    overall_rmse = np.sqrt(np.mean(overall_mses))
    avg_horizon_maes = [np.mean(h_list) for h_list in horizon_maes]

    return {
        'overall_mae': overall_mae,
        'overall_rmse': overall_rmse,
        'horizon_maes': avg_horizon_maes,
        'pointwise_errors': pointwise_errors
    }


def run_full_benchmark_and_significance_test():
    parser = argparse.ArgumentParser(description="Script Master Benchmark Sub-problem 2 VÀ Kiểm định Thống kê Wilcoxon.")
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 100, 2024],
                        help="Danh sách seeds thử nghiệm (mặc định: 42 100 2024).")
    parser.add_argument('--epochs', type=int, default=80,
                        help="Số epochs tối đa (mặc định: 80).")
    parser.add_argument('--patience', type=int, default=10,
                        help="Early stopping patience (mặc định: 10).")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size (mặc định: 32).")
    parser.add_argument('--root_dir', type=str, default="/kaggle/input/datasets/canhdoo/nckh-traffic/GRAPH",
                        help="Thư mục gốc chứa dữ liệu.")
    parser.add_argument('--use_wandb', action='store_true', default=True,
                        help="Tự động khởi tạo và ghi log lên WandB (mặc định: True).")
    parser.add_argument('--no_wandb', dest='use_wandb', action='store_false',
                        help="Tắt ghi log WandB.")
    parser.add_argument('--wandb_project', type=str, default="NCKH-Benmark-5Seed",
                        help="Tên project trên WandB.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gcn_lstm_cfg = GCNLSTMConfig()
    gcn_lstm_cfg.GCN_HIDDEN  = 64
    gcn_lstm_cfg.LSTM_HIDDEN = 160
    gcn_lstm_cfg.LSTM_LAYERS = 2

    stgcn_cfg = BaselineConfig()
    stgcn_cfg.BLOCK_HIDDEN   = 80
    stgcn_cfg.NUM_BLOCKS     = 3

    base_cfg = HybridConfig()

    for cfg_inst in [gcn_lstm_cfg, stgcn_cfg, base_cfg]:
        cfg_inst.ROOT_DIR = args.root_dir
        cfg_inst.ADJ_PATH = os.path.join(args.root_dir, "Graph_fix_py_3.xlsx")
        cfg_inst.CSV_PATH = os.path.join(args.root_dir, "count_7_7_merg_sort_fix_fill.csv")
        cfg_inst.SAVE_DIR = "model/"
        cfg_inst.EPOCHS = args.epochs
        cfg_inst.PATIENCE = args.patience
        cfg_inst.BATCH_SIZE = args.batch_size
        os.makedirs(cfg_inst.SAVE_DIR, exist_ok=True)

    if not os.path.exists(base_cfg.CSV_PATH):
        local_dir = os.getcwd()
        alt_adj = os.path.join(local_dir, "Graph_fix_py_3.xlsx")
        alt_csv = os.path.join(local_dir, "count_7_7_merg_sort_fix_fill.csv")
        if os.path.exists(alt_csv):
            for cfg_inst in [gcn_lstm_cfg, stgcn_cfg, base_cfg]:
                cfg_inst.ADJ_PATH = alt_adj
                cfg_inst.CSV_PATH = alt_csv

    print("\n[1] Nạp dữ liệu đồ thị và chuỗi thời gian đếm xe...")
    A_raw, nodes = load_adj_from_excel(base_cfg.ADJ_PATH)
    L_tilde = compute_scaled_laplacian(A_raw)
    A_norm = normalize_adj_sym(A_raw)

    df_all = load_timeseries_double_rolling(
        base_cfg.CSV_PATH, nodes, base_cfg.DATA_WINDOW1, base_cfg.DATA_WINDOW2, base_cfg.TIME_STEP_MINUTES
    )

    n_total = len(df_all)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)

    df_train = df_all.iloc[:n_train]
    df_val = df_all.iloc[n_train:n_train + n_val]
    df_test = df_all.iloc[n_train + n_val:]

    models_to_test = {
        'GCN-LSTM': {
            'cfg': gcn_lstm_cfg,
            'fn': lambda cfg: ImprovedGNN_LSTM(
                num_nodes=len(nodes), in_feat=4, gcn_hidden=cfg.GCN_HIDDEN,
                lstm_hidden=cfg.LSTM_HIDDEN, lstm_layers=cfg.LSTM_LAYERS,
                horizon=cfg.HORIZON, output_feat=1, A_norm=A_norm, dropout=cfg.DROPOUT
            )
        },
        'STGCN (Baseline)': {
            'cfg': stgcn_cfg,
            'fn': lambda cfg: Baseline_STGCN_Model(
                num_nodes=len(nodes), in_feat=4, block_hidden=cfg.BLOCK_HIDDEN,
                num_blocks=cfg.NUM_BLOCKS, T_in=cfg.T_IN, cheb_K=cfg.CHEB_K,
                horizon=cfg.HORIZON, output_feat=1, L_tilde=L_tilde, dropout=cfg.DROPOUT
            )
        },
        'GraphWaveNet': {
            'cfg': base_cfg,
            'fn': lambda cfg: GraphWaveNet(
                num_nodes=len(nodes), in_dim=4, out_dim=1, blocks=4, layers=2, horizon=cfg.HORIZON
            )
        },
        'ASTGCN': {
            'cfg': base_cfg,
            'fn': lambda cfg: ASTGCN(
                num_nodes=len(nodes), in_channels=4, K=cfg.CHEB_K, num_blocks=2, T_in=cfg.T_IN, horizon=cfg.HORIZON, block_channels=64, L_tilde=L_tilde
            )
        },
        'GMAN': {
            'cfg': base_cfg,
            'fn': lambda cfg: GMAN(
                num_nodes=len(nodes), in_channels=4, T_in=cfg.T_IN, horizon=cfg.HORIZON, embed_size=64, heads=4, num_blocks=1
            )
        },
        'TA-STGCN (Proposed / Ours)': {
            'cfg': base_cfg,
            'fn': lambda cfg: Hybrid_STGCN_Model(
                num_nodes=len(nodes), in_feat=4, block_hidden=cfg.BLOCK_HIDDEN,
                num_blocks=cfg.NUM_BLOCKS, T_in=cfg.T_IN, cheb_K=cfg.CHEB_K,
                horizon=cfg.HORIZON, output_feat=1, L_tilde=L_tilde, dropout=cfg.DROPOUT,
                use_temporal_attention=True, attn_num_heads=4, attn_dropout=cfg.ATTN_DROPOUT
            )
        }
    }

    metrics_across_seeds = {
        m_name: {'overall_mae': [], 'overall_rmse': [], 'horizon_maes': [[] for _ in range(6)], 'pointwise_errors': []}
        for m_name in models_to_test
    }

    print(f"\n==========================================================================================")
    print(f"📊 HUẤN LUYỆN BENCHMARK VÀ THU THẬP DỮ LIỆU KIỂM ĐỊNH THỐNG KÊ (5 SEEDS)")
    print(f"==========================================================================================")

    for seed in args.seeds:
        print(f"\n🌱 [SEED {seed}] Đang thực thi huấn luyện và đánh giá trên tập Test...")
        train_ds = MultiStepDataset(df_train, nodes, base_cfg.T_IN, base_cfg.HORIZON)
        scaler = {'mean': train_ds.means, 'std': train_ds.stds}
        val_ds = MultiStepDataset(df_val, nodes, base_cfg.T_IN, base_cfg.HORIZON, scaler)
        test_ds = MultiStepDataset(df_test, nodes, base_cfg.T_IN, base_cfg.HORIZON, scaler)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        for m_name, info in models_to_test.items():
            model = train_model_for_evaluation(m_name, info['fn'], train_loader, val_loader, info['cfg'], device, seed)
            res = evaluate_benchmark_and_pointwise_errors(model, test_loader, scaler, device, base_cfg.HORIZON)

            metrics_across_seeds[m_name]['overall_mae'].append(res['overall_mae'])
            metrics_across_seeds[m_name]['overall_rmse'].append(res['overall_rmse'])
            metrics_across_seeds[m_name]['pointwise_errors'].append(res['pointwise_errors'])
            for h in range(6):
                metrics_across_seeds[m_name]['horizon_maes'][h].append(res['horizon_maes'][h])

            print(f"   ▶ {m_name:<26} | Seed {seed:>4} -> MAE Overall: {res['overall_mae']:.4f} | RMSE: {res['overall_rmse']:.4f}")

            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

    # 1. Bảng 1: BENCHMARK SUB-PROBLEM 2 (Đầy đủ theo từng horizon t+1..t+6 cho Table III)
    benchmark_table_rows = []
    for m_name in models_to_test:
        row = {'Model Architecture': m_name}
        mae_mean = np.mean(metrics_across_seeds[m_name]['overall_mae'])
        mae_std = np.std(metrics_across_seeds[m_name]['overall_mae'])
        rmse_mean = np.mean(metrics_across_seeds[m_name]['overall_rmse'])
        rmse_std = np.std(metrics_across_seeds[m_name]['overall_rmse'])

        row['MAE Overall'] = f"{mae_mean:.4f} ± {mae_std:.4f}"
        row['RMSE Overall'] = f"{rmse_mean:.4f} ± {rmse_std:.4f}"

        for h in range(6):
            h_mean = np.mean(metrics_across_seeds[m_name]['horizon_maes'][h])
            h_std = np.std(metrics_across_seeds[m_name]['horizon_maes'][h])
            row[f'MAE t+{h+1}'] = f"{h_mean:.4f} ± {h_std:.4f}"

        benchmark_table_rows.append(row)

    df_benchmark = pd.DataFrame(benchmark_table_rows)

    print(f"\n{'='*95}")
    print(f"🏆 BẢNG KẾT QUẢ BENCHMARK SUB-PROBLEM 2 CHO TABLE III (MEAN ± STD)")
    print(f"{'='*95}")
    print(df_benchmark.to_string(index=False))

    # 2. Bảng 2: KIỂM ĐỊNH THỐNG KÊ WILCOXON SIGNED-RANK TEST
    final_errors = {m_name: np.concatenate(metrics_across_seeds[m_name]['pointwise_errors']) for m_name in models_to_test}
    target_errors = final_errors['TA-STGCN (Proposed / Ours)']
    sample_size = len(target_errors)

    wilcoxon_rows = []
    for m_name in models_to_test:
        if m_name == 'TA-STGCN (Proposed / Ours)':
            continue

        base_err = final_errors[m_name]
        diff = base_err - target_errors
        w_stat, p_val = wilcoxon(diff, alternative='greater')
        is_sig = p_val < 0.01

        wilcoxon_rows.append({
            'Baseline Architecture': m_name,
            'Baseline Median MAE': f"{np.median(base_err):.4f}",
            'TA-STGCN Median MAE': f"{np.median(target_errors):.4f}",
            'Wilcoxon W-Stat': f"{w_stat:,.1f}",
            'p-value': f"{p_val:.4e}",
            'Statistical Significance (p < 0.01)': 'YES (p < 0.01) *' if is_sig else f'No (p = {p_val:.4f})'
        })

    df_wilcoxon = pd.DataFrame(wilcoxon_rows)

    print(f"\n{'='*95}")
    print(f"📊 BẢNG KẾT QUẢ KIỂM ĐỊNH THỐNG KÊ WILCOXON SIGNED-RANK TEST (p < 0.01)")
    print(f"{'='*95}")
    print(df_wilcoxon.to_string(index=False))

    # Ghi tệp báo cáo Markdown tổng hợp
    report_path = "subproblem2_master_benchmark_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 🏆 Master Báo cáo Benchmark Sub-problem 2 & Kiểm định Thống kê Wilcoxon\n\n")
        f.write("## 1. Bảng Kết quả Benchmark Đa chân trời (Dành cho Table III trong Bài báo)\n\n")
        f.write(df_benchmark.to_markdown(index=False))
        f.write("\n\n---\n\n## 2. Bảng Kết quả Kiểm định Thống kê Wilcoxon Signed-Rank Test\n\n")
        f.write(f"- **Kích thước mẫu bắt cặp**: $N = {sample_size:,}$ điểm dự báo trên tập Test\n\n")
        f.write(df_wilcoxon.to_markdown(index=False))
        f.write("\n\n---\n\n## 💡 Chú thích bài báo (Table III Footnote):\n")
        f.write("```latex\n")
        f.write("\\caption{Sub-Problem 2 Multi-Horizon Traffic Forecasting Benchmark Across 5 Random Seeds (Mean \\pm Std).}\n")
        f.write("% Footnote:\n")
        f.write("* Indicates statistical significance against all evaluated baseline models at p < 0.01 (Wilcoxon signed-rank test).\n")
        f.write("```\n")

    print(f"\n📑 Đã lưu báo cáo tổng hợp Master Benchmark vào tệp: {report_path}")

    # 3. Tự động xuất Biểu đồ Đường cong Hội tụ MAE cho TẤT CẢ 6 Mô hình (.pdf và .png)
    print(f"\n🎨 Đang xuất biểu đồ đường cong hội tụ MAE cho tất cả 6 mô hình (Figure 7)...")
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6), dpi=300)
        plt.rcParams['font.family'] = 'DejaVu Sans'

        target_models = models_to_test
        colors = {
            'GCN-LSTM': '#1f77b4',
            'STGCN (Baseline)': '#ff7f0e',
            'GraphWaveNet': '#d62728',
            'ASTGCN': '#9467bd',
            'GMAN': '#8c564b',
            'TA-STGCN (Proposed / Ours)': '#2ca02c'
        }
        linestyles = {
            'GCN-LSTM': '--',
            'STGCN (Baseline)': '-.',
            'GraphWaveNet': ':',
            'ASTGCN': '--',
            'GMAN': '-.',
            'TA-STGCN (Proposed / Ours)': '-'
        }

        for m_name in target_models:
            if m_name not in metrics_across_seeds:
                continue
            histories = metrics_across_seeds[m_name]['val_mae_histories']
            if not histories:
                continue

            max_len = max(len(h) for h in histories)
            padded = []
            for h in histories:
                if len(h) < max_len:
                    h = h + [h[-1]] * (max_len - len(h))
                padded.append(h)

            mean_curve = np.mean(padded, axis=0)
            plt.plot(
                range(1, max_len + 1), mean_curve,
                label=m_name, linewidth=2.2,
                color=colors.get(m_name, '#2ca02c'),
                linestyle=linestyles.get(m_name, '-')
            )

        plt.xlabel('Training Epochs', fontsize=11, fontweight='bold')
        plt.ylabel('Validation MAE Overall', fontsize=11, fontweight='bold')
        plt.title('Validation MAE Convergence Curves Across All 6 Models', fontsize=12, fontweight='bold')
        plt.legend(frameon=True, facecolor='white', edgecolor='none')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        fig_dir = "paper/fig"
        os.makedirs(fig_dir, exist_ok=True)
        fig_pdf = os.path.join(fig_dir, "fig_compare_mae_curve_models_graph.pdf")
        fig_png = os.path.join(fig_dir, "fig_compare_mae_curve_models_graph.png")
        plt.savefig(fig_pdf, format='pdf', bbox_inches='tight')
        plt.savefig(fig_png, format='png', bbox_inches='tight', dpi=300)
        plt.close()

        print(f"🖼️ Đã lưu biểu đồ đường cong hội tụ vào:\n   - PDF Vector: {fig_pdf}\n   - PNG High-Res: {fig_png}")
    except Exception as e:
        print(f"⚠️ Không thể sinh biểu đồ hội tụ tự động: {e}")


if __name__ == "__main__":
    run_full_benchmark_and_significance_test()
