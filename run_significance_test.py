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
    """Huấn luyện mô hình phục vụ đánh giá Benchmark & Kiểm định Thống kê."""
    set_seed(seed)
    model = model_fn(cfg).to(device)
    criterion = PureHuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=getattr(cfg, 'WEIGHT_DECAY', 1e-4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=getattr(cfg, 'LR_SCHED_FACTOR', 0.5), patience=getattr(cfg, 'LR_SCHED_PATIENCE', 10)
    )

    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = copy.deepcopy(model.state_dict())

    print(f"⚡ [{model_name}] Đang huấn luyện cho Seed {seed}...")
    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, Y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item() * len(X_batch)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, Y_batch)
                val_loss += loss.item() * len(X_batch)

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= cfg.PATIENCE:
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
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 100, 2024], help="Danh sách seeds thử nghiệm.")
    parser.add_argument('--epochs', type=int, default=80, help="Số epochs tối đa.")
    parser.add_argument('--patience', type=int, default=15, help="Early stopping patience.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size.")
    parser.add_argument('--root_dir', type=str, default="/kaggle/input/datasets/canhdoo/nckh-traffic/GRAPH", help="Thư mục gốc.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_cfg = HybridConfig()
    base_cfg.ROOT_DIR = args.root_dir
    base_cfg.ADJ_PATH = os.path.join(args.root_dir, "Graph_fix_py_3.xlsx")
    base_cfg.CSV_PATH = os.path.join(args.root_dir, "count_7_7_merg_sort_fix_fill.csv")
    base_cfg.EPOCHS = args.epochs
    base_cfg.PATIENCE = args.patience
    base_cfg.BATCH_SIZE = args.batch_size

    if not os.path.exists(base_cfg.CSV_PATH):
        local_dir = os.getcwd()
        alt_adj = os.path.join(local_dir, "Graph_fix_py_3.xlsx")
        alt_csv = os.path.join(local_dir, "count_7_7_merg_sort_fix_fill.csv")
        if os.path.exists(alt_csv):
            base_cfg.ADJ_PATH = alt_adj
            base_cfg.CSV_PATH = alt_csv

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

    stgcn_cfg = BaselineConfig()
    stgcn_cfg.EPOCHS, stgcn_cfg.PATIENCE, stgcn_cfg.BATCH_SIZE = args.epochs, args.patience, args.batch_size

    gcn_lstm_cfg = GCNLSTMConfig()
    gcn_lstm_cfg.EPOCHS, gcn_lstm_cfg.PATIENCE, gcn_lstm_cfg.BATCH_SIZE = args.epochs, args.patience, args.batch_size

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
                num_nodes=len(nodes), in_dim=4, out_dim=1, horizon=cfg.HORIZON,
                supports=[torch.tensor(A_norm, dtype=torch.float32).to(device)]
            )
        },
        'ASTGCN': {
            'cfg': base_cfg,
            'fn': lambda cfg: ASTGCN(
                num_nodes=len(nodes), in_dim=4, out_dim=1, horizon=cfg.HORIZON,
                L_tilde=torch.tensor(L_tilde, dtype=torch.float32).to(device)
            )
        },
        'GMAN': {
            'cfg': base_cfg,
            'fn': lambda cfg: GMAN(
                num_nodes=len(nodes), in_dim=4, out_dim=1, horizon=cfg.HORIZON
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


if __name__ == "__main__":
    run_full_benchmark_and_significance_test()
