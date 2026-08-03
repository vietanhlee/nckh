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
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dotenv import load_dotenv

# Import kiến trúc TA-STGCN từ hybrid.py
from hybrid import STGCN_Model as Hybrid_STGCN_Model, Config as HybridConfig
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


def count_parameters(model):
    """Đếm tổng số tham số có thể huấn luyện (Trainable Parameters) của mô hình."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, dummy_input):
    """Đếm số lượng phép tính FLOPs (GFLOPs) cho 1 batch đầu vào."""
    try:
        import thop
        flops, _ = thop.profile(model, inputs=(dummy_input,), verbose=False)
        return flops / 1e9
    except Exception:
        params = count_parameters(model)
        B, T, N, F = dummy_input.shape
        approx_flops = 2 * params * T * N
        return approx_flops / 1e9


def measure_inference_latency(model, loader, device, max_batches=20):
    """Đo độ trễ suy luận (Inference Latency) ms/batch."""
    model.eval()
    with torch.no_grad():
        for i, (X, Y) in enumerate(loader):
            if i >= 3:
                break
            X = X.to(device)
            _ = model(X)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    start_time = time.time()
    count = 0
    with torch.no_grad():
        for i, (X, Y) in enumerate(loader):
            if i >= max_batches:
                break
            X = X.to(device)
            _ = model(X)
            count += 1

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed_ms = (time.time() - start_time) * 1000.0
    return elapsed_ms / max(1, count)


def train_single_ablation_variant(variant_name, model, train_loader, val_loader, test_loader, cfg, device, seed):
    """Huấn luyện và đánh giá 1 biến thể Ablation Study."""
    criterion = PureHuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg.LR_DECAY_FACTOR, patience=cfg.LR_PATIENCE
    )

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_weights = copy.deepcopy(model.state_dict())

    print(f"\n⚡ [{variant_name}] Seed {seed} | Bắt đầu huấn luyện (Max Epochs: {cfg.EPOCHS}, Patience: {cfg.PATIENCE})...")

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

        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, Y_batch)
                val_loss += loss.item() * len(X_batch)
                val_mae  += torch.abs(pred - Y_batch).mean().item() * len(X_batch)

        val_loss /= len(val_loader.dataset)
        val_mae  /= len(val_loader.dataset)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if epoch % 50 == 0 or patience_counter == cfg.PATIENCE:
            print(f"   Epoch {epoch:>3d}/{cfg.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}")

        if patience_counter >= cfg.PATIENCE:
            print(f"🛑 [Early Stopping] Dừng ở Epoch {epoch} dựa trên Validation Loss.")
            break

    # Load trọng số tốt nhất để đánh giá trên Test set
    model.load_state_dict(best_model_weights)
    model.eval()

    test_mae, test_rmse, test_mse = 0.0, 0.0, 0.0
    step_maes = [0.0] * 6
    total_samples = 0

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            pred = model(X_batch)
            bs = len(X_batch)
            total_samples += bs

            err = pred - Y_batch
            abs_err = torch.abs(err)

            test_mae  += abs_err.mean().item() * bs
            test_mse  += (err ** 2).mean().item() * bs

            for t_idx in range(6):
                step_maes[t_idx] += abs_err[:, t_idx, :, :].mean().item() * bs

    test_mae /= total_samples
    test_mse /= total_samples
    test_rmse = np.sqrt(test_mse)
    step_maes = [s / total_samples for s in step_maes]

    metrics = {
        'mae': test_mae,
        'rmse': test_rmse,
        'mse': test_mse
    }
    for t_idx in range(6):
        metrics[f'mae_t{t_idx+1}'] = step_maes[t_idx]

    return metrics


def run_ablation_benchmark():
    parser = argparse.ArgumentParser(description="Script huấn luyện Ablation Study cho các biến thể TA-STGCN.")
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 100, 2024],
                        help="Danh sách các seeds thử nghiệm Ablation Study.")
    parser.add_argument('--epochs', type=int, default=500, help="Số epochs tối đa.")
    parser.add_argument('--patience', type=int, default=30, help="Early stopping patience.")
    parser.add_argument('--batch_size', type=int, default=32, help="Kích thước batch_size.")
    parser.add_argument('--root_dir', type=str, default="/kaggle/input/datasets/canhdoo/nckh-traffic/GRAPH",
                        help="Thư mục gốc chứa dữ liệu.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_cfg = HybridConfig()
    base_cfg.ROOT_DIR = args.root_dir
    base_cfg.ADJ_PATH = os.path.join(args.root_dir, "Graph_fix_py_3.xlsx")
    base_cfg.CSV_PATH = os.path.join(args.root_dir, "count_7_7_merg_sort_fix_fill.csv")
    base_cfg.EPOCHS = args.epochs
    base_cfg.PATIENCE = args.patience
    base_cfg.BATCH_SIZE = args.batch_size

    print("\n[1] Nạp ma trận đồ thị và dữ liệu...")
    A_raw, nodes = load_adj_from_excel(base_cfg.ADJ_PATH)
    L_tilde = compute_scaled_laplacian(A_raw)

    df_all = load_timeseries_double_rolling(
        base_cfg.CSV_PATH, nodes, base_cfg.DATA_WINDOW1, base_cfg.DATA_WINDOW2, base_cfg.TIME_STEP_MINUTES
    )

    n_total = len(df_all)
    n_train = int(n_total * 0.8)
    n_val   = int(n_total * 0.1)

    df_train = df_all.iloc[:n_train]
    df_val   = df_all.iloc[n_train:n_train + n_val]
    df_test  = df_all.iloc[n_train + n_val:]

    # Đăng ký 4 biến thể Ablation Study của TA-STGCN
    ablation_registry = {
        'TA-STGCN (Full Model)': {
            'build_fn': lambda cfg: Hybrid_STGCN_Model(
                num_nodes=len(nodes), in_feat=4, block_hidden=64,
                num_blocks=2, T_in=cfg.T_IN, cheb_K=3, horizon=cfg.HORIZON,
                output_feat=1, L_tilde=L_tilde, dropout=0.1,
                use_temporal_attention=True, attn_num_heads=4, attn_dropout=0.1
            )
        },
        'TA-STGCN w/o Temporal Attention': {
            'build_fn': lambda cfg: Hybrid_STGCN_Model(
                num_nodes=len(nodes), in_feat=4, block_hidden=64,
                num_blocks=2, T_in=cfg.T_IN, cheb_K=3, horizon=cfg.HORIZON,
                output_feat=1, L_tilde=L_tilde, dropout=0.1,
                use_temporal_attention=False, attn_num_heads=4, attn_dropout=0.1
            )
        },
        'TA-STGCN w/ Single-Head Attn (h=1)': {
            'build_fn': lambda cfg: Hybrid_STGCN_Model(
                num_nodes=len(nodes), in_feat=4, block_hidden=64,
                num_blocks=2, T_in=cfg.T_IN, cheb_K=3, horizon=cfg.HORIZON,
                output_feat=1, L_tilde=L_tilde, dropout=0.1,
                use_temporal_attention=True, attn_num_heads=1, attn_dropout=0.1
            )
        },
        'TA-STGCN w/ Light Hidden Dim (C=32)': {
            'build_fn': lambda cfg: Hybrid_STGCN_Model(
                num_nodes=len(nodes), in_feat=4, block_hidden=32,
                num_blocks=2, T_in=cfg.T_IN, cheb_K=3, horizon=cfg.HORIZON,
                output_feat=1, L_tilde=L_tilde, dropout=0.1,
                use_temporal_attention=True, attn_num_heads=4, attn_dropout=0.1
            )
        }
    }

    results = {
        v_name: {
            'params': 0, 'flops_gflops': 0.0, 'inf_latencies': [],
            'maes': [], 'rmses': [], 'mses': [],
            'step_maes': {f't{t_idx+1}': [] for t_idx in range(6)}
        } for v_name in ablation_registry
    }

    for seed in args.seeds:
        set_seed(seed)
        print(f"\n{'='*65}")
        print(f"🧪 [ABLATION STUDY - SEED {seed}] THỬ NGHIỆM 4 BIẾN THỂ ARCHITECTURE")
        print(f"{'='*65}")

        for v_name, info in ablation_registry.items():
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            train_ds = MultiStepDataset(df_train, nodes, base_cfg.T_IN, base_cfg.HORIZON)
            scaler   = {'mean': train_ds.means, 'std': train_ds.stds}
            val_ds   = MultiStepDataset(df_val, nodes, base_cfg.T_IN, base_cfg.HORIZON, scaler)
            test_ds  = MultiStepDataset(df_test, nodes, base_cfg.T_IN, base_cfg.HORIZON, scaler)

            eval_batch_size = min(base_cfg.BATCH_SIZE, 32)
            train_loader = DataLoader(train_ds, batch_size=base_cfg.BATCH_SIZE, shuffle=True)
            val_loader   = DataLoader(val_ds, batch_size=eval_batch_size)
            test_loader  = DataLoader(test_ds, batch_size=eval_batch_size)

            model = info['build_fn'](base_cfg).to(device)

            params_count = count_parameters(model)
            results[v_name]['params'] = params_count

            dummy_x, _ = next(iter(test_loader))
            dummy_x = dummy_x.to(device)
            gflops = count_flops(model, dummy_x)
            results[v_name]['flops_gflops'] = gflops

            test_metrics = train_single_ablation_variant(
                v_name, model, train_loader, val_loader, test_loader, base_cfg, device, seed
            )

            inf_latency = measure_inference_latency(model, test_loader, device)
            results[v_name]['inf_latencies'].append(inf_latency)
            results[v_name]['maes'].append(test_metrics['mae'])
            results[v_name]['rmses'].append(test_metrics['rmse'])
            results[v_name]['mses'].append(test_metrics['mse'])

            for t_idx in range(6):
                results[v_name]['step_maes'][f't{t_idx+1}'].append(test_metrics[f'mae_t{t_idx+1}'])

            print(f"   ▶ Seed {seed:>4} | {v_name:<35} (Params: {params_count:,} | FLOPs: {gflops:.3f} GFLOPs) -> "
                  f"MAE: {test_metrics['mae']:.4f}")

            del model
            torch.cuda.empty_cache()
            gc.collect()

    table_data = []
    for v_name, res in results.items():
        maes, rmses = res['maes'], res['rmses']
        inf_lats = res['inf_latencies']
        p_count = res['params']
        flops_g = res['flops_gflops']

        row = {
            'Ablation Variant': v_name,
            'Params': f"{p_count:,}",
            'FLOPs (GFLOPs)': f"{flops_g:.3f}",
            'Inf Latency (ms)': f"{np.mean(inf_lats):.2f} ± {np.std(inf_lats):.2f}",
            'MAE Overall': f"{np.mean(maes):.4f} ± {np.std(maes):.4f}",
            'RMSE Overall': f"{np.mean(rmses):.4f} ± {np.std(rmses):.4f}"
        }
        table_data.append(row)

    summary_df = pd.DataFrame(table_data)
    print(f"\n{'='*90}")
    print(f"🏆 BẢNG KẾT QUẢ ABLATION STUDY (MEAN ± STD QUA {len(args.seeds)} SEEDS)")
    print(f"{'='*90}")
    print(summary_df.to_string(index=False))

    report_path = "ablation_study_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# 🧪 Báo cáo Thực nghiệm Ablation Study ({len(args.seeds)} Seeds)\n\n")
        f.write(f"- **Seeds**: `{args.seeds}`\n")
        f.write(f"- **Mô hình gốc**: TA-STGCN (2 ST-Conv blocks, $C=64$, $h=4$ heads)\n\n")
        f.write("## 🏆 Bảng Kết quả So sánh Ablation Study\n\n")
        f.write(summary_df.to_markdown(index=False))

    print(f"\n📑 Đã lưu báo cáo Ablation Study vào: {report_path}")

if __name__ == "__main__":
    run_ablation_benchmark()
