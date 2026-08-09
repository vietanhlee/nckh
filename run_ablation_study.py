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
api_key = os.getenv("WANDB_API_KEY")
use_wandb_default = False
if api_key:
    try:
        import wandb
        wandb.login(key=api_key)
        use_wandb_default = True
    except Exception as e:
        print(f"⚠️ WandB login lỗi: {e}")

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


def train_single_ablation_variant(variant_name, model, train_loader, val_loader, test_loader, cfg, device, seed, scaler, use_wandb=False, wandb_project="NCKH-Ablation-Study"):
    """Huấn luyện và đánh giá 1 biến thể Ablation Study."""
    criterion = PureHuberLoss(delta=1.0)
    weight_decay = getattr(cfg, 'WEIGHT_DECAY', 1e-4)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=weight_decay)
    
    lr_factor = getattr(cfg, 'LR_SCHED_FACTOR', getattr(cfg, 'LR_DECAY_FACTOR', 0.5))
    lr_patience = getattr(cfg, 'LR_SCHED_PATIENCE', getattr(cfg, 'LR_PATIENCE', 10))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_factor, patience=lr_patience
    )

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_weights = copy.deepcopy(model.state_dict())

    means = torch.tensor(scaler['mean'], device=device)
    stds = torch.tensor(scaler['std'], device=device)

    wandb_run = None
    if use_wandb:
        try:
            import wandb
            project_name = os.getenv('WANDB_PROJECT', wandb_project)
            wandb_run = wandb.init(
                project=project_name,
                name=f"Ablation_{variant_name}_seed_{seed}",
                config={
                    'variant_name': variant_name,
                    'seed': seed,
                    'epochs': cfg.EPOCHS,
                    'batch_size': cfg.BATCH_SIZE,
                    'learning_rate': cfg.LEARNING_RATE,
                    'patience': cfg.PATIENCE
                },
                reinit=True
            )
        except Exception as e:
            print(f"⚠️ Không thể khởi tạo WandB cho {variant_name} (Seed {seed}): {e}")
            wandb_run = None

    print(f"\n⚡ [{variant_name}] Seed {seed} | Bắt đầu huấn luyện (Max Epochs: {cfg.EPOCHS}, Patience: {cfg.PATIENCE})...")

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"   Epoch {epoch:02d}/{cfg.EPOCHS}", leave=False)
        for X_batch, Y_batch in pbar:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, Y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

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
                
                y_true = Y_batch * stds + means
                y_pred = pred * stds + means
                val_mae += torch.abs(y_pred - y_true).mean().item() * len(X_batch)

        val_loss /= len(val_loader.dataset)
        val_mae  /= len(val_loader.dataset)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_weights = copy.deepcopy(model.state_dict())
            is_best_str = " -> Saved Best"
        else:
            patience_counter += 1
            is_best_str = ""

        print(f"Ep {epoch:02d}/{cfg.EPOCHS} | Loss: {train_loss:.4f} / {val_loss:.4f} | Val MAE: {val_mae:.2f}{is_best_str}")

        if wandb_run is not None:
            try:
                import wandb
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_mae': val_mae
                })
            except Exception:
                pass

        if patience_counter >= cfg.PATIENCE:
            print(f"🛑 Early Stopping tại epoch {epoch}")
            break

    # Load trọng số tốt nhất để đánh giá trên Test set
    model.load_state_dict(best_model_weights)
    model.eval()

    total_mae, total_mape, total_mse = 0.0, 0.0, 0.0
    step_maes = [0.0] * 6
    step_mapes = [0.0] * 6
    count_batches = 0

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            pred = model(X_batch)

            y_true = Y_batch * stds + means
            y_pred = pred * stds + means

            err = y_true - y_pred
            abs_err = torch.abs(err)

            mask = (y_true > 0.5).float() # Mask xe > 0.5 để tránh chia cho 0

            total_mae += abs_err.mean().item()
            total_mse += (err ** 2).mean().item()

            mape_batch = (abs_err / (y_true + 1e-5)) * mask
            total_mape += mape_batch.sum().item() / max(mask.sum().item(), 1.0)

            for t_idx in range(6):
                step_maes[t_idx] += abs_err[:, t_idx, :, :].mean().item()
                mask_t = mask[:, t_idx, :, :]
                step_mapes[t_idx] += (mape_batch[:, t_idx, :, :].sum().item() / max(mask_t.sum().item(), 1.0))

            count_batches += 1

    avg_mae = total_mae / max(1, count_batches)
    avg_mape = total_mape / max(1, count_batches)
    avg_mse = total_mse / max(1, count_batches)
    avg_rmse = np.sqrt(avg_mse)

    metrics = {
        'mae': avg_mae,
        'mape': avg_mape,
        'rmse': avg_rmse,
        'mse': avg_mse
    }
    for t_idx in range(6):
        metrics[f'mae_t{t_idx+1}'] = step_maes[t_idx] / max(1, count_batches)
        metrics[f'mape_t{t_idx+1}'] = step_mapes[t_idx] / max(1, count_batches)

    if wandb_run is not None:
        try:
            import wandb
            wandb.log({
                'test_mae': avg_mae,
                'test_mape': avg_mape,
                'test_rmse': avg_rmse,
                'test_mae_t1': metrics['mae_t1'],
                'test_mae_t3': metrics['mae_t3'],
                'test_mae_t6': metrics['mae_t6']
            })
            wandb.finish()
        except Exception:
            pass

    return metrics


def run_ablation_benchmark():
    parser = argparse.ArgumentParser(description="Script huấn luyện Ablation Study cho các biến thể TA-STGCN.")
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 100, 2024, 777, 999],
                        help="Danh sách các seeds thử nghiệm Ablation Study.")
    parser.add_argument('--epochs', type=int, default=80, help="Số epochs tối đa.")
    parser.add_argument('--patience', type=int, default=20, help="Early stopping patience.")
    parser.add_argument('--batch_size', type=int, default=32, help="Kích thước batch_size.")
    parser.add_argument('--root_dir', type=str, default="/kaggle/input/datasets/canhdoo/nckh-traffic/GRAPH",
                        help="Thư mục gốc chứa dữ liệu.")
    parser.add_argument('--use_wandb', action='store_true', default=use_wandb_default,
                        help="Tự động log kết quả lên WandB (nếu có API key).")
    parser.add_argument('--no_wandb', action='store_true', help="Tắt log WandB.")
    parser.add_argument('--wandb_project', type=str, default="NCKH-Ablation-Study",
                        help="Tên WandB project.")
    args = parser.parse_args()

    use_wandb = args.use_wandb and not args.no_wandb

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
                use_temporal_attention=True, attn_num_heads=2, attn_dropout=0.1
            )
        },
        'TA-STGCN w/o Temporal Attention': {
            'build_fn': lambda cfg: Hybrid_STGCN_Model(
                num_nodes=len(nodes), in_feat=4, block_hidden=64,
                num_blocks=2, T_in=cfg.T_IN, cheb_K=3, horizon=cfg.HORIZON,
                output_feat=1, L_tilde=L_tilde, dropout=0.1,
                use_temporal_attention=False, attn_num_heads=2, attn_dropout=0.1
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
                use_temporal_attention=True, attn_num_heads=2, attn_dropout=0.1
            )
        }
    }

    results = {
        v_name: {
            'params': 0,
            'maes': [], 'mapes': [], 'rmses': [], 'mses': [],
            'step_maes': {f't{t_idx+1}': [] for t_idx in range(6)},
            'step_mapes': {f't{t_idx+1}': [] for t_idx in range(6)}
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

            test_metrics = train_single_ablation_variant(
                v_name, model, train_loader, val_loader, test_loader, base_cfg, device, seed, scaler,
                use_wandb=use_wandb, wandb_project=args.wandb_project
            )

            results[v_name]['maes'].append(test_metrics['mae'])
            results[v_name]['mapes'].append(test_metrics['mape'])
            results[v_name]['rmses'].append(test_metrics['rmse'])
            results[v_name]['mses'].append(test_metrics['mse'])

            for t_idx in range(6):
                results[v_name]['step_maes'][f't{t_idx+1}'].append(test_metrics[f'mae_t{t_idx+1}'])
                results[v_name]['step_mapes'][f't{t_idx+1}'].append(test_metrics[f'mape_t{t_idx+1}'])

            print(f"   ▶ Seed {seed:>4} | {v_name:<35} (Params: {params_count:,}) -> "
                  f"MAE: {test_metrics['mae']:.4f} (MAPE: {test_metrics['mape']:.2%})")

            del model
            torch.cuda.empty_cache()
            gc.collect()

    table_data = []
    for v_name, res in results.items():
        maes, mapes, rmses = res['maes'], res['mapes'], res['rmses']
        p_count = res['params']

        row = {
            'Ablation Variant': v_name,
            'Params': f"{p_count:,}",
            'MAE Overall': f"{np.mean(maes):.4f} ± {np.std(maes):.4f}",
            'MAPE Overall': f"{np.mean(mapes)*100:.2f}% ± {np.std(mapes)*100:.2f}%",
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
        f.write(f"- **Mô hình gốc**: TA-STGCN (2 ST-Conv blocks, $C=64$, $h=2$ heads)\n\n")
        f.write("## 🏆 Bảng Kết quả So sánh Ablation Study\n\n")
        f.write(summary_df.to_markdown(index=False))

    print(f"\n📑 Đã lưu báo cáo Ablation Study vào: {report_path}")

if __name__ == "__main__":
    run_ablation_benchmark()
