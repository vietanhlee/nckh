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
import torch.amp
from tqdm.auto import tqdm
from dotenv import load_dotenv

# Import 4 mô hình và cấu hình tương ứng
from stgcn import STGCN_Model as Baseline_STGCN_Model, Config as BaselineConfig
from hybrid import STGCN_Model as Hybrid_STGCN_Model, Config as HybridConfig, HuberSmoothLoss
from stgcn_block_attn import STGCN_BlockAttn_Model, Config as BlockAttnConfig
from stgcn_mixed_blocks import STGCN_Mixed_Model, Config as MixedConfig

# Import tiện ích nạp dữ liệu từ stgcn.py
from stgcn import (
    load_adj_from_excel,
    compute_scaled_laplacian,
    load_timeseries_double_rolling,
    MultiStepDataset,
    PureHuberLoss
)

# Đọc file .env
load_dotenv()

def set_seed(seed):
    """Cố định seed ngẫu nhiên đảm bảo tính lặp lại (Reproducibility)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ModelEMA:
    """Exponential Moving Average của trọng số mô hình."""
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k in self.shadow.keys():
                if self.shadow[k].dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
                else:
                    self.shadow[k].copy_(msd[k])


def evaluate_detailed(model, loader, device, scaler_stats, loss_fn=None):
    """Đánh giá chi tiết mô hình, bao gồm MAE tại cả 6 bước thời gian t+1 đến t+6."""
    model.eval()
    total_mae = 0
    total_mse = 0
    total_loss = 0
    total_step_maes = [0.0] * 6
    count_batches = 0

    means = torch.tensor(scaler_stats['mean'], device=device)
    stds = torch.tensor(scaler_stats['std'], device=device)

    pbar = tqdm(loader, desc="   Evaluating", leave=False)
    with torch.no_grad():
        for X, Y in pbar:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)

            if loss_fn is not None:
                loss_val = loss_fn(pred, Y)
                total_loss += loss_val.item()

            y_true = Y * stds + means
            y_pred = pred * stds + means

            err = y_true - y_pred
            abs_err = torch.abs(err)  # (B, Horizon=6, Nodes, 1)

            mae_val = abs_err.mean().item()
            total_mae += mae_val

            # MAE tại từng mốc bước thời gian từ t+1 đến t+6 (index 0 đến 5)
            for t_idx in range(6):
                total_step_maes[t_idx] += abs_err[:, t_idx, :, :].mean().item()

            sq_err = err ** 2
            total_mse += sq_err.mean().item()

            count_batches += 1

            # Dọn dẹp GPU Memory tức thì cho batch vừa tính
            del X, Y, pred, y_true, y_pred, err, abs_err

    if count_batches == 0:
        res = {'mae': 9999.0, 'mse': 9999.0, 'rmse': 9999.0, 'loss': 9999.0}
        for t_idx in range(6):
            res[f'mae_t{t_idx+1}'] = 9999.0
        return res

    avg_mae = total_mae / count_batches
    avg_mse = total_mse / count_batches
    avg_loss = total_loss / count_batches
    avg_rmse = np.sqrt(avg_mse)

    res = {
        'mae': avg_mae,
        'mse': avg_mse,
        'rmse': avg_rmse,
        'loss': avg_loss
    }
    for t_idx in range(6):
        res[f'mae_t{t_idx+1}'] = total_step_maes[t_idx] / count_batches

    return res


def train_single_seed(model_name, model, train_loader, val_loader, test_loader, cfg, device, seed, use_wandb=True, wandb_project="STGCN_NCKH_Benchmark"):
    """Huấn luyện 1 seed ngẫu nhiên cho 1 mô hình, tự động kết nối WandB và trả về chỉ số Test chi tiết."""
    set_seed(seed)

    # Khởi tạo WandB Run cho từng (Model, Seed)
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            project_name = os.getenv('WANDB_PROJECT', wandb_project)
            wandb_run = wandb.init(
                project=project_name,
                name=f"{model_name}_seed_{seed}",
                config={
                    'model_name': model_name,
                    'seed': seed,
                    'epochs': cfg.EPOCHS,
                    'batch_size': cfg.BATCH_SIZE,
                    'learning_rate': cfg.LEARNING_RATE,
                    'patience': cfg.PATIENCE,
                    'block_hidden': cfg.BLOCK_HIDDEN,
                    'num_blocks': getattr(cfg, 'NUM_BLOCKS', None),
                    'cheb_K': getattr(cfg, 'CHEB_K', None),
                    'loss_delta': cfg.LOSS_DELTA
                },
                reinit=True
            )
        except Exception as e:
            print(f"⚠️ Không thể khởi tạo WandB cho {model_name} (Seed {seed}): {e}")
            wandb_run = None

    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
    
    # Chọn loss function
    if model_name == "STGCN_Hybrid":
        loss_fn = HuberSmoothLoss(delta=cfg.LOSS_DELTA, smooth_weight=cfg.SMOOTH_LOSS_WEIGHT)
    else:
        loss_fn = PureHuberLoss(delta=cfg.LOSS_DELTA)

    grad_scaler = torch.amp.GradScaler('cuda')

    # LR Scheduler
    lr_scheduler = None
    if getattr(cfg, 'USE_LR_SCHEDULER', False):
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=cfg.LR_SCHED_FACTOR,
            patience=cfg.LR_SCHED_PATIENCE, min_lr=cfg.LR_SCHED_MIN_LR
        )

    # EMA
    use_ema = getattr(cfg, 'USE_EMA', False)
    ema = ModelEMA(model, decay=cfg.EMA_DECAY) if use_ema else None

    scaler_stats = {'mean': train_loader.dataset.means, 'std': train_loader.dataset.stds}
    grad_clip_norm = getattr(cfg, 'GRAD_CLIP_NORM', None)

    best_val_mae = float('inf')
    patience_cnt = 0
    checkpoint_path = os.path.join(cfg.SAVE_DIR, f"temp_{model_name}_seed_{seed}.pth")

    pbar = tqdm(range(cfg.EPOCHS), desc=f" Seed: {seed:>4} | {model_name:<18}", leave=False)
    for ep in pbar:
        model.train()
        total_loss = 0

        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            x_last = X[:, -1, :, :1].unsqueeze(1) if model_name == "STGCN_Hybrid" else None

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                pred = model(X)
                if model_name == "STGCN_Hybrid":
                    loss = loss_fn(pred, Y, x_last)
                else:
                    loss = loss_fn(pred, Y)

            grad_scaler.scale(loss).backward()

            if grad_clip_norm is not None:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            grad_scaler.step(optimizer)
            grad_scaler.update()

            if ema is not None:
                ema.update(model)

            total_loss += loss.item()
            del X, Y, pred, loss

        # Đánh giá trên tập Validation
        if ema is not None:
            raw_state = copy.deepcopy(model.state_dict())
            model.load_state_dict(ema.shadow)
            val_metrics = evaluate_detailed(model, val_loader, device, scaler_stats, loss_fn=loss_fn)
            model.load_state_dict(raw_state)
        else:
            val_metrics = evaluate_detailed(model, val_loader, device, scaler_stats, loss_fn=loss_fn)

        val_mae = val_metrics['mae']
        val_loss = val_metrics['loss']
        avg_train_loss = total_loss / len(train_loader)

        if lr_scheduler is not None:
            lr_scheduler.step(val_loss)

        # Log tiến trình từng epoch sang WandB
        if wandb_run is not None:
            try:
                import wandb
                wandb.log({
                    'epoch': ep + 1,
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    'val_mae': val_mae,
                    'val_mae_t1': val_metrics['mae_t1'],
                    'val_mae_t3': val_metrics['mae_t3'],
                    'val_mae_t6': val_metrics['mae_t6']
                })
            except Exception:
                pass

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_cnt = 0
            save_state = ema.shadow if ema is not None else model.state_dict()
            torch.save(save_state, checkpoint_path)
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.PATIENCE:
                break

    # Tải checkpoint tốt nhất và đánh giá chi tiết trên tập Test
    model.load_state_dict(torch.load(checkpoint_path))
    test_metrics = evaluate_detailed(model, test_loader, device, scaler_stats, loss_fn=loss_fn)

    # Log chỉ số Test cuối cùng lên WandB và đóng Run
    if wandb_run is not None:
        try:
            import wandb
            wandb.log({
                'test_mae': test_metrics['mae'],
                'test_rmse': test_metrics['rmse'],
                'test_mse': test_metrics['mse'],
                'test_mae_t1': test_metrics['mae_t1'],
                'test_mae_t2': test_metrics['mae_t2'],
                'test_mae_t3': test_metrics['mae_t3'],
                'test_mae_t4': test_metrics['mae_t4'],
                'test_mae_t5': test_metrics['mae_t5'],
                'test_mae_t6': test_metrics['mae_t6']
            })
            wandb.finish()
        except Exception:
            pass

    # Xoá file checkpoint tạm và giải phóng bộ nhớ GPU
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    del optimizer, grad_scaler
    if ema is not None:
        del ema
    torch.cuda.empty_cache()
    gc.collect()

    return test_metrics


def run_benchmark():
    parser = argparse.ArgumentParser(description="Script huấn luyện 5 Seeds ngẫu nhiên cho 4 mô hình STGCN.")
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 100, 2024, 777, 999],
                        help="Danh sách các seeds ngẫu nhiên (mặc định: 42 100 2024 777 999).")
    parser.add_argument('--epochs', type=int, default=500,
                        help="Số epochs chạy tối đa cho mỗi seed (mặc định: 500).")
    parser.add_argument('--patience', type=int, default=30,
                        help="Số patience early stopping (mặc định: 50).")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Kích thước batch_size (mặc định: 32 tránh CUDA OOM).")
    parser.add_argument('--root_dir', type=str, default="/kaggle/input/datasets/canhdoo/nckh-traffic/GRAPH",
                        help="Thư mục gốc chứa dữ liệu.")
    parser.add_argument('--use_wandb', action='store_true', default=True,
                        help="Tự động khởi tạo và ghi log lên WandB (mặc định: True).")
    parser.add_argument('--no_wandb', dest='use_wandb', action='store_false',
                        help="Tắt ghi log WandB.")
    parser.add_argument('--wandb_project', type=str, default="STGCN_NCKH_Benchmark",
                        help="Tên project trên WandB (mặc định: STGCN_NCKH_Benchmark).")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"============================================================")
    print(f"🚀 CHẠY BENCHMARK {len(args.seeds)} SEEDS CHO 4 MÔ HÌNH STGCN")
    print(f"   Device        : {device}")
    print(f"   Seeds         : {args.seeds}")
    print(f"   Epochs        : {args.epochs}")
    print(f"   Patience      : {args.patience}")
    print(f"   Batch Size    : {args.batch_size}")
    print(f"   WandB Logging : {args.use_wandb} (Project: {args.wandb_project})")
    print(f"============================================================")

    # Khởi tạo các Config
    stgcn_cfg = BaselineConfig()
    hybrid_cfg = HybridConfig()
    block_attn_cfg = BlockAttnConfig()
    mixed_cfg = MixedConfig()

    for cfg_inst in [stgcn_cfg, hybrid_cfg, block_attn_cfg, mixed_cfg]:
        cfg_inst.ROOT_DIR = args.root_dir
        cfg_inst.ADJ_PATH = os.path.join(args.root_dir, "Graph_fix_py_3.xlsx")
        cfg_inst.CSV_PATH = os.path.join(args.root_dir, "count_7_7_merg_sort_fix_fill.csv")
        cfg_inst.SAVE_DIR = "model/"
        cfg_inst.EPOCHS = args.epochs
        cfg_inst.PATIENCE = args.patience
        cfg_inst.BATCH_SIZE = args.batch_size
        os.makedirs(cfg_inst.SAVE_DIR, exist_ok=True)

    # Nạp ma trận đồ thị và dữ liệu
    print("\n[1] Nạp đồ thị và tiền xử lý dữ liệu...")
    A_raw, nodes = load_adj_from_excel(stgcn_cfg.ADJ_PATH)
    L_tilde = compute_scaled_laplacian(A_raw)
    df_all = load_timeseries_double_rolling(
        stgcn_cfg.CSV_PATH, nodes, stgcn_cfg.DATA_WINDOW1, stgcn_cfg.DATA_WINDOW2, stgcn_cfg.TIME_STEP_MINUTES
    )

    n_total = len(df_all)
    n_train = int(n_total * 0.8)
    n_val   = int(n_total * 0.1)

    idx_train_end = n_train
    idx_val_end   = n_train + n_val

    # Chia dữ liệu đồng bộ: Train 80%, Val 10% ở giữa, Test 10% ở cuối
    df_train = df_all.iloc[:idx_train_end]
    df_val   = df_all.iloc[idx_train_end:idx_val_end]
    df_test  = df_all.iloc[idx_val_end:]

    print(f"   - Dataset size: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    # 4 Mô hình đăng ký thử nghiệm
    models_registry = {
        'STGCN (Baseline)': {
            'class': Baseline_STGCN_Model,
            'config': stgcn_cfg,
            'build_fn': lambda cfg: Baseline_STGCN_Model(
                num_nodes=len(nodes), in_feat=4, block_hidden=cfg.BLOCK_HIDDEN,
                num_blocks=cfg.NUM_BLOCKS, T_in=cfg.T_IN, cheb_K=cfg.CHEB_K,
                horizon=cfg.HORIZON, output_feat=1, L_tilde=L_tilde, dropout=cfg.DROPOUT
            )
        },
        'STGCN_Hybrid': {
            'class': Hybrid_STGCN_Model,
            'config': hybrid_cfg,
            'build_fn': lambda cfg: Hybrid_STGCN_Model(
                num_nodes=len(nodes), in_feat=4, block_hidden=cfg.BLOCK_HIDDEN,
                num_blocks=cfg.NUM_BLOCKS, T_in=cfg.T_IN, cheb_K=cfg.CHEB_K,
                horizon=cfg.HORIZON, output_feat=1, L_tilde=L_tilde, dropout=cfg.DROPOUT,
                use_temporal_attention=cfg.USE_TEMPORAL_ATTENTION,
                attn_num_heads=cfg.ATTN_NUM_HEADS, attn_dropout=cfg.ATTN_DROPOUT
            )
        },
        'STGCN_BlockAttn': {
            'class': STGCN_BlockAttn_Model,
            'config': block_attn_cfg,
            'build_fn': lambda cfg: STGCN_BlockAttn_Model(
                num_nodes=len(nodes), in_feat=4, block_hidden=cfg.BLOCK_HIDDEN,
                num_blocks=cfg.NUM_BLOCKS, T_in=cfg.T_IN, cheb_K=cfg.CHEB_K,
                horizon=cfg.HORIZON, output_feat=1, L_tilde=L_tilde,
                num_heads=cfg.ATTN_NUM_HEADS, dropout=cfg.DROPOUT,
                use_final_attention=cfg.USE_FINAL_ATTENTION
            )
        },
        'STGCN_MixedBlocks': {
            'class': STGCN_Mixed_Model,
            'config': mixed_cfg,
            'build_fn': lambda cfg: STGCN_Mixed_Model(
                num_nodes=len(nodes), in_feat=4, block_hidden=cfg.BLOCK_HIDDEN,
                T_in=cfg.T_IN, cheb_K=cfg.CHEB_K, horizon=cfg.HORIZON, output_feat=1,
                L_tilde=L_tilde, num_heads=cfg.ATTN_NUM_HEADS, dropout=cfg.DROPOUT,
                use_final_attention=cfg.USE_FINAL_ATTENTION
            )
        }
    }

    # Lưu kết quả theo mô hình
    results = {
        model_name: {
            'maes': [], 'rmses': [], 'mses': [],
            'step_maes': {f't{t_idx+1}': [] for t_idx in range(6)}
        } for model_name in models_registry
    }

    # VÒNG LẶP NGOÀI: THEO TỪNG SEED
    for seed in args.seeds:
        print(f"\n{'='*65}")
        print(f"🌱 [SEED {seed}] BẮT ĐẦU CHẠY CẢ 4 PHƯƠNG PHÁP MÔ HÌNH")
        print(f"{'='*65}")

        # VÒNG LẶP TRONG: CHẠY LẦN LƯỢT 4 MÔ HÌNH VỚI SEED NÀY
        for model_name, info in models_registry.items():
            cfg = info['config']
            gc.collect()
            torch.cuda.empty_cache()

            train_ds = MultiStepDataset(df_train, nodes, cfg.T_IN, cfg.HORIZON)
            scaler   = {'mean': train_ds.means, 'std': train_ds.stds}
            val_ds   = MultiStepDataset(df_val, nodes, cfg.T_IN, cfg.HORIZON, scaler)
            test_ds  = MultiStepDataset(df_test, nodes, cfg.T_IN, cfg.HORIZON, scaler)

            eval_batch_size = min(cfg.BATCH_SIZE, 32)
            train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
            val_loader   = DataLoader(val_ds, batch_size=eval_batch_size)
            test_loader  = DataLoader(test_ds, batch_size=eval_batch_size)

            model = info['build_fn'](cfg).to(device)
            test_metrics = train_single_seed(
                model_name, model, train_loader, val_loader, test_loader, cfg, device, seed,
                use_wandb=args.use_wandb, wandb_project=args.wandb_project
            )

            results[model_name]['maes'].append(test_metrics['mae'])
            results[model_name]['rmses'].append(test_metrics['rmse'])
            results[model_name]['mses'].append(test_metrics['mse'])

            for t_idx in range(6):
                results[model_name]['step_maes'][f't{t_idx+1}'].append(test_metrics[f'mae_t{t_idx+1}'])

            print(f"   ▶ Seed {seed:>4} | {model_name:<18} -> "
                  f"MAE: {test_metrics['mae']:.4f} (t+1: {test_metrics['mae_t1']:.4f}, t+3: {test_metrics['mae_t3']:.4f}, t+6: {test_metrics['mae_t6']:.4f})")

            # Xoá mô hình khỏi GPU RAM sau mỗi lượt
            del model
            torch.cuda.empty_cache()
            gc.collect()

    # Tổng hợp báo cáo Markdown
    print(f"\n{'='*90}")
    print(f"🏆 BẢNG KẾT QUẢ TỔNG HỢP TẤT CẢ 6 BƯỚC THỜI GIAN (MEAN ± STD QUA {len(args.seeds)} SEEDS)")
    print(f"{'='*90}")

    table_data = []
    for model_name, res in results.items():
        maes, rmses, mses = res['maes'], res['rmses'], res['mses']

        row = {
            'Model': model_name,
            'MAE Overall': f"{np.mean(maes):.4f} ± {np.std(maes):.4f}",
            'MAE t+1 (5m)': f"{np.mean(res['step_maes']['t1']):.4f} ± {np.std(res['step_maes']['t1']):.4f}",
            'MAE t+2 (10m)': f"{np.mean(res['step_maes']['t2']):.4f} ± {np.std(res['step_maes']['t2']):.4f}",
            'MAE t+3 (15m)': f"{np.mean(res['step_maes']['t3']):.4f} ± {np.std(res['step_maes']['t3']):.4f}",
            'MAE t+4 (20m)': f"{np.mean(res['step_maes']['t4']):.4f} ± {np.std(res['step_maes']['t4']):.4f}",
            'MAE t+5 (25m)': f"{np.mean(res['step_maes']['t5']):.4f} ± {np.std(res['step_maes']['t5']):.4f}",
            'MAE t+6 (30m)': f"{np.mean(res['step_maes']['t6']):.4f} ± {np.std(res['step_maes']['t6']):.4f}",
            'RMSE': f"{np.mean(rmses):.4f} ± {np.std(rmses):.4f}",
            'MSE': f"{np.mean(mses):.4f} ± {np.std(mses):.4f}"
        }
        table_data.append(row)

    summary_df = pd.DataFrame(table_data)
    print(summary_df.to_string(index=False))

    # Ghi báo cáo ra file benchmark_5seeds_report.md
    report_path = "benchmark_5seeds_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# 📊 Báo cáo Thực nghiệm {len(args.seeds)} Seeds Ngẫu nhiên & Báo cáo Đầy đủ 6 Bước Thời gian (Mean ± Std)\n\n")
        f.write(f"- **Seeds sử dụng**: `{args.seeds}`\n")
        f.write(f"- **Tập dữ liệu**: Train 80%, Val 10% (ở giữa), Test 10% (ở cuối)\n")
        f.write(f"- **Cấu hình**: Epochs={args.epochs}, Patience={args.patience}, Batch Size={args.batch_size}\n\n")
        f.write("## 🏆 Bảng Kết quả So sánh 6 Bước Horizon Chi tiết\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n---\n\n## 📝 Chi tiết Metrics thô theo từng Seed\n\n")
        for model_name, res in results.items():
            f.write(f"### 🔹 {model_name}\n")
            f.write(f"- **MAE Overall**: {res['maes']}\n")
            for t_idx in range(6):
                f.write(f"- **MAE t+{t_idx+1} ({(t_idx+1)*5}m)**: {res['step_maes'][f't{t_idx+1}']}\n")
            f.write(f"- **RMSE Overall**: {res['rmses']}\n")
            f.write(f"- **MSE Overall**: {res['mses']}\n\n")

    print(f"\n📑 Đã lưu báo cáo chi tiết vào tệp: {report_path}")


if __name__ == "__main__":
    run_benchmark()
