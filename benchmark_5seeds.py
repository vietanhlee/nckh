import os
# Cấu hình PyTorch Allocator tránh phân mảnh bộ nhớ CUDA OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import copy
import time
import random
import logging
import torch.nn.functional as F
from scipy.stats import wilcoxon, friedmanchisquare, t
import warnings
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

import sys

# Custom Dual Logger: Tự động ghi 100% tất cả lệnh print/log vừa ra CMD vừa lưu vào file log
class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = TeeLogger("logs/benchmark_5seeds.log")

# Import các mô hình và cấu hình tương ứng
from gcn_lstm import ImprovedGNN_LSTM, Config as GCNLSTMConfig, normalize_adj_sym
from stgcn import STGCN_Model as Baseline_STGCN_Model, Config as BaselineConfig
from hybrid import STGCN_Model as Hybrid_STGCN_Model, Config as HybridConfig, HuberSmoothLoss
from advanced_baselines import GraphWaveNet, ASTGCN, GMAN, AGCRN
from sota_2023_baselines import STAEformerProxy, MegaCRNProxy, DSTAGNNProxy

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


def count_parameters(model):
    """Đếm tổng số tham số có thể huấn luyện (Trainable Parameters) của mô hình."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, dummy_input):
    """
    Đếm tổng số phép tính toán FLOPs (tính bằng GFLOPs) cho 1 batch đầu vào.
    Thử dùng thư viện thop nếu có, nếu không thì dùng công thức xấp xỉ chính xác cho GNN.
    """
    try:
        import thop
        flops, _ = thop.profile(model, inputs=(dummy_input,), verbose=False)
        return flops / 1e9
    except Exception:
        params = count_parameters(model)
        B, T, N, F = dummy_input.shape
        approx_flops = 2 * params * T * N
        return approx_flops / 1e9


def measure_gpu_peak_memory(device):
    """Đo dung lượng bộ nhớ GPU đỉnh (Peak Memory Allocation tính bằng MB)."""
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return 0.0


def measure_inference_latency(model, loader, device, max_batches=20):
    """
    Đo độ trễ suy luận (Inference Latency) trung bình trên từng batch (tính bằng miligiây ms).
    """
    model.eval()
    # Warmup GPU
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
    """Đánh giá chi tiết mô hình cho bài toán dự báo phân loại (Car, Bike)."""
    model.eval()
    total_mae, car_mae, bike_mae = 0, 0, 0
    total_mape, car_mape, bike_mape = 0, 0, 0
    total_mse = 0
    total_loss = 0
    
    total_step_maes = [0.0] * 6
    total_step_mapes = [0.0] * 6
    count_batches = 0

    pw_total_list, pw_car_list, pw_bike_list = [], [], []

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

            # Tính MAE riêng lẻ
            err = y_true - y_pred
            abs_err = torch.abs(err)  # (B, H, N, 2)
            
            mae_car_batch = abs_err[:, :, :, 0].mean().item()
            mae_bike_batch = abs_err[:, :, :, 1].mean().item()
            
            # Tính MAE tổng phương tiện
            y_true_total = y_true.sum(dim=-1) # (B, H, N)
            y_pred_total = y_pred.sum(dim=-1)
            err_total = y_true_total - y_pred_total
            abs_err_total = torch.abs(err_total)
            mae_total_batch = abs_err_total.mean().item()

            pw_total_list.append(abs_err_total.cpu().numpy().flatten())
            pw_car_list.append(abs_err[:, :, :, 0].cpu().numpy().flatten())
            pw_bike_list.append(abs_err[:, :, :, 1].cpu().numpy().flatten())
            
            total_mae += mae_total_batch
            car_mae += mae_car_batch
            bike_mae += mae_bike_batch

            # Mape tổng
            mask_total = (y_true_total > 0.5).float()
            mape_batch_total = (abs_err_total / (y_true_total + 1e-5)) * mask_total
            total_mape += mape_batch_total.sum().item() / max(mask_total.sum().item(), 1.0)
            
            sq_err = err_total ** 2
            total_mse += sq_err.mean().item()

            # Horizon MAE cho tổng phương tiện
            for t_idx in range(6):
                total_step_maes[t_idx] += abs_err_total[:, t_idx, :].mean().item()
                mask_t = mask_total[:, t_idx, :]
                total_step_mapes[t_idx] += (mape_batch_total[:, t_idx, :].sum().item() / max(mask_t.sum().item(), 1.0))

            count_batches += 1
            del X, Y, pred, y_true, y_pred, err, abs_err, err_total, abs_err_total

    if count_batches == 0:
        res = {'mae': 9999.0, 'mae_car': 9999.0, 'mae_bike': 9999.0, 'mape': 9999.0, 'mse': 9999.0, 'rmse': 9999.0, 'loss': 9999.0, 'pw_total': np.array([]), 'pw_car': np.array([]), 'pw_bike': np.array([])}
        for t_idx in range(6):
            res[f'mae_t{t_idx+1}'] = 9999.0
            res[f'mape_t{t_idx+1}'] = 9999.0
        return res

    res = {
        'mae': total_mae / count_batches,
        'mae_car': car_mae / count_batches,
        'mae_bike': bike_mae / count_batches,
        'mape': total_mape / count_batches,
        'mse': total_mse / count_batches,
        'rmse': np.sqrt(total_mse / count_batches),
        'loss': total_loss / count_batches,
        'pw_total': np.concatenate(pw_total_list),
        'pw_car': np.concatenate(pw_car_list),
        'pw_bike': np.concatenate(pw_bike_list)
    }
    for t_idx in range(6):
        res[f'mae_t{t_idx+1}'] = total_step_maes[t_idx] / count_batches
        res[f'mape_t{t_idx+1}'] = total_step_mapes[t_idx] / count_batches

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
                    'block_hidden': getattr(cfg, 'BLOCK_HIDDEN', getattr(cfg, 'GCN_HIDDEN', 64)),
                    'gcn_hidden': getattr(cfg, 'GCN_HIDDEN', None),
                    'lstm_hidden': getattr(cfg, 'LSTM_HIDDEN', None),
                    'num_blocks': getattr(cfg, 'NUM_BLOCKS', None),
                    'cheb_K': getattr(cfg, 'CHEB_K', None),
                    'loss_delta': getattr(cfg, 'LOSS_DELTA', 1.0)
                },
                reinit=True
            )
        except Exception as e:
            print(f"⚠️ Không thể khởi tạo WandB cho {model_name} (Seed {seed}): {e}")
            wandb_run = None

    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
    
    # Chọn loss function (chuẩn hóa PureHuberLoss)
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
    val_mae_history = []
    epoch_times = []

    for ep in range(1, cfg.EPOCHS + 1):
        if device.type == 'cuda': torch.cuda.synchronize()
        t_ep_start = time.time()
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"   Epoch {ep:02d}/{cfg.EPOCHS}", leave=False)
        for X, Y in pbar:
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
            pbar.set_postfix(loss=f"{loss.item():.4f}")
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
        val_mae_history.append(val_mae)

        if lr_scheduler is not None:
            lr_scheduler.step(val_loss)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_cnt = 0
            save_state = ema.shadow if ema is not None else model.state_dict()
            torch.save(save_state, checkpoint_path)
            is_best_str = " -> Saved Best"
        else:
            patience_cnt += 1
            is_best_str = ""

        print(f"Ep {ep:02d}/{cfg.EPOCHS} | Loss: {avg_train_loss:.4f} / {val_loss:.4f} | Val MAE: {val_mae:.2f}{is_best_str}")

        # Log tiến trình từng epoch sang WandB
        if wandb_run is not None:
            try:
                import wandb
                wandb.log({
                    'epoch': ep,
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    'val_mae': val_mae,
                    'val_mae_t1': val_metrics['mae_t1'],
                    'val_mae_t3': val_metrics['mae_t3'],
                    'val_mae_t6': val_metrics['mae_t6'],
                    'val_mape': val_metrics['mape']
                })
            except Exception:
                pass

        if device.type == 'cuda': torch.cuda.synchronize()
        epoch_times.append(time.time() - t_ep_start)

        if patience_cnt >= cfg.PATIENCE:
            print(f"🛑 Early Stopping tại epoch {ep}")
            break

    # Tải checkpoint tốt nhất và đánh giá chi tiết trên tập Test
    model.load_state_dict(torch.load(checkpoint_path))
    test_metrics = evaluate_detailed(model, test_loader, device, scaler_stats, loss_fn=loss_fn)
    test_metrics['val_mae_history'] = val_mae_history
    test_metrics['epoch_sec'] = float(np.mean(epoch_times)) if epoch_times else 0.0

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
                'test_mae_t6': test_metrics['mae_t6'],
                'test_mape': test_metrics['mape']
            })
            wandb.finish()
        except Exception:
            pass

    # Xoá file checkpoint tạm và giải phóng bộ nhớ GPU triệt để
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    try:
        model.cpu()
    except Exception:
        pass

    del model, optimizer, grad_scaler
    if ema is not None:
        del ema
    gc.collect()
    torch.cuda.empty_cache()

    return test_metrics


def run_benchmark():
    parser = argparse.ArgumentParser(description="Script huấn luyện 5 Seeds ngẫu nhiên cho 5 mô hình (GCN-LSTM & STGCN).")
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 100, 2024, 22, 99],
                        help="Danh sách 5 seeds ngẫu nhiên (mặc định: 42 100 2024 22 99).")
    parser.add_argument('--model_group', type=str, choices=['all', 'advanced', 'standard', 'ablation'], default='all',
                        help="Nhóm mô hình cần chạy (mặc định 'all' tự động chạy tất cả 15 mô hình: SOTA Baselines + TA-STGCN + Tất cả các biến thể Ablation Study).")
    parser.add_argument('--epochs', type=int, default=90,
                        help="Số epochs chạy tối đa cho mỗi seed (mặc định: 80).")
    parser.add_argument('--patience', type=int, default=18,
                        help="Số patience early stopping (mặc định: 18).")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Kích thước batch_size (mặc định: 64 tối ưu cho 24GB VRAM GPU).")
    parser.add_argument('--root_dir', type=str, default="/workspace/GRAPH",
                        help="Thư mục gốc chứa dữ liệu.")
    parser.add_argument('--use_wandb', action='store_true', default=True,
                        help="Tự động khởi tạo và ghi log lên WandB (mặc định: True).")
    parser.add_argument('--no_wandb', dest='use_wandb', action='store_false',
                        help="Tắt ghi log WandB.")
    parser.add_argument('--wandb_project', type=str, default="NCKH-Benmark-5Seed",
                        help="Tên project trên WandB (mặc định: NCKH-Benmark-5Seed).")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"============================================================")
    print(f"🚀 CHẠY BENCHMARK {len(args.seeds)} SEEDS (24GB VRAM Optimized)")
    print(f"   Device        : {device}")
    print(f"   Seeds         : {args.seeds}")
    print(f"   Epochs        : {args.epochs}")
    print(f"   Patience      : {args.patience}")
    print(f"   Batch Size    : {args.batch_size}")
    print(f"   WandB Logging : {args.use_wandb} (Project: {args.wandb_project})")
    print(f"============================================================")

    # Khởi tạo các Config (Cùng hạng cân ~450k-570k params)
    gcn_lstm_cfg = GCNLSTMConfig()
    gcn_lstm_cfg.GCN_HIDDEN  = 80
    gcn_lstm_cfg.LSTM_HIDDEN = 200
    gcn_lstm_cfg.LSTM_LAYERS = 2

    stgcn_cfg = BaselineConfig()
    stgcn_cfg.BLOCK_HIDDEN   = 80
    stgcn_cfg.NUM_BLOCKS     = 3

    hybrid_cfg = HybridConfig()
    hybrid_cfg.BLOCK_HIDDEN = 80
    hybrid_cfg.NUM_BLOCKS = 3

    for cfg_inst in [gcn_lstm_cfg, stgcn_cfg, hybrid_cfg]:
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
    A_norm = normalize_adj_sym(A_raw)
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

    # Đăng ký thử nghiệm các mô hình
    models_registry = {
        'Graph_WaveNet': {
            'class': GraphWaveNet,
            'config': stgcn_cfg,
            'build_fn': lambda cfg: GraphWaveNet(
                num_nodes=len(nodes), in_dim=5, out_dim=2, residual_channels=40, dilation_channels=40, blocks=4, layers=2, horizon=cfg.HORIZON
            )
        },
        'ASTGCN': {
            'class': ASTGCN,
            'config': stgcn_cfg,
            'build_fn': lambda cfg: ASTGCN(
                num_nodes=len(nodes), in_channels=5, K=cfg.CHEB_K, num_blocks=2, T_in=cfg.T_IN, horizon=cfg.HORIZON, block_channels=36, L_tilde=L_tilde, out_dim=2
            )
        },
        'GMAN': {
            'class': GMAN,
            'config': stgcn_cfg,
            'build_fn': lambda cfg: GMAN(
                num_nodes=len(nodes), in_channels=5, T_in=cfg.T_IN, horizon=cfg.HORIZON, embed_size=128, heads=4, num_blocks=2, out_dim=2
            )
        },
        'GCN_LSTM': {
            'class': ImprovedGNN_LSTM,
            'config': gcn_lstm_cfg,
            'build_fn': lambda cfg: ImprovedGNN_LSTM(
                num_nodes=len(nodes), in_feat=5, gcn_hidden=cfg.GCN_HIDDEN,
                lstm_hidden=cfg.LSTM_HIDDEN, lstm_layers=cfg.LSTM_LAYERS,
                horizon=cfg.HORIZON, output_feat=2, A_norm=A_norm, dropout=cfg.DROPOUT
            )
        },
        'STGCN_Baseline': {
            'class': Baseline_STGCN_Model,
            'config': stgcn_cfg,
            'build_fn': lambda cfg: Baseline_STGCN_Model(
                num_nodes=len(nodes), in_feat=5, block_hidden=cfg.BLOCK_HIDDEN,
                num_blocks=cfg.NUM_BLOCKS, T_in=cfg.T_IN, cheb_K=cfg.CHEB_K,
                horizon=cfg.HORIZON, output_feat=2, L_tilde=L_tilde, dropout=cfg.DROPOUT
            )
        },
        'TA-STGCN': {
            'class': Hybrid_STGCN_Model,
            'config': hybrid_cfg,
            'build_fn': lambda cfg: Hybrid_STGCN_Model(
                num_nodes=len(nodes), in_feat=5, block_hidden=cfg.BLOCK_HIDDEN,
                num_blocks=cfg.NUM_BLOCKS, T_in=cfg.T_IN, cheb_K=cfg.CHEB_K,
                horizon=cfg.HORIZON, output_feat=2, L_tilde=L_tilde, dropout=cfg.DROPOUT,
                use_temporal_attention=cfg.USE_TEMPORAL_ATTENTION,
                attn_num_heads=4, attn_dropout=cfg.ATTN_DROPOUT
            )
        },
        'TA-STGCN (h=1)': {
            'class': Hybrid_STGCN_Model,
            'config': hybrid_cfg,
            'build_fn': lambda cfg: Hybrid_STGCN_Model(
                num_nodes=len(nodes), in_feat=5, block_hidden=cfg.BLOCK_HIDDEN,
                num_blocks=cfg.NUM_BLOCKS, T_in=cfg.T_IN, cheb_K=cfg.CHEB_K,
                horizon=cfg.HORIZON, output_feat=2, L_tilde=L_tilde, dropout=cfg.DROPOUT,
                use_temporal_attention=cfg.USE_TEMPORAL_ATTENTION,
                attn_num_heads=1, attn_dropout=cfg.ATTN_DROPOUT
            )
        },
        'TA-STGCN (C=32)': {
            'class': Hybrid_STGCN_Model,
            'config': hybrid_cfg,
            'build_fn': lambda cfg: Hybrid_STGCN_Model(
                num_nodes=len(nodes), in_feat=5, block_hidden=32,
                num_blocks=cfg.NUM_BLOCKS, T_in=cfg.T_IN, cheb_K=cfg.CHEB_K,
                horizon=cfg.HORIZON, output_feat=2, L_tilde=L_tilde, dropout=cfg.DROPOUT,
                use_temporal_attention=cfg.USE_TEMPORAL_ATTENTION,
                attn_num_heads=4, attn_dropout=cfg.ATTN_DROPOUT
            )
        },
        'TA-STGCN (K=1, No Spatial)': {
            'class': Hybrid_STGCN_Model,
            'config': hybrid_cfg,
            'build_fn': lambda cfg: Hybrid_STGCN_Model(
                num_nodes=len(nodes), in_feat=5, block_hidden=cfg.BLOCK_HIDDEN,
                num_blocks=cfg.NUM_BLOCKS, T_in=cfg.T_IN, cheb_K=1,
                horizon=cfg.HORIZON, output_feat=2, L_tilde=L_tilde, dropout=cfg.DROPOUT,
                use_temporal_attention=cfg.USE_TEMPORAL_ATTENTION,
                attn_num_heads=4, attn_dropout=cfg.ATTN_DROPOUT
            )
        },
        'TA-STGCN (h=8, 8-Heads)': {
            'class': Hybrid_STGCN_Model,
            'config': hybrid_cfg,
            'build_fn': lambda cfg: Hybrid_STGCN_Model(
                num_nodes=len(nodes), in_feat=5, block_hidden=cfg.BLOCK_HIDDEN,
                num_blocks=cfg.NUM_BLOCKS, T_in=cfg.T_IN, cheb_K=cfg.CHEB_K,
                horizon=cfg.HORIZON, output_feat=2, L_tilde=L_tilde, dropout=cfg.DROPOUT,
                use_temporal_attention=cfg.USE_TEMPORAL_ATTENTION,
                attn_num_heads=8, attn_dropout=cfg.ATTN_DROPOUT
            )
        },
        'TA-STGCN (Depth=2)': {
            'class': Hybrid_STGCN_Model,
            'config': hybrid_cfg,
            'build_fn': lambda cfg: Hybrid_STGCN_Model(
                num_nodes=len(nodes), in_feat=5, block_hidden=cfg.BLOCK_HIDDEN,
                num_blocks=2, T_in=cfg.T_IN, cheb_K=cfg.CHEB_K,
                horizon=cfg.HORIZON, output_feat=2, L_tilde=L_tilde, dropout=cfg.DROPOUT,
                use_temporal_attention=cfg.USE_TEMPORAL_ATTENTION,
                attn_num_heads=4, attn_dropout=cfg.ATTN_DROPOUT
            )
        },
        'AGCRN': {
            'class': AGCRN,
            'config': stgcn_cfg,
            'build_fn': lambda cfg: AGCRN(
                num_nodes=len(nodes), in_channels=5, T_in=cfg.T_IN, horizon=cfg.HORIZON, embed_dim=10, rnn_units=85, cheb_k=2, out_dim=2
            )
        },
        'STAEformer': {
            'class': STAEformerProxy,
            'config': stgcn_cfg,
            'build_fn': lambda cfg: STAEformerProxy(
                num_nodes=len(nodes), in_channels=5, T_in=cfg.T_IN, horizon=cfg.HORIZON, embed_size=144, heads=4, out_dim=2
            )
        },
        'MegaCRN': {
            'class': MegaCRNProxy,
            'config': stgcn_cfg,
            'build_fn': lambda cfg: MegaCRNProxy(
                num_nodes=len(nodes), in_channels=5, T_in=cfg.T_IN, horizon=cfg.HORIZON, embed_size=180, out_dim=2
            )
        },
        'DSTAGNN': {
            'class': DSTAGNNProxy,
            'config': stgcn_cfg,
            'build_fn': lambda cfg: DSTAGNNProxy(
                num_nodes=len(nodes), in_channels=5, T_in=cfg.T_IN, horizon=cfg.HORIZON, embed_size=200, heads=4, out_dim=2
            )
        }
    }

    # Lọc danh sách mô hình theo nhóm được yêu cầu
    if args.model_group == 'advanced':
        advanced_keys = ['Graph_WaveNet', 'ASTGCN', 'GMAN', 'STAEformer', 'MegaCRN', 'DSTAGNN', 'AGCRN']
        models_registry = {k: v for k, v in models_registry.items() if k in advanced_keys}
    elif args.model_group == 'standard':
        advanced_keys = ['Graph_WaveNet', 'ASTGCN', 'GMAN', 'STAEformer', 'MegaCRN', 'DSTAGNN', 'AGCRN', 'TA-STGCN (h=1)', 'TA-STGCN (C=32)']
        models_registry = {k: v for k, v in models_registry.items() if k not in advanced_keys}
    elif args.model_group == 'ablation':
        ablation_keys = ['STGCN_Baseline', 'TA-STGCN', 'TA-STGCN (h=1)', 'TA-STGCN (C=32)', 'TA-STGCN (K=1, No Spatial)', 'TA-STGCN (h=8, 8-Heads)', 'TA-STGCN (Depth=2)']
        models_registry = {k: v for k, v in models_registry.items() if k in ablation_keys}

    # Lưu kết quả theo mô hình
    results = {
        model_name: {
            'params': 0,
            'flops_gflops': 0.0,
            'peak_mem_mb': 0.0,
            'epoch_sec': [],
            'inf_latencies': [],
            'maes': [], 'car_maes': [], 'bike_maes': [], 'mapes': [], 'rmses': [], 'mses': [],
            'step_maes': {f't{t_idx+1}': [] for t_idx in range(6)},
            'step_mapes': {f't{t_idx+1}': [] for t_idx in range(6)},
            'val_mae_histories': [],
            'pw_totals': [], 'pw_cars': [], 'pw_bikes': []
        } for model_name in models_registry
    }

    # VÒNG LẶP NGOÀI: THEO TỪNG SEED
    for seed in args.seeds:
        print(f"\n{'='*65}")
        print(f"🌱 [SEED {seed}] BẮT ĐẦU CHẠY CÁC MÔ HÌNH THỬ NGHIỆM")
        print(f"{'='*65}")

        # VÒNG LẶP TRONG: CHẠY LẦN LƯỢT CÁC MÔ HÌNH VỚI SEED NÀY
        for model_name, info in models_registry.items():
            cfg = info['config']
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()

            train_ds = MultiStepDataset(df_train, nodes, cfg.T_IN, cfg.HORIZON)
            scaler   = {'mean': train_ds.means, 'std': train_ds.stds}
            val_ds   = MultiStepDataset(df_val, nodes, cfg.T_IN, cfg.HORIZON, scaler)
            test_ds  = MultiStepDataset(df_test, nodes, cfg.T_IN, cfg.HORIZON, scaler)

            eval_batch_size = cfg.BATCH_SIZE
            train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
            val_loader   = DataLoader(val_ds, batch_size=eval_batch_size)
            test_loader  = DataLoader(test_ds, batch_size=eval_batch_size)

            # Dọn dẹp GPU cache trước khi build mô hình mới
            gc.collect()
            torch.cuda.empty_cache()

            model = info['build_fn'](cfg).to(device)

            # Đếm số lượng tham số mô hình
            params_count = count_parameters(model)
            results[model_name]['params'] = params_count

            test_metrics = train_single_seed(
                model_name, model, train_loader, val_loader, test_loader, cfg, device, seed,
                use_wandb=args.use_wandb, wandb_project=args.wandb_project
            )

            # Đo Peak GPU memory
            peak_mem_mb = measure_gpu_peak_memory(device)
            results[model_name]['peak_mem_mb'] = max(results[model_name]['peak_mem_mb'], peak_mem_mb)

            results[model_name]['maes'].append(test_metrics['mae'])
            results[model_name]['car_maes'].append(test_metrics['mae_car'])
            results[model_name]['bike_maes'].append(test_metrics['mae_bike'])
            results[model_name]['mapes'].append(test_metrics['mape'])
            results[model_name]['rmses'].append(test_metrics['rmse'])
            results[model_name]['mses'].append(test_metrics['mse'])
            results[model_name]['val_mae_histories'].append(test_metrics['val_mae_history'])
            results[model_name]['pw_totals'].append(test_metrics['pw_total'])
            results[model_name]['pw_cars'].append(test_metrics['pw_car'])
            results[model_name]['pw_bikes'].append(test_metrics['pw_bike'])

            results[model_name]['epoch_sec'].append(test_metrics['epoch_sec'])

            for t_idx in range(6):
                results[model_name]['step_maes'][f't{t_idx+1}'].append(test_metrics[f'mae_t{t_idx+1}'])

            print(f"   ▶ Seed {seed:>4} | {model_name:<18} | Speed: {test_metrics['epoch_sec']:.2f} s/ep | MAE Total: {test_metrics['mae']:.4f} | Car: {test_metrics['mae_car']:.4f} | Bike: {test_metrics['mae_bike']:.4f}")

            # Xoá mô hình khỏi GPU RAM sau mỗi lượt
            del model
            torch.cuda.empty_cache()
            gc.collect()

    # Tổng hợp báo cáo Markdown
    print(f"\n{'='*110}")
    print(f"🏆 BẢNG KẾT QUẢ TỔNG HỢP (PARAMS & MAE)")
    print(f"{'='*110}")

    table_data = []
    for model_name, res in results.items():
        maes = res['maes']
        car_maes = res['car_maes']
        bike_maes = res['bike_maes']
        p_count = res['params']
        peak_mem = res['peak_mem_mb']

        def get_ci_str(arr):
            if not arr or len(arr) == 0:
                return "N/A"
            mean = np.mean(arr)
            std = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
            n = len(arr)
            t_crit = t.ppf(0.975, df=n-1) if n > 1 else 0.0
            margin = t_crit * (std / np.sqrt(n)) if n > 1 else 0.0
            return f"{mean:.4f} ± {std:.4f} (95% CI: {mean-margin:.4f}-{mean+margin:.4f})"

        train_speed_str = f"{np.mean(res['epoch_sec']):.2f} ± {np.std(res['epoch_sec']):.2f}" if res['epoch_sec'] else "N/A"
        row = {
            'Model': model_name,
            'Params': f"{p_count:,}",
            'Peak Mem (MB)': f"{peak_mem:.1f}",
            'Train Speed (s/ep)': train_speed_str,
            'MAE Overall': get_ci_str(maes),
            'MAE Car': get_ci_str(car_maes),
            'MAE Bike': get_ci_str(bike_maes),
            'MAE t+1': f"{np.mean(res['step_maes']['t1']):.4f}" if res['step_maes']['t1'] else "N/A",
            'MAE t+3': f"{np.mean(res['step_maes']['t3']):.4f}" if res['step_maes']['t3'] else "N/A",
            'MAE t+6': f"{np.mean(res['step_maes']['t6']):.4f}" if res['step_maes']['t6'] else "N/A"
        }
        table_data.append(row)

    summary_df = pd.DataFrame(table_data)
    print(summary_df.to_string(index=False))

    # Ghi báo cáo ra file benchmark_5seeds_report.md
    report_path = "benchmark_5seeds_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# 📊 Báo cáo Thực nghiệm {len(args.seeds)} Seeds Ngẫu nhiên\n\n")
        f.write(f"- **Seeds sử dụng**: `{args.seeds}`\n")
        f.write(f"- **Tập dữ liệu**: Train 80%, Val 10% (ở giữa), Test 10% (ở cuối)\n")
        f.write(f"- **Cấu hình**: Epochs={args.epochs}, Patience={args.patience}, Batch Size={args.batch_size}\n")
        f.write(f"- **Advanced Baselines Added**: Graph WaveNet, ASTGCN, GMAN. Các model này được chọn số block/channel sao cho dung lượng tham số tiệm cận hoặc cao hơn STGCN để so sánh công bằng.\n\n")
        f.write("## 🏆 Bảng Kết quả So sánh Tổng quan\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n---\n\n## 📝 Chi tiết Metrics thô theo từng Seed\n\n")
        for model_name, res in results.items():
            f.write(f"### 🔹 {model_name}\n")
            f.write(f"- **Trainable Parameters**: {res['params']:,}\n")
            f.write(f"- **MAE Total**: {res['maes']}\n")
            f.write(f"- **MAE Car**: {res['car_maes']}\n")
            f.write(f"- **MAE Bike**: {res['bike_maes']}\n")
            f.write(f"- **MAPE Overall**: {res['mapes']}\n")
            for t_idx in range(6):
                f.write(f"- **MAE t+{t_idx+1} ({(t_idx+1)*5}m)**: {res['step_maes'][f't{t_idx+1}']}\n")
            f.write(f"- **RMSE Overall**: {res['rmses']}\n")
            f.write(f"- **MSE Overall**: {res['mses']}\n\n")

        # --- Variance Decomposition Analysis ---
        f.write("\n## 📉 Phân tích Variance (Seed Stochasticity)\n\n")
        f.write("*Bảng dưới đây thống kê phương sai nội tại do thay đổi Seed (Seed Variance) trên tổng phương tiện, giúp đánh giá tính ổn định (Robustness) của mô hình.*\n\n")
        f.write("| Model | Seed Variance (Var) | Seed Std (Std) | Tỷ lệ biến động tương đối (Std / Mean) |\n")
        f.write("|---|---|---|---|\n")
        for m_name, res in results.items():
            if not res['maes']: continue
            mean_val = np.mean(res['maes'])
            var_val = np.var(res['maes'], ddof=1) if len(res['maes']) > 1 else 0.0
            std_val = np.std(res['maes'], ddof=1) if len(res['maes']) > 1 else 0.0
            cv = (std_val / mean_val) * 100 if mean_val > 0 else 0.0
            f.write(f"| {m_name} | {var_val:.4e} | {std_val:.4f} | {cv:.2f}% |\n")
            
        f.write("\n> 💡 **Nhận xét:** Nếu hệ số biến động (Std / Mean) rất nhỏ (ví dụ < 2%), điều này chứng tỏ sự chênh lệch hiệu năng chủ yếu đến từ bản chất thiết kế kiến trúc, thay vì phụ thuộc vào độ may rủi của Random Seed.\n\n")

        # --- Omnibus Test (Friedman Test) ---
        f.write("\n## 🔬 Kiểm định Tổng quát Friedman Test (Omnibus Test)\n\n")
        f.write("*Trước khi thực hiện các phép thử cặp (Wilcoxon), Friedman Test được chạy trên các mô hình baseline chính (loại trừ các biến thể ablation) để xác định xem có sự khác biệt có ý nghĩa thống kê trên toàn cục hay không, nhằm tránh lỗi Multiple Comparisons Problem.*\n\n")
        
        main_models = [m for m in results.keys() if not (m.startswith('TA-STGCN (') and m != 'TA-STGCN')]
        if len(main_models) > 1:
            all_pw_totals = [np.concatenate(results[m]['pw_totals']) for m in main_models]
            stat, p_friedman = friedmanchisquare(*all_pw_totals)
            
            f.write(f"- **H0:** Tất cả các mô hình baseline ({', '.join(main_models)}) có hiệu năng tương đương nhau.\n")
            f.write(f"- **Friedman Chi-Square Statistic:** {stat:.4f}\n")
            f.write(f"- **p-value:** {p_friedman:.4e}\n")
            
            if p_friedman < 0.05:
                f.write(f"\n> ✅ **Kết luận:** Trắc nghiệm Friedman cho ra $p < 0.05$. Có sự khác biệt có ý nghĩa thống kê giữa các mô hình baseline. Đủ điều kiện để tiến hành Post-hoc Test (Wilcoxon) bên dưới.\n\n")
            else:
                f.write(f"\n> ⚠️ **Kết luận:** Trắc nghiệm Friedman cho ra $p \\geq 0.05$. Không có đủ bằng chứng thống kê để bác bỏ H0. Không nên tin cậy vào Post-hoc Test.\n\n")
        else:
            f.write("⚠️ Không đủ số lượng mô hình baseline (>1) để chạy Friedman Test.\n\n")

        # --- Wilcoxon Signed-Rank Test & Effect Size (Main Baselines) ---
        baseline_model = "TA-STGCN" if "TA-STGCN" in results else list(results.keys())[0]
        f.write(f"\n## 🔬 Thống kê Post-Hoc: Wilcoxon Signed-Rank Test & Effect Size ({baseline_model} vs Main Baselines)\n\n")
        f.write("*Do cỡ mẫu ghép cặp cực lớn ($N \\approx 180,000$), p-value gần như luôn < 0.01. Do đó, báo cáo Effect Size (Cohen's $d_z$) được thêm vào để đánh giá ý nghĩa thực tiễn (Practical Significance).*\n")
        f.write("*Công thức Cohen's $d_z = \\frac{\\mu_{\\Delta}}{\\sigma_{\\Delta}}$. Thang đo: >0.2 (Small), >0.5 (Medium), >0.8 (Large).*\n\n")
        
        target_pw_total = np.concatenate(results[baseline_model]['pw_totals'])
        target_pw_car = np.concatenate(results[baseline_model]['pw_cars'])
        target_pw_bike = np.concatenate(results[baseline_model]['pw_bikes'])
        
        f.write("| Baseline Model | P-value (Total) | Cohen's d (Total) | P-value (Car) | Cohen's d (Car) | P-value (Bike) | Cohen's d (Bike) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        
        print(f"\n{'='*110}")
        print(f"🔬 WILCOXON TEST & EFFECT SIZE (Base: {baseline_model})")
        print(f"{'='*110}")
        
        def calc_cohens_dz(base_err, comp_err):
            if len(base_err) == 0 or len(comp_err) == 0: return 0.0
            diff = comp_err - base_err
            std_diff = np.std(diff, ddof=1)
            if std_diff == 0: return 0.0
            return np.mean(diff) / std_diff

        def format_sig(p, d):
            sig_star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            return f"{p:.2e}{sig_star}", f"{d:.3f}"

        for m_name in main_models:
            if m_name == baseline_model: continue
            comp_pw_total = np.concatenate(results[m_name]['pw_totals'])
            comp_pw_car = np.concatenate(results[m_name]['pw_cars'])
            comp_pw_bike = np.concatenate(results[m_name]['pw_bikes'])
            
            _, p_tot = wilcoxon(target_pw_total, comp_pw_total)
            _, p_car = wilcoxon(target_pw_car, comp_pw_car)
            _, p_bike = wilcoxon(target_pw_bike, comp_pw_bike)
            
            d_tot = calc_cohens_dz(target_pw_total, comp_pw_total)
            d_car = calc_cohens_dz(target_pw_car, comp_pw_car)
            d_bike = calc_cohens_dz(target_pw_bike, comp_pw_bike)
            
            p_tot_str, d_tot_str = format_sig(p_tot, d_tot)
            p_car_str, d_car_str = format_sig(p_car, d_car)
            p_bike_str, d_bike_str = format_sig(p_bike, d_bike)
            
            f.write(f"| {m_name} | {p_tot_str} | {d_tot_str} | {p_car_str} | {d_car_str} | {p_bike_str} | {d_bike_str} |\n")
            print(f"   ▶ {m_name:<16} | p(Total): {p_tot_str} (d={d_tot_str}) | p(Car): {p_car_str} (d={d_car_str}) | p(Bike): {p_bike_str} (d={d_bike_str})")

        # --- Dedicated Ablation Variants Statistical Test ---
        ablation_models = [m for m in results.keys() if m.startswith('TA-STGCN (') and m != 'TA-STGCN']
        if ablation_models and "TA-STGCN" in results:
            f.write(f"\n\n## 🔬 Thống kê Ablation Variants vs. Full Model (TA-STGCN)\n\n")
            f.write("| Ablation Variant | P-value (Total) | Cohen's d (Total) | P-value (Car) | Cohen's d (Car) | P-value (Bike) | Cohen's d (Bike) |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for m_name in ablation_models:
                comp_pw_total = np.concatenate(results[m_name]['pw_totals'])
                comp_pw_car = np.concatenate(results[m_name]['pw_cars'])
                comp_pw_bike = np.concatenate(results[m_name]['pw_bikes'])

                _, p_tot = wilcoxon(target_pw_total, comp_pw_total)
                _, p_car = wilcoxon(target_pw_car, comp_pw_car)
                _, p_bike = wilcoxon(target_pw_bike, comp_pw_bike)

                d_tot = calc_cohens_dz(target_pw_total, comp_pw_total)
                d_car = calc_cohens_dz(target_pw_car, comp_pw_car)
                d_bike = calc_cohens_dz(target_pw_bike, comp_pw_bike)

                p_tot_str, d_tot_str = format_sig(p_tot, d_tot)
                p_car_str, d_car_str = format_sig(p_car, d_car)
                p_bike_str, d_bike_str = format_sig(p_bike, d_bike)

                f.write(f"| {m_name} | {p_tot_str} | {d_tot_str} | {p_car_str} | {d_car_str} | {p_bike_str} | {d_bike_str} |\n")

    print(f"\n📑 Đã lưu báo cáo chi tiết vào tệp: {report_path}")

    # Plot Validation MAE curves
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 7))
    
    for model_name, res in results.items():
        histories = res['val_mae_histories']
        if not histories: continue
        
        max_len = max(len(h) for h in histories)
        padded_histories = []
        for h in histories:
            if len(h) < max_len:
                h = h + [h[-1]] * (max_len - len(h))
            padded_histories.append(h)
            
        mean_curve = np.mean(padded_histories, axis=0)
        plt.plot(range(1, max_len + 1), mean_curve, label=model_name, linewidth=2)
        
    plt.xlabel('Epoch')
    plt.ylabel('Validation MAE')
    plt.title('Validation MAE Convergence Curves (Average over Seeds)')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    os.makedirs('plots', exist_ok=True)
    plot_path = os.path.join('plots', 'val_mae_benchmark.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Đã lưu biểu đồ đường cong Validation MAE vào: {plot_path}")

    # Plot Error Growth by Horizon (t+1 to t+6: 5 min to 30 min ahead)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    x_steps = np.array([1, 2, 3, 4, 5, 6])
    x_labels = [f"t+{t}\n({t*5}m)" for t in x_steps]

    colors = {
        'TA-STGCN': '#d62728',       # Bold Red / Ours
        'STGCN_Baseline': '#1f77b4',
        'GCN_LSTM': '#7f7f7f',
        'Graph_WaveNet': '#2ca02c',
        'ASTGCN': '#9467bd',
        'GMAN': '#8c564b',
        'AGCRN': '#e377c2',
        'STAEformer': '#bcbd22',
        'MegaCRN': '#17becf',
        'DSTAGNN': '#8c564b'
    }

    # Left Subplot: MAE by Horizon
    for model_name, res in results.items():
        if not res['step_maes']['t1']: continue
        step_maes_mean = [np.mean(res['step_maes'][f't{t}']) for t in range(1, 7)]
        color = colors.get(model_name, '#333333')
        is_ours = (model_name == 'TA-STGCN')
        linewidth = 2.5 if is_ours else 1.5
        linestyle = '-' if is_ours else '--'
        marker = 'o' if is_ours else 's'
        
        axes[0].plot(x_steps, step_maes_mean, label=model_name, color=color, 
                     marker=marker, linewidth=linewidth, linestyle=linestyle, markersize=7)

    axes[0].set_xlabel('Prediction Horizon (Minutes Ahead)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Mean Absolute Error (MAE)', fontsize=11, fontweight='bold')
    axes[0].set_title('(a) MAE Growth across Prediction Horizons', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x_steps)
    axes[0].set_xticklabels(x_labels)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    # Right Subplot: MAPE by Horizon (%)
    for model_name, res in results.items():
        if not res['step_mapes']['t1']: continue
        step_mapes_mean = [np.mean(res['step_mapes'][f't{t}']) * 100 for t in range(1, 7)]
        color = colors.get(model_name, '#333333')
        is_ours = (model_name == 'TA-STGCN')
        linewidth = 2.5 if is_ours else 1.5
        linestyle = '-' if is_ours else '--'
        marker = 'o' if is_ours else 's'
        
        axes[1].plot(x_steps, step_mapes_mean, label=model_name, color=color, 
                     marker=marker, linewidth=linewidth, linestyle=linestyle, markersize=7)

    axes[1].set_xlabel('Prediction Horizon (Minutes Ahead)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Mean Absolute Percentage Error (MAPE %)', fontsize=11, fontweight='bold')
    axes[1].set_title('(b) MAPE Growth across Prediction Horizons', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x_steps)
    axes[1].set_xticklabels(x_labels)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    paper_fig_dir = os.path.join('paper', 'fig')
    os.makedirs(paper_fig_dir, exist_ok=True)
    
    horizon_png_path = os.path.join('plots', 'error_by_horizon.png')
    horizon_pdf_path = os.path.join('plots', 'error_by_horizon.pdf')
    plt.savefig(horizon_png_path, dpi=300, bbox_inches='tight')
    plt.savefig(horizon_pdf_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(paper_fig_dir, 'error_by_horizon.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(paper_fig_dir, 'error_by_horizon.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📉 Đã lưu biểu đồ tăng trưởng sai số Error by Horizon vào: {horizon_png_path} và paper/fig/")

    # Plot Grouped Bar Chart for MAE Total, Car, and Bike
    models = [m for m in results.keys() if len(results[m]['maes']) > 0]
    if models:
        mae_totals = [np.mean(results[m]['maes']) for m in models]
        mae_cars = [np.mean(results[m]['car_maes']) for m in models]
        mae_bikes = [np.mean(results[m]['bike_maes']) for m in models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(14, len(models) * 0.8), 7))
    rects1 = ax.bar(x - width, mae_totals, width, label='Total Vehicles', color='#1f77b4')
    rects2 = ax.bar(x, mae_cars, width, label='Car', color='#ff7f0e')
    rects3 = ax.bar(x + width, mae_bikes, width, label='Bike', color='#2ca02c')

    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison by Vehicle Category', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Attach a text label above each bar, displaying its height.
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    category_png_path = os.path.join('plots', 'mae_by_category.png')
    plt.savefig(category_png_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(paper_fig_dir, 'mae_by_category.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(paper_fig_dir, 'mae_by_category.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Đã lưu biểu đồ so sánh phân loại phương tiện vào: {category_png_path} và paper/fig/")

    # Dedicated Ablation Study Figure Generator
    ablation_models = [m for m in models if 'TA-STGCN' in m or m == 'STGCN_Baseline']
    if len(ablation_models) > 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        abl_maes = [np.mean(results[m]['maes']) for m in ablation_models]
        abl_cars = [np.mean(results[m]['car_maes']) for m in ablation_models]
        abl_bikes = [np.mean(results[m]['bike_maes']) for m in ablation_models]

        x_abl = np.arange(len(ablation_models))
        w_abl = 0.25

        r1 = ax.bar(x_abl - w_abl, abl_maes, w_abl, label='Overall MAE', color='#d62728')
        r2 = ax.bar(x_abl, abl_cars, w_abl, label='Car MAE', color='#ff7f0e')
        r3 = ax.bar(x_abl + w_abl, abl_bikes, w_abl, label='Bike MAE', color='#1f77b4')

        ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=11, fontweight='bold')
        ax.set_title('Ablation Study: Component Contribution Analysis', fontsize=13, fontweight='bold')
        ax.set_xticks(x_abl)
        ax.set_xticklabels(ablation_models, rotation=25, ha='right', fontsize=8.5)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        def label_abl(rects):
            for rect in rects:
                h = rect.get_height()
                ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=7.5)

        label_abl(r1)
        label_abl(r2)
        label_abl(r3)

        plt.tight_layout()
        abl_png_path = os.path.join('plots', 'ablation_breakdown.png')
        plt.savefig(abl_png_path, dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(paper_fig_dir, 'ablation_breakdown.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(paper_fig_dir, 'ablation_breakdown.pdf'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"🔬 Đã lưu biểu đồ phân tích Ablation Study vào: {abl_png_path} và paper/fig/")

if __name__ == "__main__":
    run_benchmark()
