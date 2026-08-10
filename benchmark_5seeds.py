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

# Import các mô hình và cấu hình tương ứng
from gcn_lstm import ImprovedGNN_LSTM, Config as GCNLSTMConfig, normalize_adj_sym
from stgcn import STGCN_Model as Baseline_STGCN_Model, Config as BaselineConfig
from hybrid import STGCN_Model as Hybrid_STGCN_Model, Config as HybridConfig, HuberSmoothLoss
from stgcn_mixed_blocks import STGCN_Mixed_Model, Config as MixedConfig
from advanced_baselines import GraphWaveNet, ASTGCN, GMAN

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
    """Đánh giá chi tiết mô hình, bao gồm MAE, MAPE tại cả 6 bước thời gian t+1 đến t+6."""
    model.eval()
    total_mae = 0
    total_mape = 0
    total_mse = 0
    total_loss = 0
    total_step_maes = [0.0] * 6
    total_step_mapes = [0.0] * 6
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

            mask = (y_true > 0.5).float() # mask nơi số lượng xe > 0.5 để tránh chia cho 0

            mae_val = abs_err.mean().item()
            total_mae += mae_val
            
            mape_batch = (abs_err / (y_true + 1e-5)) * mask
            mape_val = mape_batch.sum().item() / max(mask.sum().item(), 1.0)
            total_mape += mape_val

            # MAE và MAPE tại từng mốc bước thời gian từ t+1 đến t+6 (index 0 đến 5)
            for t_idx in range(6):
                total_step_maes[t_idx] += abs_err[:, t_idx, :, :].mean().item()
                mask_t = mask[:, t_idx, :, :]
                total_step_mapes[t_idx] += (mape_batch[:, t_idx, :, :].sum().item() / max(mask_t.sum().item(), 1.0))

            sq_err = err ** 2
            total_mse += sq_err.mean().item()

            count_batches += 1

            # Dọn dẹp GPU Memory tức thì cho batch vừa tính
            del X, Y, pred, y_true, y_pred, err, abs_err, mask, mape_batch

    if count_batches == 0:
        res = {'mae': 9999.0, 'mape': 9999.0, 'mse': 9999.0, 'rmse': 9999.0, 'loss': 9999.0}
        for t_idx in range(6):
            res[f'mae_t{t_idx+1}'] = 9999.0
            res[f'mape_t{t_idx+1}'] = 9999.0
        return res

    avg_mae = total_mae / count_batches
    avg_mape = total_mape / count_batches
    avg_mse = total_mse / count_batches
    avg_loss = total_loss / count_batches
    avg_rmse = np.sqrt(avg_mse)

    res = {
        'mae': avg_mae,
        'mape': avg_mape,
        'mse': avg_mse,
        'rmse': avg_rmse,
        'loss': avg_loss
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
    val_mae_history = []

    for ep in range(1, cfg.EPOCHS + 1):
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

        if patience_cnt >= cfg.PATIENCE:
            print(f"🛑 Early Stopping tại epoch {ep}")
            break

    # Tải checkpoint tốt nhất và đánh giá chi tiết trên tập Test
    model.load_state_dict(torch.load(checkpoint_path))
    test_metrics = evaluate_detailed(model, test_loader, device, scaler_stats, loss_fn=loss_fn)
    test_metrics['val_mae_history'] = val_mae_history

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
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 100, 2024, 777, 999],
                        help="Danh sách các seeds ngẫu nhiên (mặc định: 42 100 2024 777 999).")
    parser.add_argument('--model_group', type=str, choices=['all', 'advanced', 'standard'], default='all',
                        help="Nhóm mô hình cần chạy: 'advanced' (GraphWaveNet, ASTGCN, GMAN), 'standard' (GCN_LSTM, STGCN, STGCN_Hybrid, STGCN_MixedBlocks), 'all' (Tất cả).")
    parser.add_argument('--epochs', type=int, default=70,
                        help="Số epochs chạy tối đa cho mỗi seed (mặc định: 60).")
    parser.add_argument('--patience', type=int, default=15,
                        help="Số patience early stopping (mặc định: 50).")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Kích thước batch_size (mặc định: 32 tránh CUDA OOM).")
    parser.add_argument('--root_dir', type=str, default="/kaggle/input/datasets/canhdoo/nckh-traffic/GRAPH",
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
    print(f"🚀 CHẠY BENCHMARK {len(args.seeds)} SEEDS (Bao gồm Graph WaveNet, ASTGCN, GMAN)")
    print(f"   Device        : {device}")
    print(f"   Seeds         : {args.seeds}")
    print(f"   Epochs        : {args.epochs}")
    print(f"   Patience      : {args.patience}")
    print(f"   Batch Size    : {args.batch_size}")
    print(f"   WandB Logging : {args.use_wandb} (Project: {args.wandb_project})")
    print(f"============================================================")

    # Khởi tạo các Config
    gcn_lstm_cfg = GCNLSTMConfig()
    gcn_lstm_cfg.GCN_HIDDEN  = 64
    gcn_lstm_cfg.LSTM_HIDDEN = 160
    gcn_lstm_cfg.LSTM_LAYERS = 2

    stgcn_cfg = BaselineConfig()
    stgcn_cfg.BLOCK_HIDDEN   = 80
    stgcn_cfg.NUM_BLOCKS     = 3

    hybrid_cfg = HybridConfig()
    mixed_cfg = MixedConfig()

    for cfg_inst in [gcn_lstm_cfg, stgcn_cfg, hybrid_cfg, mixed_cfg]:
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

    # Đăng ký thử nghiệm các mô hình (Các mô hình Advanced Baselines được chạy trước để phát hiện lỗi sớm)
    models_registry = {
        'Graph_WaveNet': {
            'class': GraphWaveNet,
            'config': stgcn_cfg,
            'build_fn': lambda cfg: GraphWaveNet(
                num_nodes=len(nodes), in_dim=4, out_dim=1, blocks=4, layers=2, horizon=cfg.HORIZON
            )
        },
        'ASTGCN': {
            'class': ASTGCN,
            'config': stgcn_cfg,
            'build_fn': lambda cfg: ASTGCN(
                num_nodes=len(nodes), in_channels=4, K=cfg.CHEB_K, num_blocks=2, T_in=cfg.T_IN, horizon=cfg.HORIZON, block_channels=64, L_tilde=L_tilde
            )
        },
        'GMAN': {
            'class': GMAN,
            'config': stgcn_cfg,
            'build_fn': lambda cfg: GMAN(
                num_nodes=len(nodes), in_channels=4, T_in=cfg.T_IN, horizon=cfg.HORIZON, embed_size=64, heads=4, num_blocks=1
            )
        },
        'GCN_LSTM': {
            'class': ImprovedGNN_LSTM,
            'config': gcn_lstm_cfg,
            'build_fn': lambda cfg: ImprovedGNN_LSTM(
                num_nodes=len(nodes), in_feat=4, gcn_hidden=cfg.GCN_HIDDEN,
                lstm_hidden=cfg.LSTM_HIDDEN, lstm_layers=cfg.LSTM_LAYERS,
                horizon=cfg.HORIZON, output_feat=1, A_norm=A_norm, dropout=cfg.DROPOUT
            )
        },
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
                attn_num_heads=4, attn_dropout=cfg.ATTN_DROPOUT
            )
        },
        'STGCN_MixedBlocks': {
            'class': STGCN_Mixed_Model,
            'config': mixed_cfg,
            'build_fn': lambda cfg: STGCN_Mixed_Model(
                num_nodes=len(nodes), in_feat=4, block_hidden=cfg.BLOCK_HIDDEN,
                T_in=cfg.T_IN, cheb_K=cfg.CHEB_K, horizon=cfg.HORIZON, output_feat=1,
                L_tilde=L_tilde, num_heads=4, dropout=cfg.DROPOUT,
                use_final_attention=cfg.USE_FINAL_ATTENTION
            )
        }
    }

    # Lọc danh sách mô hình theo nhóm được yêu cầu
    if args.model_group == 'advanced':
        advanced_keys = ['Graph_WaveNet', 'ASTGCN', 'GMAN']
        models_registry = {k: v for k, v in models_registry.items() if k in advanced_keys}
        print("📌 Chạy nhóm Advanced Baselines: Graph_WaveNet, ASTGCN, GMAN")
    elif args.model_group == 'standard':
        advanced_keys = ['Graph_WaveNet', 'ASTGCN', 'GMAN']
        models_registry = {k: v for k, v in models_registry.items() if k not in advanced_keys}
        print("📌 Chạy nhóm Standard Models: GCN_LSTM, STGCN, STGCN_Hybrid, STGCN_MixedBlocks")

    # Lưu kết quả theo mô hình
    results = {
        model_name: {
            'params': 0,
            'flops_gflops': 0.0,
            'peak_mem_mb': 0.0,
            'inf_latencies': [],
            'maes': [], 'mapes': [], 'rmses': [], 'mses': [],
            'step_maes': {f't{t_idx+1}': [] for t_idx in range(6)},
            'step_mapes': {f't{t_idx+1}': [] for t_idx in range(6)},
            'val_mae_histories': []
        } for model_name in models_registry
    }

    # VÒNG LẶP NGOÀI: THEO TỪNG SEED
    for seed in args.seeds:
        print(f"\n{'='*65}")
        print(f"🌱 [SEED {seed}] BẮT ĐẦU CHẠY CẢ 4 MÔ HÌNH THỬ NGHIỆM")
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

            eval_batch_size = min(cfg.BATCH_SIZE, 32)
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
            results[model_name]['mapes'].append(test_metrics['mape'])
            results[model_name]['rmses'].append(test_metrics['rmse'])
            results[model_name]['mses'].append(test_metrics['mse'])
            results[model_name]['val_mae_histories'].append(test_metrics['val_mae_history'])

            for t_idx in range(6):
                results[model_name]['step_maes'][f't{t_idx+1}'].append(test_metrics[f'mae_t{t_idx+1}'])
                results[model_name]['step_mapes'][f't{t_idx+1}'].append(test_metrics[f'mape_t{t_idx+1}'])

            print(f"   ▶ Seed {seed:>4} | {model_name:<18} (Params: {params_count:,} | Mem: {peak_mem_mb:.1f}MB) -> "
                  f"MAE: {test_metrics['mae']:.4f} (MAPE: {test_metrics['mape']:.2%})")

            # Xoá mô hình khỏi GPU RAM sau mỗi lượt
            del model
            torch.cuda.empty_cache()
            gc.collect()

    # Tổng hợp báo cáo Markdown
    print(f"\n{'='*110}")
    print(f"🏆 BẢNG KẾT QUẢ TỔNG HỢP (PARAMS & MAE/MAPE MEAN ± STD QUA {len(args.seeds)} SEEDS)")
    print(f"{'='*110}")

    table_data = []
    for model_name, res in results.items():
        maes, mapes, rmses, mses = res['maes'], res['mapes'], res['rmses'], res['mses']
        inf_lats = res['inf_latencies']
        p_count = res['params']
        flops_g = res['flops_gflops']
        peak_mem = res['peak_mem_mb']

        row = {
            'Model': model_name,
            'Params': f"{p_count:,}",
            'Peak Mem (MB)': f"{peak_mem:.1f}",
            'MAE Overall': f"{np.mean(maes):.4f} ± {np.std(maes):.4f}",
            'MAPE Overall': f"{np.mean(mapes)*100:.2f}% ± {np.std(mapes)*100:.2f}%",
            'MAE t+1': f"{np.mean(res['step_maes']['t1']):.4f}",
            'MAE t+3': f"{np.mean(res['step_maes']['t3']):.4f}",
            'MAE t+6': f"{np.mean(res['step_maes']['t6']):.4f}",
            'RMSE': f"{np.mean(rmses):.4f}",
            'MSE': f"{np.mean(mses):.4f}"
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
            f.write(f"- **MAE Overall**: {res['maes']}\n")
            f.write(f"- **MAPE Overall**: {res['mapes']}\n")
            for t_idx in range(6):
                f.write(f"- **MAE t+{t_idx+1} ({(t_idx+1)*5}m)**: {res['step_maes'][f't{t_idx+1}']}\n")
            f.write(f"- **RMSE Overall**: {res['rmses']}\n")
            f.write(f"- **MSE Overall**: {res['mses']}\n\n")

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
    plt.legend()
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
        'STGCN_Hybrid': '#d62728',       # Bold Red / Ours
        'STGCN_MixedBlocks': '#ff7f0e',
        'STGCN (Baseline)': '#1f77b4',
        'GCN_LSTM': '#7f7f7f',
        'Graph_WaveNet': '#2ca02c',
        'ASTGCN': '#9467bd',
        'GMAN': '#8c564b'
    }
    markers = {
        'STGCN_Hybrid': 'o',
        'STGCN_MixedBlocks': 's',
        'STGCN (Baseline)': '^',
        'GCN_LSTM': 'x',
        'Graph_WaveNet': 'D',
        'ASTGCN': 'v',
        'GMAN': 'p'
    }

    # Left Subplot: MAE by Horizon
    for model_name, res in results.items():
        step_maes_mean = [np.mean(res['step_maes'][f't{t}']) for t in range(1, 7)]
        color = colors.get(model_name, '#333333')
        marker = markers.get(model_name, 'o')
        linewidth = 2.5 if 'Hybrid' in model_name or 'Ours' in model_name else 1.5
        linestyle = '-' if 'Hybrid' in model_name or 'Ours' in model_name else '--'
        
        axes[0].plot(x_steps, step_maes_mean, label=model_name, color=color, 
                     marker=marker, linewidth=linewidth, linestyle=linestyle, markersize=7)

    axes[0].set_xlabel('Prediction Horizon (Minutes Ahead)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Mean Absolute Error (MAE)', fontsize=11, fontweight='bold')
    axes[0].set_title('(a) MAE Growth across Prediction Horizons', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x_steps)
    axes[0].set_xticklabels(x_labels)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(fontsize=9, loc='upper left')

    # Right Subplot: MAPE by Horizon (%)
    for model_name, res in results.items():
        step_mapes_mean = [np.mean(res['step_mapes'][f't{t}']) * 100 for t in range(1, 7)]
        color = colors.get(model_name, '#333333')
        marker = markers.get(model_name, 'o')
        linewidth = 2.5 if 'Hybrid' in model_name or 'Ours' in model_name else 1.5
        linestyle = '-' if 'Hybrid' in model_name or 'Ours' in model_name else '--'
        
        axes[1].plot(x_steps, step_mapes_mean, label=model_name, color=color, 
                     marker=marker, linewidth=linewidth, linestyle=linestyle, markersize=7)

    axes[1].set_xlabel('Prediction Horizon (Minutes Ahead)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Mean Absolute Percentage Error (MAPE %)', fontsize=11, fontweight='bold')
    axes[1].set_title('(b) MAPE Growth across Prediction Horizons', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x_steps)
    axes[1].set_xticklabels(x_labels)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    horizon_png_path = os.path.join('plots', 'error_by_horizon.png')
    horizon_pdf_path = os.path.join('plots', 'error_by_horizon.pdf')
    plt.savefig(horizon_png_path, dpi=300, bbox_inches='tight')
    plt.savefig(horizon_pdf_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📉 Đã lưu biểu đồ tăng trưởng sai số Error by Horizon vào: {horizon_png_path} và {horizon_pdf_path}")


if __name__ == "__main__":
    run_benchmark()
