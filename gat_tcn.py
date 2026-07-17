import os
import gc
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.amp
from tqdm import tqdm


class Config:
    ADJ_PATH = "/content/drive/MyDrive/GRAPH/Graph_fix_py_3.xlsx"
    CSV_PATH = "/content/drive/MyDrive/GRAPH/count_7_7_merg_sort_fix_fill.csv"
    SAVE_DIR = "/content/drive/MyDrive/GRAPH/model/"

    TIME_STEP_MINUTES = 5
    HISTORY_MINUTES   = 120
    HORIZON           = 6

    LOSS_DELTA = 1.0

    # --- MODEL ---
    GAT_HIDDEN  = 32
    GAT_HEADS   = 2
    TCN_CHANNELS = [64, 64]
    TCN_KERNEL   = 3
    DROPOUT     = 0.3

    # --- TRAIN ---
    BATCH_SIZE  = 16
    EPOCHS      = 100
    LEARNING_RATE = 0.001
    PATIENCE    = 20
    DATA_WINDOW1 = 3
    DATA_WINDOW2 = 5

    @property
    def T_IN(self):
      return int(self.HISTORY_MINUTES/self.TIME_STEP_MINUTES)

    @property
    def T_OUT(self):
      return self.HORIZON

    @property
    def PRED_MINUTES(self):
      return self.HORIZON * self.TIME_STEP_MINUTES

    @property
    def FULL_SAVE_PATH(self):
        if not os.path.exists(self.SAVE_DIR): os.makedirs(self.SAVE_DIR)
        return os.path.join(self.SAVE_DIR, f"model_GAT_TCN_{self.HORIZON}steps.pth")

CFG = Config()

# ============================================================
#  DATA UTILITIES (tái sử dụng từ gcn_lstm.py)
# ============================================================

def load_adj_from_excel(excel_path):
    df = pd.read_excel(excel_path, sheet_name=0, index_col=0)
    mat = df.apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy(dtype=float)
    nonzero = mat[mat > 0]
    sigma = nonzero.mean() if nonzero.size > 0 else 1.0
    weights = np.zeros_like(mat)
    mask = mat > 0
    weights[mask] = np.exp(-mat[mask] / (sigma + 1e-9))
    return weights, list(df.index)

def normalize_adj_sym(A):
    A = A.astype(float)
    A = A + np.eye(A.shape[0])
    d = A.sum(axis=1)
    d_inv_sqrt = np.power(d, -0.5, where=d>0)
    d_inv_sqrt[d<=0] = 0
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt

def add_rich_time_features(timestamps):
    tod = timestamps.hour * 60 + timestamps.minute
    tod_rad = 2 * np.pi * tod / 1440.0
    hour_norm = timestamps.hour / 24.0
    features = np.stack([np.sin(tod_rad), np.cos(tod_rad), hour_norm], axis=1)
    return features

def load_timeseries_double_rolling(csv_path, node_list, window1 = 3, window2 = 5, step_minutes=5):
    print(f"   Reading CSV: {csv_path}...")
    df = pd.read_csv(csv_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])

    if node_list is not None:
        df = df[df['STT'].isin(node_list)]
    else:
        node_list = sorted(df['STT'].unique())

    pivot = df.pivot_table(index='Timestamp', columns='STT', values='Total Vehicles', aggfunc='mean')
    pivot = pivot.reindex(columns=node_list)

    pivot_1min = pivot.resample('1min').mean().interpolate(method='linear', limit=30).fillna(0.0)

    smooth_1 = pivot_1min.rolling(window=window1, center=False, min_periods=1).mean()
    smooth_2 = smooth_1.ewm(span=window2, adjust=False).mean()

    resample_rule = f'{step_minutes}min'
    pivot_final = smooth_2.asfreq(resample_rule).fillna(0.0)

    pivot_final.columns = pd.MultiIndex.from_product([pivot_final.columns, ['Total Vehicles']], names=['Node', 'Feature'])

    print(f"   Double Rolling Data loaded. Shape: {pivot_final.shape}")
    return pivot_final

# ============================================================
#  DATASET (tái sử dụng từ gcn_lstm.py)
# ============================================================

class MultiStepDataset(Dataset):
    def __init__(self, data_df, node_order, T_in, Horizon, scaler=None):
        self.T_in = T_in
        self.Horizon = Horizon
        self.node_order = node_order

        df_sorted = data_df.sort_index(axis=1, level='Node')
        desired_cols = pd.MultiIndex.from_product([node_order, ['Total Vehicles']], names=['Node', 'Feature'])
        self.df = df_sorted.reindex(columns=desired_cols)

        self.timestamps = self.df.index
        self.N = len(node_order)

        self.values = self.df.values.astype(float).reshape(-1, self.N, 1)

        self.time_feats = add_rich_time_features(self.timestamps)

        if scaler is None:
            self.means = np.mean(self.values, axis=0, keepdims=True)
            self.stds = np.std(self.values, axis=0, keepdims=True) + 1e-6
        else:
            self.means = scaler['mean']
            self.stds = scaler['std']

        self.valid_len = self.values.shape[0] - self.T_in - self.Horizon + 1

    def __len__(self): return max(0, self.valid_len)

    def __getitem__(self, idx):
        x_node = self.values[idx : idx + self.T_in]
        y_node = self.values[idx + self.T_in : idx + self.T_in + self.Horizon]

        x_node = (x_node - self.means) / self.stds
        y_node = (y_node - self.means) / self.stds

        t_in_feats = self.time_feats[idx : idx + self.T_in]
        t_in_expanded = np.tile(np.expand_dims(t_in_feats, axis=1), (1, self.N, 1))

        x_final = np.concatenate([x_node, t_in_expanded], axis=-1)
        return torch.from_numpy(x_final.astype(np.float32)), torch.from_numpy(y_node.astype(np.float32))

# ============================================================
#  MODEL: GAT + TCN
# ============================================================

class GATLayer(nn.Module):
    """
    Graph Attention Layer (Veličković et al., 2018)
    Multi-head attention trên đồ thị — tự học trọng số attention
    giữa các node láng giềng thay vì dùng trọng số cố định như GCN.
    """
    def __init__(self, in_feats, out_feats, num_heads=2, dropout=0.3, concat=True):
        super().__init__()
        self.num_heads = num_heads
        self.concat = concat

        if concat:
            assert out_feats % num_heads == 0, \
                f"out_feats ({out_feats}) phải chia hết cho num_heads ({num_heads}) khi concat=True"
            self.out_per_head = out_feats // num_heads
        else:
            self.out_per_head = out_feats

        # Learnable weight matrix và attention vectors
        self.W = nn.Linear(in_feats, self.out_per_head * num_heads, bias=False)
        self.a_src = nn.Parameter(torch.empty(num_heads, self.out_per_head))
        self.a_dst = nn.Parameter(torch.empty(num_heads, self.out_per_head))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, adj):
        """
        Args:
            x:   (B, N, F_in) — node features
            adj: (N, N)       — adjacency matrix (dùng làm mask kết nối)
        Returns:
            (B, N, out_feats)
        """
        B, N, _ = x.shape

        # Linear projection: (B, N, F_in) -> (B, N, H, F_per_head)
        h = self.W(x).view(B, N, self.num_heads, self.out_per_head)

        # Tính attention score: e_ij = LeakyReLU(a_src^T h_i + a_dst^T h_j)
        attn_src = (h * self.a_src).sum(dim=-1)                  # (B, N, H)
        attn_dst = (h * self.a_dst).sum(dim=-1)                  # (B, N, H)
        attn = attn_src.unsqueeze(2) + attn_dst.unsqueeze(1)     # (B, N_i, N_j, H)
        attn = self.leaky_relu(attn)

        # Mask: chỉ attend đến neighbors + self-loop
        adj_mask = (adj > 0) | torch.eye(N, dtype=torch.bool, device=adj.device)
        attn = attn.masked_fill(~adj_mask.unsqueeze(0).unsqueeze(-1), float('-inf'))

        # Softmax qua neighbors
        attn = F.softmax(attn, dim=2)    # (B, N, N, H) — chuẩn hóa qua source nodes
        attn = self.attn_drop(attn)

        # Weighted aggregation: out_i = Σ_j α_ij h_j
        out = torch.einsum('bijh,bjhf->bihf', attn, h)   # (B, N, H, F_per_head)

        if self.concat:
            out = out.reshape(B, N, -1)     # (B, N, H * F_per_head)
        else:
            out = out.mean(dim=2)           # (B, N, F_per_head)

        return out


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Block — Dilated Causal Convolution.
    Xử lý song song (không sequential như LSTM), receptive field
    tăng exponentially theo dilation rate.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.3):
        super().__init__()
        self.causal_padding = (kernel_size - 1) * dilation   # left-padding for causality

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.causal_padding)
        self.norm = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual projection nếu kênh thay đổi
        self.residual = nn.Conv1d(in_channels, out_channels, 1) \
                        if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: (B, T, C_in)
        Returns:
            (B, T, C_out)
        """
        # Chuyển sang (B, C, T) cho Conv1d
        res = self.residual(x.transpose(1, 2))        # (B, C_out, T)

        out = self.conv(x.transpose(1, 2))             # (B, C_out, T + padding)
        if self.causal_padding > 0:
            out = out[:, :, :-self.causal_padding]     # Cắt future → causal

        out = out + res                                 # Residual connection
        out = out.transpose(1, 2)                       # (B, T, C_out)
        out = self.norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out


class GAT_TCN_Model(nn.Module):
    """
    GAT + TCN: Graph Attention Network cho spatial modeling,
    Temporal Convolutional Network cho temporal modeling.

    Pipeline: Input → GAT(×2) → TCN(stacked, dilation tăng dần) → Projection → Output
    """
    def __init__(self, num_nodes, in_feat, gat_hidden, gat_heads, tcn_channels,
                 tcn_kernel, horizon, output_feat, A_raw=None, dropout=0.3):
        super().__init__()
        self.horizon = horizon
        self.output_feat = output_feat

        # === Spatial: 2 lớp GAT ===
        self.gat1 = GATLayer(in_feat, gat_hidden, gat_heads, dropout, concat=True)
        self.gat2 = GATLayer(gat_hidden, gat_hidden, gat_heads, dropout, concat=True)

        self.dropout = nn.Dropout(dropout)

        # === Temporal: TCN với dilation tăng dần (1, 2, 4, ...) ===
        tcn_layers = []
        in_ch = gat_hidden
        for i, out_ch in enumerate(tcn_channels):
            tcn_layers.append(TCNBlock(in_ch, out_ch, tcn_kernel,
                                       dilation=2**i, dropout=dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*tcn_layers)

        # === Output projection ===
        self.proj_shared = nn.Sequential(
            nn.Linear(tcn_channels[-1], tcn_channels[-1] // 2),
            nn.LayerNorm(tcn_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.proj_out = nn.Linear(tcn_channels[-1] // 2, horizon * output_feat)

        # Ma trận kề gốc (dùng làm mask cho GAT attention)
        if A_raw is None:
            self.register_buffer('A_raw', torch.eye(num_nodes))
        else:
            self.register_buffer('A_raw', torch.tensor(A_raw, dtype=torch.float32))

    def forward(self, x):
        """
        Args:
            x: (B, T, N, F) — B=batch, T=time steps, N=nodes, F=features
        Returns:
            (B, Horizon, N, 1) — dự đoán multi-step
        """
        B, T, N, F_in = x.shape

        # --- Spatial: GAT xử lý từng time step ---
        x_flat = x.view(B * T, N, F_in)
        h = F.elu(self.gat1(x_flat, self.A_raw))
        h = self.dropout(h)
        h = F.elu(self.gat2(h, self.A_raw))
        h = self.dropout(h)
        # h: (B*T, N, gat_hidden)

        # --- Temporal: TCN xử lý từng node ---
        h = h.view(B, T, N, -1)                     # (B, T, N, gat_hidden)
        h = h.permute(0, 2, 1, 3).contiguous()      # (B, N, T, gat_hidden)
        h = h.view(B * N, T, -1)                     # (B*N, T, gat_hidden)

        h = self.tcn(h)                              # (B*N, T, tcn_out)
        last = h[:, -1, :]                           # (B*N, tcn_out) — lấy step cuối

        # --- Output ---
        shared = self.proj_shared(last)
        out = self.proj_out(shared)
        out = out.view(B, N, self.horizon, self.output_feat)
        y_pred = out.permute(0, 2, 1, 3)            # (B, Horizon, N, output_feat)
        return y_pred

# ============================================================
#  LOSS (tái sử dụng từ gcn_lstm.py)
# ============================================================

class PureHuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.loss_fn = nn.HuberLoss(delta=delta)

    def forward(self, pred, target, x_last=None):
        return self.loss_fn(pred, target)

# ============================================================
#  TRAINING UTILITIES (tái sử dụng từ gcn_lstm.py)
# ============================================================

def train_one_epoch(model, loader, opt, loss_fn, device, scaler_obj, scaler_stats):

    model.train()
    total_loss = 0
    total_mae = 0
    count_batches = 0

    means = torch.tensor(scaler_stats['mean'], device=device)
    stds = torch.tensor(scaler_stats['std'], device=device)

    pbar = tqdm(loader, desc="   Training", leave=False)
    for X, Y in pbar:
        X, Y = X.to(device), Y.to(device)
        x_last = X[:, -1, :, :1].unsqueeze(1)

        opt.zero_grad()
        with torch.amp.autocast('cuda'):
            pred = model(X)
            loss = loss_fn(pred, Y, x_last)

        scaler_obj.scale(loss).backward()
        scaler_obj.step(opt)
        scaler_obj.update()

        total_loss += loss.item()

        with torch.no_grad():
            y_true = Y * stds + means
            y_pred = pred * stds + means
            mae_batch = torch.abs(y_true - y_pred).mean()
            total_mae += mae_batch.item()

        count_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", mae=f"{mae_batch.item():.2f}")

    avg_loss = total_loss / count_batches
    avg_mae = total_mae / count_batches
    return avg_loss, avg_mae

def evaluate(model, loader, device, scaler_stats, loss_fn=None, verbose=False):

    model.eval()
    total_mae = 0
    total_mse = 0
    total_loss = 0
    count_batches = 0

    means = torch.tensor(scaler_stats['mean'], device=device)
    stds = torch.tensor(scaler_stats['std'], device=device)

    pbar = tqdm(loader, desc="   Evaluating", leave=False)
    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            x_last = X[:, -1, :, :1].unsqueeze(1)

            pred = model(X)

            if loss_fn is not None:
                loss_val = loss_fn(pred, Y, x_last)
                total_loss += loss_val.item()

            y_true = Y * stds + means
            y_pred = pred * stds + means

            err = y_true - y_pred

            # MAE
            abs_err = torch.abs(err)
            mae_val = abs_err.mean().item()
            total_mae += mae_val

            # MSE
            sq_err = err ** 2
            total_mse += sq_err.mean().item()

            count_batches += 1
            pbar.set_postfix(mae=f"{mae_val:.2f}")

    if count_batches == 0:
        return {'mae': 9999.0, 'mse': 9999.0, 'rmse': 9999.0, 'loss': 9999.0}

    avg_mae = total_mae / count_batches
    avg_mse = total_mse / count_batches
    avg_loss = total_loss / count_batches
    avg_rmse = np.sqrt(avg_mse)

    if verbose:
        pass

    return {'mae': avg_mae, 'mse': avg_mse, 'rmse': avg_rmse, 'loss': avg_loss}

def plot_training_history(train_losses, val_losses, train_maes, val_maes):

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'orange', label='Validation Loss')
    plt.title('Loss History (Huber - Normalized)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_maes, 'b--', label='Training MAE', alpha=0.7)
    plt.plot(epochs, val_maes, 'r-o', label='Validation MAE', markersize=4)

    if len(val_maes) > 0:
        min_mae = min(val_maes)
        min_epoch = val_maes.index(min_mae) + 1
        plt.plot(min_epoch, min_mae, 'g*', markersize=15, label=f'Best Val MAE: {min_mae:.2f}')

    plt.title('MAE History (Real Scale - Vehicles)')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

def visualize_last_step(model, loader, device, scaler, cfg, node_list=None):
    if node_list is None: node_list = [383, 266]
    plot_step_idx = cfg.T_OUT - 1
    print(f"\n{'='*20} VISUALIZING STEP {plot_step_idx+1} ({ (plot_step_idx+1)*cfg.TIME_STEP_MINUTES } mins ahead) {'='*20}")

    model.eval()
    y_true_list, y_pred_list = [], []
    means = scaler['mean']
    stds = scaler['std']

    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            pred_seq = model(X).cpu().numpy()
            Y_seq = Y.numpy()
            pred_last = pred_seq[:, plot_step_idx, :, :]
            y_last    = Y_seq[:, plot_step_idx, :, :]
            y_true_list.append(y_last * stds + means)
            y_pred_list.append(pred_last * stds + means)

    if len(y_true_list) == 0:
        print("Không có dữ liệu để visualize (Dataset trống).")
        return

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    total_nodes_avail = y_true.shape[1]

    for node_idx in node_list:
        if node_idx >= total_nodes_avail: continue
        gt_node = y_true[:, node_idx, 0]
        pred_node = y_pred[:, node_idx, 0]
        mae_node = np.mean(np.abs(gt_node - pred_node))
        print(f"Node {node_idx} | MAE: {mae_node:.2f}")

        plt.figure(figsize=(12, 6))
        plt.plot(gt_node, label='Ground Truth', color='black', alpha=0.7)
        plt.plot(pred_node, label=f'Pred +{cfg.PRED_MINUTES}m', color='red', linestyle='--', alpha=0.9)
        plt.title(f"Node {node_idx} - MAE: {mae_node:.2f} - Prediction {cfg.PRED_MINUTES} mins ahead")
        plt.ylabel("Total Vehicles Count")
        plt.xlabel("Time Steps")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# ============================================================
#  MAIN TRAINING PIPELINE
# ============================================================

def run_training():
    gc.collect(); torch.cuda.empty_cache()
    print(f"TRAINING GAT+TCN (Steps={CFG.HORIZON}, Future={CFG.PRED_MINUTES} mins)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load adjacency — GAT dùng A_raw trực tiếp (không cần normalize)
    A_raw, nodes = load_adj_from_excel(CFG.ADJ_PATH)

    print(f"Loading ALL Data from: {CFG.CSV_PATH}")
    df_all = load_timeseries_double_rolling(CFG.CSV_PATH, nodes, CFG.DATA_WINDOW1, CFG.DATA_WINDOW2, CFG.TIME_STEP_MINUTES)

    n_total = len(df_all)
    n_train  = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    idx_train_end  = n_train
    idx_val_end = n_train + n_val

    print(f"\nChia dữ liệu (TEST -> TRAIN -> VAL): Total={n_total}")
    print(f" 1. Train:  0 -> {idx_train_end} (80%)")
    print(f" 2. Val:   {idx_train_end} -> {idx_val_end} (10%)")
    print(f" 3. Test:    {idx_val_end} -> {n_total} (10%)")


    df_train  = df_all.iloc[:idx_train_end]
    df_val = df_all.iloc[idx_train_end:idx_val_end]
    df_test   = df_all.iloc[idx_val_end:]

    def get_time_range(df, name):
        if len(df) == 0: return f"{name}: EMPTY"
        start_t = df.index[0].strftime('%Y-%m-%d %H:%M')
        end_t   = df.index[-1].strftime('%Y-%m-%d %H:%M')
        return f"{name:<6} ({len(df):>4} samples): {start_t}  --->  {end_t}"

    print("\n" + "="*60)
    print("📅  DATASET TIME RANGES")
    print("="*60)
    print(get_time_range(df_val, "VAL"))
    print(get_time_range(df_test, "TEST"))
    print(get_time_range(df_train, "TRAIN"))
    print("="*60 + "\n")



    train_ds = MultiStepDataset(df_train, nodes, CFG.T_IN, CFG.HORIZON)
    scaler = {'mean': train_ds.means, 'std': train_ds.stds}
    val_ds  = MultiStepDataset(df_val, nodes, CFG.T_IN, CFG.HORIZON, scaler)
    test_ds = MultiStepDataset(df_test, nodes, CFG.T_IN, CFG.HORIZON, scaler)

    if len(train_ds) == 0:
        print("!!! CẢNH BÁO: Tập Train đang có 0 mẫu.")
        return

    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE)
    test_loader  = DataLoader(test_ds, batch_size=CFG.BATCH_SIZE)

    model = GAT_TCN_Model(
        num_nodes=len(nodes),
        in_feat=4,
        gat_hidden=CFG.GAT_HIDDEN,
        gat_heads=CFG.GAT_HEADS,
        tcn_channels=CFG.TCN_CHANNELS,
        tcn_kernel=CFG.TCN_KERNEL,
        horizon=CFG.HORIZON,
        output_feat=1,
        A_raw=A_raw,
        dropout=CFG.DROPOUT
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE)
    loss_fn = PureHuberLoss(delta=CFG.LOSS_DELTA)
    grad_scaler = torch.amp.GradScaler('cuda')

    print("\nStart Training...")
    best_mae = float('inf')
    patience_cnt = 0

    history = {
        'train_loss': [], 'val_loss': [],
        'train_mae': [], 'val_mae': []
    }

    for ep in range(CFG.EPOCHS):

        train_loss, train_mae = train_one_epoch(model, train_loader, optimizer, loss_fn, device, grad_scaler, scaler)

        val_metrics = evaluate(model, val_loader, device, scaler, loss_fn=loss_fn, verbose=False)
        val_mae = val_metrics['mae']
        val_loss = val_metrics['loss']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)

        print(f"Ep {ep+1:03d} | Loss: {train_loss:.4f} / {val_loss:.4f} | MAE: {train_mae:.2f} / {val_mae:.2f}", end="")

        if val_mae < best_mae:
            best_mae = val_mae
            patience_cnt = 0
            torch.save(model.state_dict(), CFG.FULL_SAVE_PATH)
            print(" -> Saved Best")
        else:
            patience_cnt += 1
            print(f" | Patience: {patience_cnt}/{CFG.PATIENCE}")
            if patience_cnt >= CFG.PATIENCE:
                print("Early Stopping")
                break

    print("\n" + "="*40)
    print("PLOTTING TRAINING HISTORY")
    print("="*40)
    plot_training_history(history['train_loss'], history['val_loss'],
                          history['train_mae'], history['val_mae'])

    print("\n" + "="*40)
    print("FINAL EVALUATION ON TEST SET")
    print("="*40)

    model.load_state_dict(torch.load(CFG.FULL_SAVE_PATH))
    test_metrics = evaluate(model, test_loader, device, scaler, loss_fn=loss_fn)

    print(f"FINAL TEST LOSS: {test_metrics['loss']:.4f}")
    print(f"FINAL TEST MAE : {test_metrics['mae']:.4f}")
    print(f"FINAL TEST MSE : {test_metrics['mse']:.4f}")
    print(f"FINAL TEST RMSE: {test_metrics['rmse']:.4f}")
    print("="*40)

    visualize_last_step(model, test_loader, device, scaler, CFG, node_list=[7, 266, 489, 89, 26, 32, 380, 365, 557])

if __name__ == "__main__":
    run_training()
