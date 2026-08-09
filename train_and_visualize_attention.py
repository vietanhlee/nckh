import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

# Import everything from hybrid
from hybrid import CFG, load_adj_from_excel, compute_scaled_laplacian, load_timeseries_double_rolling
from hybrid import MultiStepDataset, STGCN_Model, HuberSmoothLoss, optim
from hybrid import train_one_epoch

def train_and_visualize():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Load data
    A_raw, nodes = load_adj_from_excel(CFG.ADJ_PATH)
    L_tilde = compute_scaled_laplacian(A_raw)
    
    print("Loading data...")
    df_all = load_timeseries_double_rolling(CFG.CSV_PATH, nodes, CFG.DATA_WINDOW1, CFG.DATA_WINDOW2, CFG.TIME_STEP_MINUTES)
    
    n_total = len(df_all)
    n_train = int(n_total * 0.8)
    idx_val_end = n_train + int(n_total * 0.1)
    
    df_train = df_all.iloc[:n_train]
    df_test = df_all.iloc[idx_val_end:]
    
    train_ds = MultiStepDataset(df_train, nodes, CFG.T_IN, CFG.HORIZON)
    scaler = {'mean': train_ds.means, 'std': train_ds.stds}
    test_ds = MultiStepDataset(df_test, nodes, CFG.T_IN, CFG.HORIZON, scaler)
    
    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True)
    
    # 2. Init model
    model = STGCN_Model(
        num_nodes=len(nodes),
        in_feat=4,
        block_hidden=CFG.BLOCK_HIDDEN,
        num_blocks=CFG.NUM_BLOCKS,
        T_in=CFG.T_IN,
        cheb_K=CFG.CHEB_K,
        horizon=CFG.HORIZON,
        output_feat=1,
        L_tilde=L_tilde,
        dropout=CFG.DROPOUT,
        use_temporal_attention=True,
        attn_num_heads=CFG.ATTN_NUM_HEADS,
        attn_dropout=CFG.ATTN_DROPOUT
    ).to(device)
    
    # Train or load weights
    model_path = CFG.FULL_SAVE_PATH
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path} for visualization...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Training model for 2 epochs just for visualization (since weights not found)...")
        print("To get a perfect heatmap, please train the model fully using run_training() first.")
        optimizer = optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE)
        loss_fn = HuberSmoothLoss(delta=CFG.LOSS_DELTA, smooth_weight=CFG.SMOOTH_LOSS_WEIGHT)
        scaler_obj = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        for epoch in range(2):
            print(f"Epoch {epoch+1}/2...")
            # For CPU, bypass amp grad scaler
            if scaler_obj is None:
                model.train()
                for X, Y in train_loader:
                    X, Y = X.to(device), Y.to(device)
                    x_last = X[:, -1, :, :1].unsqueeze(1)
                    optimizer.zero_grad()
                    pred = model(X)
                    loss = loss_fn(pred, Y, x_last)
                    loss.backward()
                    optimizer.step()
            else:
                train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler_obj, scaler)
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"✅ Đã lưu checkpoint mô hình vào: {model_path}")
            
    # 3. Hook to get attention weights
    attention_weights = None
    def hook_fn(module, input, output):
        nonlocal attention_weights
        # nn.MultiheadAttention returns (attn_output, attn_output_weights)
        attention_weights = output[1].detach().cpu().numpy() # Shape: (B*N, T_in, T_in)
        
    # Register the hook on the MultiheadAttention layer
    model.temporal_attn.attn.register_forward_hook(hook_fn)
    model.eval()
    
    # 4. Find peak and off-peak times from test set
    peak_idx = -1
    offpeak_idx = -1
    
    for i in range(len(test_ds)):
        hour = test_ds.time_feats[i, -1] * 24
        # 17:00 - 18:00
        if 17 <= hour <= 18 and peak_idx == -1:
            peak_idx = i
        # 02:00 - 03:00
        if 2 <= hour <= 3 and offpeak_idx == -1:
            offpeak_idx = i
            
        if peak_idx != -1 and offpeak_idx != -1:
            break
            
    if peak_idx == -1: peak_idx = 0
    if offpeak_idx == -1: offpeak_idx = len(test_ds) // 2
    
    # 5. Extract attention heatmaps for Peak, Off-Peak, and Difference
    def get_attn_matrix(idx):
        X, Y = test_ds[idx]
        X_tensor = torch.tensor(X).unsqueeze(0).to(device)
        with torch.no_grad():
            model(X_tensor)
        return np.mean(attention_weights, axis=0) # (T_in, T_in)

    attn_peak = get_attn_matrix(peak_idx)
    attn_offpeak = get_attn_matrix(offpeak_idx)
    attn_diff = attn_peak - attn_offpeak

    # Labels cho trục thời gian (từ -120 phút đến -5 phút)
    time_ticks = [f"-{(24-i)*5}m" for i in range(0, 24, 3)]
    tick_indices = list(range(0, 24, 3))

    os.makedirs(CFG.PLOT_DIR, exist_ok=True)

    # A. Vẽ ảnh đơn lẻ cho Peak và Off-Peak
    def save_single_heatmap(attn_mat, label, title, cmap='viridis'):
        plt.figure(figsize=(9, 7))
        ax = sns.heatmap(attn_mat, cmap=cmap, annot=False, cbar_kws={'label': 'Attention Weight'})
        plt.title(title, fontsize=13, fontweight='bold', pad=12)
        plt.xlabel('Historical Key Steps (Past Mins)', fontsize=11)
        plt.ylabel('Query Time Steps (Current Mins)', fontsize=11)
        plt.xticks(tick_indices, time_ticks, rotation=0)
        plt.yticks(tick_indices, time_ticks, rotation=0)
        save_path = os.path.join(CFG.PLOT_DIR, f'attention_heatmap_{label.lower()}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved heatmap for {label} at {save_path}")

    save_single_heatmap(attn_peak, 'Peak_Hour', 'Temporal Attention Heatmap (Peak Hour: 17:00 - 18:00)')
    save_single_heatmap(attn_offpeak, 'OffPeak_Hour', 'Temporal Attention Heatmap (Off-Peak Hour: 02:00 - 03:00)')

    # B. Vẽ ảnh ghép 3 Subplots chuẩn IEEE Paper (Peak vs Off-Peak vs Difference)
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    sns.heatmap(attn_peak, ax=axes[0], cmap='viridis', cbar_kws={'label': 'Weight'})
    axes[0].set_title('(a) Peak Hour Attention (17:00 - 18:00)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Historical Key Steps', fontsize=10)
    axes[0].set_ylabel('Current Query Steps', fontsize=10)
    axes[0].set_xticks(tick_indices)
    axes[0].set_xticklabels(time_ticks)
    axes[0].set_yticks(tick_indices)
    axes[0].set_yticklabels(time_ticks)

    sns.heatmap(attn_offpeak, ax=axes[1], cmap='viridis', cbar_kws={'label': 'Weight'})
    axes[1].set_title('(b) Off-Peak Attention (02:00 - 03:00)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Historical Key Steps', fontsize=10)
    axes[1].set_ylabel('Current Query Steps', fontsize=10)
    axes[1].set_xticks(tick_indices)
    axes[1].set_xticklabels(time_ticks)
    axes[1].set_yticks(tick_indices)
    axes[1].set_yticklabels(time_ticks)

    sns.heatmap(attn_diff, ax=axes[2], cmap='coolwarm', center=0, cbar_kws={'label': 'Δ Weight'})
    axes[2].set_title('(c) Difference (Peak - Off-Peak)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Historical Key Steps', fontsize=10)
    axes[2].set_ylabel('Current Query Steps', fontsize=10)
    axes[2].set_xticks(tick_indices)
    axes[2].set_xticklabels(time_ticks)
    axes[2].set_yticks(tick_indices)
    axes[2].set_yticklabels(time_ticks)

    plt.tight_layout()
    comparison_path = os.path.join(CFG.PLOT_DIR, 'attention_heatmap_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🖼️ Saved 3-subplot IEEE comparison figure at {comparison_path}")

    # C. Trích xuất thống kê định lượng (Quantitative Statistics)
    recent_peak = np.mean(attn_peak[:, -4:]) # 20 phút gần nhất (t-4 đến t)
    longterm_peak = np.mean(attn_peak[:, :12]) # 60-120 phút trước

    recent_offpeak = np.mean(attn_offpeak[:, -4:])
    longterm_offpeak = np.mean(attn_offpeak[:, :12])

    print("\n" + "="*70)
    print("📈 KẾT QUẢ PHÂN TÍCH ĐỊNH LƯỢNG TEMPORAL ATTENTION WEIGHTS")
    print("="*70)
    print(f"🔴 Peak Hour   | Recent (last 20m): {recent_peak:.4f} | Long-term (60-120m): {longterm_peak:.4f} | Ratio: {recent_peak/longterm_peak:.2f}x")
    print(f"🔵 Off-Peak    | Recent (last 20m): {recent_offpeak:.4f} | Long-term (60-120m): {longterm_offpeak:.4f} | Ratio: {recent_offpeak/longterm_offpeak:.2f}x")
    print("="*70)

if __name__ == "__main__":
    train_and_visualize()
