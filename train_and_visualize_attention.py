import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import argparse

# Import everything from hybrid
from hybrid import CFG, load_adj_from_excel, compute_scaled_laplacian, load_timeseries_double_rolling
from hybrid import MultiStepDataset, STGCN_Model, HuberSmoothLoss, optim
from hybrid import train_one_epoch

def train_and_visualize():
    parser = argparse.ArgumentParser(description="Trích xuất và vẽ ma trận Temporal Attention so sánh 4 khung giờ với Giờ thấp điểm Đêm.")
    parser.add_argument('--model_path', type=str, default="/kaggle/input/models/canhdoo/weight/pytorch/default/1/model_STGCN_Attn_6steps.pth", help="Đường dẫn tới file trọng số checkpoint (.pth). Nếu là None hoặc không tìm thấy, script sẽ tự động train lại.")
    parser.add_argument('--epochs', type=int, default=100, help="Số epochs huấn luyện nếu train từ đầu (mặc định: 100).")
    parser.add_argument('--batch_size', type=int, default=32, help="Kích thước batch size.")
    parser.add_argument('--root_dir', type=str, default=None, help="Thư mục gốc chứa dữ liệu.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚡ Device: {device}")

    # Xử lý đường dẫn dữ liệu nếu truyền root_dir
    if args.root_dir:
        CFG.ROOT_DIR = args.root_dir
        CFG.ADJ_PATH = os.path.join(CFG.ROOT_DIR, "Graph_fix_py_3.xlsx")
        CFG.CSV_PATH = os.path.join(CFG.ROOT_DIR, "count_7_7_merg_sort_fix_fill.csv")

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
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    
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
        attn_num_heads=2,
        attn_dropout=CFG.ATTN_DROPOUT
    ).to(device)
    
    # Kiểm tra đường dẫn load model weights
    target_model_path = args.model_path if args.model_path is not None else CFG.FULL_SAVE_PATH
    
    if target_model_path and os.path.exists(target_model_path):
        print(f"✅ Đã tìm thấy weight model! Đang nạp checkpoint từ: {target_model_path}")
        model.load_state_dict(torch.load(target_model_path, map_location=device))
    else:
        print(f"⚠️ Không tìm thấy file weight (hoặc path=None). Bắt đầu huấn luyện mô hình {args.epochs} epochs từ đầu...")
        optimizer = optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE)
        loss_fn = HuberSmoothLoss(delta=CFG.LOSS_DELTA, smooth_weight=CFG.SMOOTH_LOSS_WEIGHT)
        scaler_obj = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        for epoch in range(1, args.epochs + 1):
            if scaler_obj is None:
                model.train()
                total_loss = 0.0
                for X, Y in train_loader:
                    X, Y = X.to(device), Y.to(device)
                    x_last = X[:, -1, :, :1].unsqueeze(1)
                    optimizer.zero_grad()
                    pred = model(X)
                    loss = loss_fn(pred, Y, x_last)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                if epoch % 10 == 0 or epoch == 1:
                    print(f"   Epoch {epoch:03d}/{args.epochs} | Loss: {total_loss/len(train_loader):.4f}")
            else:
                train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler_obj, scaler)
                if epoch % 10 == 0 or epoch == 1:
                    print(f"   Epoch {epoch:03d}/{args.epochs} completed.")
        
        save_path = target_model_path if target_model_path else CFG.FULL_SAVE_PATH
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"✅ Đã lưu checkpoint mô hình vào: {save_path}")
            
    # 3. Hook to get attention weights
    attention_weights = None
    def hook_fn(module, input, output):
        nonlocal attention_weights
        attention_weights = output[1].detach().cpu().numpy() # Shape: (B*N, T_in, T_in)
        
    model.temporal_attn.attn.register_forward_hook(hook_fn)
    model.eval()
    
    # 4. Định nghĩa 5 khung giờ (1 mốc làm mốc chuẩn Off-Peak + 4 mốc so sánh)
    periods = {
        'Night_OffPeak': {'range': (2.0, 4.0), 'title': 'Night Off-Peak (02:00 - 04:00)', 'short': 'Off-Peak', 'idx': -1},
        'Morning_Peak': {'range': (7.5, 9.5), 'title': 'Morning Peak (07:30 - 09:30)', 'short': 'Morning Peak', 'idx': -1},
        'Noon_Normal': {'range': (11.5, 13.5), 'title': 'Noon Normal (11:30 - 13:30)', 'short': 'Noon Normal', 'idx': -1},
        'Evening_Peak': {'range': (16.5, 18.5), 'title': 'Evening Peak (16:30 - 18:30)', 'short': 'Evening Peak', 'idx': -1},
        'Late_Evening': {'range': (21.0, 23.0), 'title': 'Late Evening (21:00 - 23:00)', 'short': 'Late Evening', 'idx': -1}
    }

    for i in range(len(test_ds)):
        hour = test_ds.time_feats[i, -1] * 24.0
        for pkey, pinfo in periods.items():
            if pinfo['idx'] == -1:
                low, high = pinfo['range']
                if low <= hour <= high:
                    pinfo['idx'] = i

    for pkey, pinfo in periods.items():
        if pinfo['idx'] == -1:
            pinfo['idx'] = len(test_ds) // 2

    def get_attn_matrix(idx):
        X, Y = test_ds[idx]
        X_tensor = torch.tensor(X).unsqueeze(0).to(device)
        with torch.no_grad():
            model(X_tensor)
        return np.mean(attention_weights, axis=0) # (T_in, T_in)

    # Labels thời gian (-120m đến -5m)
    time_ticks = [f"-{(24-i)*5}m" for i in range(0, 24, 3)]
    tick_indices = list(range(0, 24, 3))
    os.makedirs(CFG.PLOT_DIR, exist_ok=True)

    # Nạp ma trận Attention cho cả 5 khung giờ
    attn_matrices = {pkey: get_attn_matrix(pinfo['idx']) for pkey, pinfo in periods.items()}
    attn_offpeak = attn_matrices['Night_OffPeak']

    # A. Lưu ảnh đơn lẻ cho cả 5 khung giờ
    for pkey, pinfo in periods.items():
        attn_mat = attn_matrices[pkey]
        plt.figure(figsize=(9, 7))
        sns.heatmap(attn_mat, cmap='viridis', annot=False, cbar_kws={'label': 'Attention Weight'})
        plt.title(f"Global Temporal Attention Heatmap\n{pinfo['title']}", fontsize=13, fontweight='bold', pad=12)
        plt.xlabel('Historical Key Steps (Past Mins)', fontsize=11, labelpad=8)
        plt.ylabel('Query Time Steps (Current Mins)', fontsize=11, labelpad=8)
        plt.xticks(tick_indices, time_ticks, rotation=45, ha='right')
        plt.yticks(tick_indices, time_ticks, rotation=0)
        
        save_path = os.path.join(CFG.PLOT_DIR, f'attention_heatmap_{pkey.lower()}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved single heatmap for {pkey} at {save_path}")

    # B. Tạo 4 phiên bản so sánh (Mỗi phiên bản gồm 3 ảnh đơn lẻ + 1 ảnh ghép 3-Subplot)
    targets_to_compare = ['Morning_Peak', 'Noon_Normal', 'Evening_Peak', 'Late_Evening']

    for pkey in targets_to_compare:
        pinfo = periods[pkey]
        attn_target = attn_matrices[pkey]
        attn_diff = attn_target - attn_offpeak

        # B1. Lưu ảnh Difference Heatmap đơn lẻ
        plt.figure(figsize=(9, 7))
        sns.heatmap(attn_diff, cmap='coolwarm', center=0, annot=False, cbar_kws={'label': 'Δ Attention Weight'})
        plt.title(f"Temporal Attention Difference Heatmap\n{pinfo['title']} vs. Night Off-Peak", fontsize=13, fontweight='bold', pad=12)
        plt.xlabel('Historical Key Steps (Past Mins)', fontsize=11, labelpad=8)
        plt.ylabel('Query Time Steps (Current Mins)', fontsize=11, labelpad=8)
        plt.xticks(tick_indices, time_ticks, rotation=45, ha='right')
        plt.yticks(tick_indices, time_ticks, rotation=0)
        
        diff_single_png = os.path.join(CFG.PLOT_DIR, f'attention_difference_{pkey.lower()}.png')
        plt.savefig(diff_single_png, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved single difference heatmap for {pkey} at {diff_single_png}")

        # B2. Lưu ảnh ghép 3-Subplot (a: Target, b: Off-Peak, c: Difference)
        fig, axes = plt.subplots(1, 3, figsize=(22, 6))

        # (a) Target Period
        sns.heatmap(attn_target, ax=axes[0], cmap='viridis', cbar_kws={'label': 'Weight'})
        axes[0].set_title(f"(a) {pinfo['title']}", fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Historical Key Steps', fontsize=10, labelpad=6)
        axes[0].set_ylabel('Current Query Steps', fontsize=10, labelpad=6)
        axes[0].set_xticks(tick_indices)
        axes[0].set_xticklabels(time_ticks, rotation=45, ha='right')
        axes[0].set_yticks(tick_indices)
        axes[0].set_yticklabels(time_ticks)

        # (b) Night Off-Peak
        sns.heatmap(attn_offpeak, ax=axes[1], cmap='viridis', cbar_kws={'label': 'Weight'})
        axes[1].set_title(f"(b) {periods['Night_OffPeak']['title']}", fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Historical Key Steps', fontsize=10, labelpad=6)
        axes[1].set_ylabel('Current Query Steps', fontsize=10, labelpad=6)
        axes[1].set_xticks(tick_indices)
        axes[1].set_xticklabels(time_ticks, rotation=45, ha='right')
        axes[1].set_yticks(tick_indices)
        axes[1].set_yticklabels(time_ticks)

        # (c) Difference (Target - Off-Peak)
        sns.heatmap(attn_diff, ax=axes[2], cmap='coolwarm', center=0, cbar_kws={'label': 'Δ Weight'})
        axes[2].set_title(f"(c) Difference ({pinfo['short']} - Off-Peak)", fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Historical Key Steps', fontsize=10, labelpad=6)
        axes[2].set_ylabel('Current Query Steps', fontsize=10, labelpad=6)
        axes[2].set_xticks(tick_indices)
        axes[2].set_xticklabels(time_ticks, rotation=45, ha='right')
        axes[2].set_yticks(tick_indices)
        axes[2].set_yticklabels(time_ticks)

        plt.tight_layout()
        cmp_png = os.path.join(CFG.PLOT_DIR, f'attention_comparison_{pkey.lower()}.png')
        cmp_pdf = os.path.join(CFG.PLOT_DIR, f'attention_comparison_{pkey.lower()}.pdf')
        plt.savefig(cmp_png, dpi=300, bbox_inches='tight')
        plt.savefig(cmp_pdf, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"🖼️ Saved 3-subplot comparison for {pkey} at {cmp_png} and {cmp_pdf}")

    # C. Trích xuất thống kê định lượng cho cả 5 khung giờ
    print("\n" + "="*85)
    print("📈 BẢNG TỔNG HỢP THỐNG KÊ ĐỊNH LƯỢNG TEMPORAL ATTENTION WEIGHTS (5 KHUNG GIỜ)")
    print("="*85)
    print(f"{'Khung Giờ (Time Period)':<35} | {'Recent (last 20m)':<18} | {'Long-term (60-120m)':<18} | {'Ratio':<6}")
    print("-" * 85)

    for pkey, pinfo in periods.items():
        amat = attn_matrices[pkey]
        recent = np.mean(amat[:, -4:])
        longterm = np.mean(amat[:, :12])
        ratio = recent / max(longterm, 1e-6)
        print(f"{pinfo['title']:<35} | {recent:<18.4f} | {longterm:<18.4f} | {ratio:<6.2f}x")

    print("="*85)

if __name__ == "__main__":
    train_and_visualize()
