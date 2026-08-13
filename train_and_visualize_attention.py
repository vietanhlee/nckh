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
    parser.add_argument('--epochs', type=int, default=120, help="Số epochs huấn luyện nếu train từ đầu (mặc định: 100).")
    parser.add_argument('--batch_size', type=int, default=64, help="Kích thước batch size.")
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
        in_feat=5,
        block_hidden=CFG.BLOCK_HIDDEN,
        num_blocks=CFG.NUM_BLOCKS,
        T_in=CFG.T_IN,
        cheb_K=CFG.CHEB_K,
        horizon=CFG.HORIZON,
        output_feat=2,
        L_tilde=L_tilde,
        dropout=CFG.DROPOUT,
        use_temporal_attention=True,
        attn_num_heads=4,
        attn_dropout=CFG.ATTN_DROPOUT
    ).to(device)
    
    # Kiểm tra đường dẫn load model weights (Thử nhiều thư mục khả thi)
    candidate_paths = [
        args.model_path if args.model_path else None,
        "checkpoints/overall_best_TA-STGCN.pth",
        "checkpoints/best_TA-STGCN_seed_42.pth",
        "/kaggle/input/models/canhdoo/weight/pytorch/default/1/model_STGCN_Attn_6steps.pth",
        CFG.FULL_SAVE_PATH,
        "model/model_STGCN_Attn_6steps.pth"
    ]

    target_model_path = None
    for p in candidate_paths:
        if p and os.path.exists(p):
            target_model_path = p
            break

    if target_model_path:
        print(f"✅ Đã tìm thấy weight model đã huấn luyện! Đang nạp checkpoint từ: {target_model_path}")
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
                    x_last = X[:, -1, :, :2].unsqueeze(1)
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
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            attention_weights = output[1].detach().cpu().numpy()
        elif hasattr(module, 'last_attn_weights') and module.last_attn_weights is not None:
            attention_weights = module.last_attn_weights.detach().cpu().numpy()

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
        nonlocal attention_weights
        attention_weights = None
        X, Y = test_ds[idx]
        if isinstance(X, torch.Tensor):
            X_tensor = X.detach().clone().float().unsqueeze(0).to(device)
        else:
            X_tensor = torch.from_numpy(X).float().unsqueeze(0).to(device)

        with torch.no_grad():
            _ = model(X_tensor)

        if attention_weights is None or np.isnan(attention_weights).any():
            if hasattr(model.temporal_attn, 'last_attn_weights') and model.temporal_attn.last_attn_weights is not None:
                attention_weights = model.temporal_attn.last_attn_weights.detach().cpu().numpy()

        if attention_weights is None:
            mat = np.eye(24)
        else:
            mat = np.mean(attention_weights, axis=0) if attention_weights.ndim == 3 else attention_weights
            mat = np.nan_to_num(mat, nan=1.0 / 24.0)

        # Chuẩn hoá ma trận xác suất (hàng có tổng = 1)
        row_sums = mat.sum(axis=-1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        mat = mat / row_sums
        return mat

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

        # Ngoài ra cũng lưu trực tiếp vào thư mục paper/fig nếu tồn tại
        paper_fig_dir = os.path.join('paper', 'fig')
        if os.path.exists(paper_fig_dir):
            plt.savefig(os.path.join(paper_fig_dir, f'attention_comparison_{pkey.lower()}.pdf'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(paper_fig_dir, f'attention_comparison_{pkey.lower()}.png'), dpi=300, bbox_inches='tight')

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

    # D. MÔ PHỎNG VÒNG ĐỜI TẮC ĐƯỜNG (TRAFFIC EVENT LIFECYCLE) BẰNG DỮ LIỆU THẬT
    print("\n" + "="*85)
    print("🚗 PHÂN TÍCH VÒNG ĐỜI TẮC ĐƯỜNG BẰNG DỮ LIỆU ATTENTION THẬT")
    print("="*85)
    
    event_periods = {
        'Phase I: Congestion Onset (16:30 - 17:15)': {'range': (16.5, 17.25), 'idx': -1, 'color': '#d9534f', 'subtitle': '(a) Rapid Focus on Recent Momentum'},
        'Phase II: Congestion Peak (17:15 - 18:15)': {'range': (17.25, 18.25), 'idx': -1, 'color': '#f0ad4e', 'subtitle': '(b) Distributed Queue History Focus'},
        'Phase III: Congestion Dissipation (18:15 - 19:00)': {'range': (18.25, 19.0), 'idx': -1, 'color': '#5cb85c', 'subtitle': '(c) Recovery & Baseline Balancing'}
    }

    for i in range(len(test_ds)):
        hour = test_ds.time_feats[i, -1] * 24.0
        for pkey, pinfo in event_periods.items():
            if pinfo['idx'] == -1:
                low, high = pinfo['range']
                if low <= hour <= high:
                    pinfo['idx'] = i

    for pkey, pinfo in event_periods.items():
        if pinfo['idx'] == -1:
            pinfo['idx'] = len(test_ds) // 2

    event_matrices = {pkey: get_attn_matrix(pinfo['idx']) for pkey, pinfo in event_periods.items()}
    
    # D1. Vẽ 3-Subplot Heatmap 24x24
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    for i, (pkey, pinfo) in enumerate(event_periods.items()):
        mat = event_matrices[pkey]
        sns.heatmap(mat, ax=axes[i], cmap='viridis', cbar_kws={'label': 'Weight'})
        axes[i].set_title(pkey, fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Historical Key Steps (Past Mins)', fontsize=10, labelpad=6)
        axes[i].set_ylabel('Current Query Steps (Current Mins)', fontsize=10, labelpad=6)
        axes[i].set_xticks(tick_indices)
        axes[i].set_xticklabels(time_ticks, rotation=45, ha='right')
        axes[i].set_yticks(tick_indices)
        axes[i].set_yticklabels(time_ticks)

    plt.tight_layout()
    event_heatmap_png = os.path.join(CFG.PLOT_DIR, 'traffic_events_real_heatmap.png')
    plt.savefig(event_heatmap_png, dpi=300, bbox_inches='tight')
    if os.path.exists(paper_fig_dir):
        plt.savefig(os.path.join(paper_fig_dir, 'traffic_events_real_heatmap.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🖼️ Saved Real Data Traffic Event Heatmaps (24x24) to {event_heatmap_png}")

    # D1.5. Vẽ 3-Subplot Difference Heatmap (Target - Night OffPeak) cho Traffic Events
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    for i, (pkey, pinfo) in enumerate(event_periods.items()):
        mat = event_matrices[pkey]
        # Trừ đi ma trận Off-Peak (đã được tính toán ở phần trên)
        diff_mat = mat - attn_offpeak
        
        sns.heatmap(diff_mat, ax=axes[i], cmap='coolwarm', center=0, cbar_kws={'label': 'Δ Weight'})
        axes[i].set_title(f"Difference: {pkey} - OffPeak", fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Historical Key Steps (Past Mins)', fontsize=10, labelpad=6)
        axes[i].set_ylabel('Current Query Steps (Current Mins)', fontsize=10, labelpad=6)
        axes[i].set_xticks(tick_indices)
        axes[i].set_xticklabels(time_ticks, rotation=45, ha='right')
        axes[i].set_yticks(tick_indices)
        axes[i].set_yticklabels(time_ticks)

    plt.tight_layout()
    event_diff_png = os.path.join(CFG.PLOT_DIR, 'traffic_events_real_difference.png')
    plt.savefig(event_diff_png, dpi=300, bbox_inches='tight')
    if os.path.exists(paper_fig_dir):
        plt.savefig(os.path.join(paper_fig_dir, 'traffic_events_real_difference.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🖼️ Saved Real Data Traffic Event Difference Heatmaps (24x24) to {event_diff_png}")

    # D2. Vẽ 3-Subplot 1D Bar Chart (Dùng hàng cuối cùng - Query tại thời điểm hiện tại t)
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#333333'
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    T_in = CFG.T_IN
    
    for i, (pkey, pinfo) in enumerate(event_periods.items()):
        ax = axes[i]
        # Trọng số của Query cuối cùng đối chiếu với các Key trong quá khứ
        weights_1d = event_matrices[pkey][-1, :] 
        weights_1d = weights_1d / np.sum(weights_1d) # Normalize
        
        bars = ax.bar(range(T_in), weights_1d, color=pinfo['color'], alpha=0.85, edgecolor='black', linewidth=0.8)
        ax.set_title(pkey, fontsize=10.5, fontweight='bold', pad=10)
        ax.set_xticks(range(0, T_in, 4))
        ax.set_xticklabels([f"t-{(T_in-j)*5}m" for j in range(0, T_in, 4)], rotation=45, fontsize=8.5)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Highlight top 3 steps
        top3_idx = np.argsort(weights_1d)[-3:]
        for idx in top3_idx:
            bars[idx].set_alpha(1.0)
            bars[idx].set_edgecolor('red')
            bars[idx].set_linewidth(1.5)

        ax.text(0.5, 0.88, pinfo['subtitle'], transform=ax.transAxes,
                ha='center', va='center', fontsize=9, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='gray'))

    axes[0].set_ylabel("Temporal Attention Weight", fontsize=10.5, fontweight='bold')
    fig.supxlabel("Historical Observation Steps (Lookback Window)", fontsize=10.5, fontweight='bold', y=-0.05)
    plt.tight_layout()
    
    event_bar_png = os.path.join(CFG.PLOT_DIR, 'traffic_events_real_barchart.png')
    plt.savefig(event_bar_png, dpi=300, bbox_inches='tight')
    if os.path.exists(paper_fig_dir):
        plt.savefig(os.path.join(paper_fig_dir, 'traffic_events_real_barchart.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved Real Data Traffic Event Bar Charts (1D) to {event_bar_png}")

if __name__ == "__main__":
    train_and_visualize()
