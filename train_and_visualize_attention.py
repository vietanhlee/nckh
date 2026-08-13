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
    parser.add_argument('--model_path', type=str, default=None, help="Đường dẫn tới file trọng số checkpoint (.pth). Nếu là None hoặc không tìm thấy, script sẽ tự động train lại.")
    parser.add_argument('--epochs', type=int, default=30, help="Số epochs huấn luyện nếu train từ đầu (mặc định: 100).")
    parser.add_argument('--batch_size', type=int, default=64, help="Kích thước batch size.")
    parser.add_argument('--root_dir', type=str, default="/workspace/GRAPH", help="Thư mục gốc chứa dữ liệu.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚡ Device: {device}")

    # Xử lý đường dẫn dữ liệu nếu truyền root_dir
    if args.root_dir:
        CFG.ROOT_DIR = args.root_dir
        CFG.ADJ_PATH = os.path.join(CFG.ROOT_DIR, "Graph_fix_py_3.xlsx")
        CFG.CSV_PATH = os.path.join(CFG.ROOT_DIR, "count_7_7_merg_sort_fix_fill.csv")

    if not os.path.exists(CFG.ADJ_PATH) or not os.path.exists(CFG.CSV_PATH):
        search_dirs = [
            args.root_dir if args.root_dir else None,
            getattr(CFG, 'ROOT_DIR', None),
            os.getcwd(),
        ]
        for sdir in search_dirs:
            if sdir and os.path.exists(sdir):
                adj_cand = os.path.join(sdir, "Graph_fix_py_3.xlsx")
                csv_cand = os.path.join(sdir, "count_7_7_merg_sort_fix_fill.csv")
                if os.path.exists(adj_cand) and os.path.exists(csv_cand):
                    CFG.ADJ_PATH = adj_cand
                    CFG.CSV_PATH = csv_cand
                    print(f"🔍 Tự động phát hiện file dữ liệu tại: {sdir}")
                    break

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
    
    # 2. Kiểm tra đường dẫn load model weights và tự động phát hiện kích thước channels trong Checkpoint
    candidate_paths = [
        args.model_path if args.model_path else None,
        "/workspace/nckh/model/overall_best_TA-STGCN.pth",
        os.path.join(CFG.ROOT_DIR, "checkpoints", "overall_best_TA-STGCN.pth"),
        os.path.join(os.getcwd(), "checkpoints", "overall_best_TA-STGCN.pth"),
        os.path.join(os.getcwd(), "model", "overall_best_TA-STGCN.pth")
    ]

    target_model_path = None
    state_dict = None
    block_hidden = CFG.BLOCK_HIDDEN
    num_blocks = CFG.NUM_BLOCKS
    in_feat = 5
    output_feat = 2

    for p in candidate_paths:
        if p and os.path.exists(p):
            target_model_path = p
            break
    
    if target_model_path:
        print(f"✅ Đã tìm thấy weight model! Đang nạp checkpoint từ: {target_model_path}")
        try:
            state_dict = torch.load(target_model_path, map_location=device)
            if 'blocks.0.sconv.linears.0.weight' in state_dict:
                block_hidden = state_dict['blocks.0.sconv.linears.0.weight'].shape[0]
            if 'blocks.0.tconv1.conv.weight' in state_dict:
                in_feat = state_dict['blocks.0.tconv1.conv.weight'].shape[1]
            if 'final_conv.weight' in state_dict:
                out_channels = state_dict['final_conv.weight'].shape[0]
                output_feat = out_channels // CFG.HORIZON
            print(f"   ℹ️ Checkpoint config: in_feat={in_feat}, block_hidden={block_hidden}, output_feat={output_feat}")
        except Exception as e:
            print(f"⚠️ Không thể đọc file checkpoint ({e})")
            target_model_path = None
            state_dict = None

    # Khởi tạo mô hình phù hợp 100% với cấu hình checkpoint
    model = STGCN_Model(
        num_nodes=len(nodes),
        in_feat=in_feat,
        block_hidden=block_hidden,
        num_blocks=num_blocks,
        T_in=CFG.T_IN,
        cheb_K=CFG.CHEB_K,
        horizon=CFG.HORIZON,
        output_feat=output_feat,
        L_tilde=L_tilde,
        dropout=CFG.DROPOUT,
        use_temporal_attention=True,
        attn_num_heads=4,
        attn_dropout=CFG.ATTN_DROPOUT
    ).to(device)

    if state_dict is not None:
        try:
            model.load_state_dict(state_dict)
            print(f"🎉 Đã nạp thành công 100% trọng số checkpoint!")
        except Exception as e:
            print(f"⚠️ Nạp state_dict thất bại ({e})")
            target_model_path = None

    if target_model_path is None:
        raise FileNotFoundError("❌ Không tìm thấy model best từ phase benchmark (overall_best_TA-STGCN.pth) và cũng không có model nào được truyền vào qua --model_path. Vui lòng chạy chạy benchmark trước để lấy mô hình tốt nhất!")
            
    # 3. Hook to get attention weights
    attention_weights = None
    def hook_fn(module, input, output):
        nonlocal attention_weights
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            attention_weights = output[1].detach().cpu().numpy() # Shape: (B*N, T_in, T_in)
        elif hasattr(module, 'last_attn_weights') and module.last_attn_weights is not None:
            attention_weights = module.last_attn_weights.detach().cpu().numpy()
        
    model.temporal_attn.attn.register_forward_hook(hook_fn)
    model.eval()
    
    # 4. Định nghĩa 5 khung giờ (1 mốc làm mốc chuẩn Off-Peak + 4 mốc so sánh)
    periods = {
        'Night_OffPeak': {'range': (2.0, 4.0), 'title': 'Night Off-Peak (02:00 - 04:00)', 'short': 'Off-Peak', 'idx': -1, 'type': 'min'},
        'Morning_Peak': {'range': (7.5, 9.5), 'title': 'Morning Peak (07:30 - 09:30)', 'short': 'Morning Peak', 'idx': -1, 'type': 'max'},
        'Noon_Normal': {'range': (11.5, 13.5), 'title': 'Noon Normal (11:30 - 13:30)', 'short': 'Noon Normal', 'idx': -1, 'type': 'median'},
        'Evening_Peak': {'range': (16.5, 18.5), 'title': 'Evening Peak (16:30 - 18:30)', 'short': 'Evening Peak', 'idx': -1, 'type': 'max'},
        'Late_Evening': {'range': (21.0, 23.0), 'title': 'Late Evening (21:00 - 23:00)', 'short': 'Late Evening', 'idx': -1, 'type': 'max'}
    }

    # Nạp scaler để tính toán lưu lượng thực tế (lấy trực tiếp từ train_ds)
    means = np.array(train_ds.means)[:2] # Car, Bike
    stds = np.array(train_ds.stds)[:2]

    # Quét dữ liệu để thu thập ứng viên (index và lưu lượng)
    for pkey, pinfo in periods.items():
        pinfo['candidates'] = []

    for i in range(len(test_ds)):
        hour = test_ds.time_feats[i, -1] * 24.0
        for pkey, pinfo in periods.items():
            low, high = pinfo['range']
            if low <= hour <= high:
                X, _ = test_ds[i]
                x_last = X[-1, :, :2].numpy() # (num_nodes, 2)
                x_last_unnorm = x_last * stds + means
                total_veh = x_last_unnorm.sum()
                avg_veh_per_node = total_veh / X.shape[1]
                pinfo['candidates'].append((i, avg_veh_per_node))

    # Chọn TẤT CẢ các mẫu trong khung giờ (Statistically Averaged Period)
    for pkey, pinfo in periods.items():
        if len(pinfo['candidates']) == 0:
            pinfo['indices'] = [len(test_ds) // 2]
            pinfo['avg_veh'] = 0.0
            continue
            
        pinfo['indices'] = [cand[0] for cand in pinfo['candidates']]
        pinfo['avg_veh'] = np.mean([cand[1] for cand in pinfo['candidates']])

    def get_averaged_attn_matrix(indices):
        all_mats = []
        for idx in indices:
            nonlocal attention_weights
            attention_weights = None
            X, Y = test_ds[idx]
            if isinstance(X, torch.Tensor):
                X_tensor = X.detach().clone().float().unsqueeze(0).to(device)
            else:
                X_tensor = torch.from_numpy(X).float().unsqueeze(0).to(device)
                
            X_tensor = X_tensor[..., :in_feat]
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
                row_sums = mat.sum(axis=-1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                mat = mat / row_sums
                
            all_mats.append(mat)
            
        return np.mean(all_mats, axis=0)

    # Labels thời gian (-120m đến -5m)
    time_ticks = [f"-{(24-i)*5}m" for i in range(0, 24, 3)]
    tick_indices = list(range(0, 24, 3))
    os.makedirs(CFG.PLOT_DIR, exist_ok=True)

    # Nạp ma trận Attention trung bình cho cả 5 khung giờ
    print("\n🔄 Đang tính toán Statistically Averaged Attention Matrix cho toàn bộ tập Test...")
    attn_matrices = {pkey: get_averaged_attn_matrix(pinfo['indices']) for pkey, pinfo in periods.items()}
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
        
        save_png = os.path.join(CFG.PLOT_DIR, f'attention_heatmap_{pkey.lower()}.png')
        save_pdf = os.path.join(CFG.PLOT_DIR, f'attention_heatmap_{pkey.lower()}.pdf')
        plt.savefig(save_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_pdf, dpi=300, bbox_inches='tight')
        
        paper_fig_dir = os.path.join('paper', 'fig')
        if os.path.exists(paper_fig_dir):
            plt.savefig(os.path.join(paper_fig_dir, f'attention_heatmap_{pkey.lower()}.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(paper_fig_dir, f'attention_heatmap_{pkey.lower()}.pdf'), dpi=300, bbox_inches='tight')
            
        plt.close()
        print(f"📊 Saved single heatmap for {pkey} at {save_png} and {save_pdf}")

    # B. Tạo 4 bức ảnh so sánh chuẩn 3-Subplot (Target vs Night Off-Peak vs Difference)
    targets_to_compare = ['Morning_Peak', 'Noon_Normal', 'Evening_Peak', 'Late_Evening']

    for pkey in targets_to_compare:
        pinfo = periods[pkey]
        attn_target = attn_matrices[pkey]
        attn_diff = attn_target - attn_offpeak

        # B.1. Lưu riêng 1 bức ảnh độc lập cho Difference Heatmap
        plt.figure(figsize=(9, 7))
        sns.heatmap(attn_diff, cmap='coolwarm', center=0, annot=False, cbar_kws={'label': 'Δ Attention Weight'})
        plt.title(f"Attention Difference Heatmap\n({pinfo['short']} - Off-Peak)", fontsize=13, fontweight='bold', pad=12)
        plt.xlabel('Historical Key Steps (Past Mins)', fontsize=11, labelpad=8)
        plt.ylabel('Current Query Steps (Current Mins)', fontsize=11, labelpad=8)
        plt.xticks(tick_indices, time_ticks, rotation=45, ha='right')
        plt.yticks(tick_indices, time_ticks, rotation=0)
        
        diff_png = os.path.join(CFG.PLOT_DIR, f'attention_difference_{pkey.lower()}.png')
        diff_pdf = os.path.join(CFG.PLOT_DIR, f'attention_difference_{pkey.lower()}.pdf')
        plt.savefig(diff_png, dpi=300, bbox_inches='tight')
        plt.savefig(diff_pdf, dpi=300, bbox_inches='tight')
        
        if os.path.exists(paper_fig_dir):
            plt.savefig(os.path.join(paper_fig_dir, f'attention_difference_{pkey.lower()}.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(paper_fig_dir, f'attention_difference_{pkey.lower()}.pdf'), dpi=300, bbox_inches='tight')
            
        plt.close()
        print(f"📊 Saved standalone difference heatmap for {pkey} at {diff_png}")

        # B.2. Vẽ biểu đồ 3-Subplot So sánh (Target vs Night Off-Peak vs Difference)
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
    print("\n" + "="*105)
    print("📈 BẢNG TỔNG HỢP THỐNG KÊ ĐỊNH LƯỢNG TEMPORAL ATTENTION WEIGHTS (5 KHUNG GIỜ)")
    print("="*105)
    print(f"{'Khung Giờ (Time Period)':<35} | {'Avg Vehicles/Node':<18} | {'Recent (last 20m)':<18} | {'Long-term (60-120m)':<18} | {'Ratio':<6}")
    print("-" * 105)

    for pkey, pinfo in periods.items():
        amat = attn_matrices[pkey]
        recent = np.mean(amat[:, -4:])
        longterm = np.mean(amat[:, :12])
        ratio = recent / max(longterm, 1e-6)
        print(f"{pinfo['title']:<35} | {pinfo['avg_veh']:<18.1f} | {recent:<18.4f} | {longterm:<18.4f} | {ratio:<6.2f}x")

    print("="*105)

if __name__ == "__main__":
    train_and_visualize()
