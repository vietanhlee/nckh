import os
import gc
import json
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Import Stage 1 dependencies
from train_counting import get_dataset, build_counting_model
from benchmark_5seeds import set_seed

# Import Stage 2 dependencies
from stgcn import STGCN_Model as Baseline_STGCN_Model, Config as BaselineConfig, normalize_adj_sym
from hybrid import STGCN_Model as Hybrid_STGCN_Model, Config as HybridConfig
from advanced_baselines import GraphWaveNet, ASTGCN, GMAN, AGCRN
from sota_2023_baselines import STAEformerProxy, MegaCRNProxy, DSTAGNNProxy, iTransformerProxy
from stgcn import (
    load_adj_from_excel,
    compute_scaled_laplacian,
    load_timeseries_double_rolling,
    MultiStepDataset,
)

def format_mean_std(data_list):
    if not data_list: return "-"
    mean = np.mean(data_list)
    std = np.std(data_list, ddof=1) if len(data_list) > 1 else 0.0
    return f"{mean:.4f} ± {std:.4f}"

def format_mean_std_bias(data_list):
    if not data_list: return "-"
    mean = np.mean(data_list)
    std = np.std(data_list, ddof=1) if len(data_list) > 1 else 0.0
    prefix = "+" if mean > 0 else ""
    return f"{prefix}{mean:.2f} ± {std:.2f}"

def evaluate_stage1_density(args):
    print("="*80)
    print("🚀 GIAI ĐOẠN 1 (STAGE 1): PHÂN TÍCH ĐÁNH GIÁ MÔ HÌNH COUNTING THEO MỨC MẬT ĐỘ")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = args.seeds
    models = ['resnet', 'efficientnet', 'vit', 'convnext', 'mobilenet']
    
    # Load dataset for stage 1
    from train_counting import Config as CountingConfig
    csv_file = CountingConfig.CSV_FILE
    image_dir = CountingConfig.IMAGE_DIR
    if not os.path.exists(csv_file):
        local_csv = os.path.join(os.getcwd(), "labels1.csv")
        local_img = os.path.join(os.getcwd(), "images")
        if os.path.exists(local_csv):
            csv_file = local_csv
            image_dir = local_img

    full_dataset = get_dataset(image_dir, csv_file, is_train=False)
    
    n_total = len(full_dataset)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    n_test = n_total - n_train - n_val

    md_content = "# Báo cáo Đánh giá Theo Mật độ (Stage 1)\n\n"
    md_content += "Mật độ: Low (<10), Medium (10-25), High (>25) xe/camera.\n\n"
    
    plot_data = []

    for model_name in models:
        print(f"\n📌 ĐÁNH GIÁ MÔ HÌNH STAGE 1: {model_name.upper()}")
        
        results = {
            'Low (<10)': {'mae_car': [], 'mae_moto': [], 'bias_car': [], 'bias_moto': [], 'counts': []},
            'Medium (10-25)': {'mae_car': [], 'mae_moto': [], 'bias_car': [], 'bias_moto': [], 'counts': []},
            'High (>25)': {'mae_car': [], 'mae_moto': [], 'bias_car': [], 'bias_moto': [], 'counts': []}
        }

        for seed in seeds:
            print(f"   --- Đang chạy đánh giá cho Seed {seed} ---")
            set_seed(seed)
            torch.manual_seed(seed) 
            _, _, test_ds = torch.utils.data.random_split(full_dataset, [n_train, n_val, n_test])
            
            test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)
            
            possible_paths = [
                os.path.join(args.root_dir, "checkpoints", f"best_counting_{model_name}_seed_{seed}.pth"),
                os.path.join(args.root_dir, "model", f"best_counting_{model_name}_seed_{seed}.pth"),
                os.path.join(args.root_dir, "checkpoints", f"temp_counting_{model_name}_seed_{seed}.pth"),
                f"checkpoints/best_counting_{model_name}_seed_{seed}.pth",
                f"model/best_counting_{model_name}_seed_{seed}.pth"
            ]
            
            checkpoint_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    checkpoint_path = p
                    break
                    
            if checkpoint_path is None:
                print(f"   ⚠️ Không tìm thấy trọng số cho seed {seed}. Bỏ qua seed này!")
                continue
                
            model = build_counting_model(model_name, num_classes=2, pretrained=False).to(device)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
            model.eval()

            test_preds, test_trues = [], []
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    outputs = model(images)
                    test_preds.append(outputs.cpu().numpy())
                    test_trues.append(labels.numpy())

            test_preds = np.vstack(test_preds)
            test_trues = np.vstack(test_trues)

            total_vehicles = test_trues[:, 0] + test_trues[:, 1]

            masks = {
                'Low (<10)': total_vehicles < 10,
                'Medium (10-25)': (total_vehicles >= 10) & (total_vehicles <= 25),
                'High (>25)': total_vehicles > 25
            }

            for level, mask in masks.items():
                count = np.sum(mask)
                results[level]['counts'].append(count)
                
                if count == 0: continue
                
                preds_level = test_preds[mask]
                trues_level = test_trues[mask]

                err = preds_level - trues_level 
                results[level]['mae_car'].append(np.mean(np.abs(err[:, 0])))
                results[level]['mae_moto'].append(np.mean(np.abs(err[:, 1])))
                results[level]['bias_car'].append(np.mean(err[:, 0]))
                results[level]['bias_moto'].append(np.mean(err[:, 1]))

        md_content += f"## {model_name.upper()}\n\n"
        md_content += "| Mức mật độ | Số lượng ảnh (TB) | MAE Xe ô tô | MAE Xe máy | Độ chệch Ô tô | Độ chệch Xe máy |\n"
        md_content += "|---|---|---|---|---|---|\n"

        for level in ['Low (<10)', 'Medium (10-25)', 'High (>25)']:
            avg_count = int(np.mean(results[level]['counts'])) if results[level]['counts'] else 0
            mae_car_str = format_mean_std(results[level]['mae_car'])
            mae_moto_str = format_mean_std(results[level]['mae_moto'])
            bias_car_str = format_mean_std_bias(results[level]['bias_car'])
            bias_moto_str = format_mean_std_bias(results[level]['bias_moto'])
            
            md_content += f"| **{level}** | {avg_count} | {mae_car_str} | {mae_moto_str} | {bias_car_str} | {bias_moto_str} |\n"
            
            if results[level]['mae_car']:
                plot_data.append({
                    'Model': model_name.upper(),
                    'Density': level.split()[0],
                    'Vehicle': 'Car',
                    'MAE': np.mean(results[level]['mae_car'])
                })
            if results[level]['mae_moto']:
                plot_data.append({
                    'Model': model_name.upper(),
                    'Density': level.split()[0],
                    'Vehicle': 'Motorcycle',
                    'MAE': np.mean(results[level]['mae_moto'])
                })

    with open("eval_stage1_density_report.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    
    print("\n✅ Đã lưu kết quả Stage 1 vào eval_stage1_density_report.md")

    if plot_data:
        df_plot = pd.DataFrame(plot_data)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_plot, x='Density', y='MAE', hue='Model', 
                    palette='Set2', errorbar=None)
        plt.title('Stage 1 MAE by Density Level (Average over 5 seeds)', fontsize=14, fontweight='bold')
        plt.ylabel('Mean Absolute Error (MAE)', fontsize=12)
        plt.xlabel('Density Level', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        os.makedirs('plots', exist_ok=True)
        fig_png = os.path.join('plots', 'stage1_density_mae.png')
        fig_pdf = os.path.join('plots', 'stage1_density_mae.pdf')
        plt.savefig(fig_png)
        plt.savefig(fig_pdf, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"📊 Đã lưu biểu đồ Stage 1 MAE: {fig_png} (và .pdf)")

def evaluate_stage2_density(args):
    print("\n"+"="*80)
    print("🚀 GIAI ĐOẠN 2 (STAGE 2): PHÂN TÍCH LAN TRUYỀN LỖI TRONG DỰ BÁO (ERROR PROPAGATION)")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = args.seeds
    
    stgcn_cfg = BaselineConfig()
    stgcn_cfg.BLOCK_HIDDEN = 80
    stgcn_cfg.NUM_BLOCKS = 3
    stgcn_cfg.ROOT_DIR = args.root_dir
    stgcn_cfg.ADJ_PATH = os.path.join(args.root_dir, "Graph_fix_py_3.xlsx")
    stgcn_cfg.CSV_PATH = os.path.join(args.root_dir, "count_7_7_merg_sort_fix_fill.csv")
    
    hybrid_cfg = HybridConfig()
    hybrid_cfg.BLOCK_HIDDEN = 80
    hybrid_cfg.NUM_BLOCKS = 3
    hybrid_cfg.ROOT_DIR = stgcn_cfg.ROOT_DIR
    hybrid_cfg.ADJ_PATH = stgcn_cfg.ADJ_PATH
    hybrid_cfg.CSV_PATH = stgcn_cfg.CSV_PATH
    
    A_raw, nodes = load_adj_from_excel(stgcn_cfg.ADJ_PATH)
    L_tilde = compute_scaled_laplacian(A_raw)
    
    df_all = load_timeseries_double_rolling(
        stgcn_cfg.CSV_PATH, nodes, stgcn_cfg.DATA_WINDOW1, stgcn_cfg.DATA_WINDOW2, stgcn_cfg.TIME_STEP_MINUTES
    )
    
    n_total = len(df_all)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    
    df_train = df_all.iloc[:n_train]
    df_test = df_all.iloc[n_train+n_val:]
    
    models_registry = {
        'STGCN_Baseline': {
            'config': stgcn_cfg,
            'build_fn': lambda cfg: Baseline_STGCN_Model(
                num_nodes=len(nodes), in_feat=5, block_hidden=cfg.BLOCK_HIDDEN,
                num_blocks=cfg.NUM_BLOCKS, T_in=cfg.T_IN, cheb_K=cfg.CHEB_K,
                horizon=cfg.HORIZON, output_feat=2, L_tilde=L_tilde, dropout=cfg.DROPOUT
            )
        },
        'Graph_WaveNet': {
            'config': stgcn_cfg,
            'build_fn': lambda cfg: GraphWaveNet(
                num_nodes=len(nodes), in_dim=5, out_dim=2, residual_channels=64, dilation_channels=64, blocks=4, layers=2, horizon=cfg.HORIZON
            )
        },
        'ASTGCN': {
            'config': stgcn_cfg,
            'build_fn': lambda cfg: ASTGCN(
                num_nodes=len(nodes), in_channels=5, K=cfg.CHEB_K, num_blocks=2, T_in=cfg.T_IN, horizon=cfg.HORIZON, block_channels=36, L_tilde=L_tilde, out_dim=2
            )
        },
        'STAEformer': {
            'config': stgcn_cfg,
            'build_fn': lambda cfg: STAEformerProxy(
                num_nodes=len(nodes), in_channels=5, T_in=cfg.T_IN, horizon=cfg.HORIZON, embed_size=160, heads=4, out_dim=2
            )
        },
        'MegaCRN': {
            'config': stgcn_cfg,
            'build_fn': lambda cfg: MegaCRNProxy(
                num_nodes=len(nodes), in_channels=5, T_in=cfg.T_IN, horizon=cfg.HORIZON, embed_size=200, out_dim=2
            )
        },
        'DSTAGNN': {
            'config': stgcn_cfg,
            'build_fn': lambda cfg: DSTAGNNProxy(
                num_nodes=len(nodes), in_channels=5, T_in=cfg.T_IN, horizon=cfg.HORIZON, embed_size=224, heads=4, out_dim=2
            )
        },
        'iTransformer': {
            'config': stgcn_cfg,
            'build_fn': lambda cfg: iTransformerProxy(
                num_nodes=len(nodes), in_channels=5, T_in=cfg.T_IN, horizon=cfg.HORIZON, embed_size=128, heads=4, out_dim=2
            )
        },
        'TA-STGCN': {
            'config': hybrid_cfg,
            'build_fn': lambda cfg: Hybrid_STGCN_Model(
                num_nodes=len(nodes), in_feat=5, block_hidden=cfg.BLOCK_HIDDEN,
                num_blocks=cfg.NUM_BLOCKS, T_in=cfg.T_IN, cheb_K=cfg.CHEB_K,
                horizon=cfg.HORIZON, output_feat=2, L_tilde=L_tilde, dropout=cfg.DROPOUT,
                use_temporal_attention=cfg.USE_TEMPORAL_ATTENTION,
                attn_num_heads=4, attn_dropout=cfg.ATTN_DROPOUT
            )
        }
    }
    
    results = {m: {'low': [], 'med': [], 'high': []} for m in models_registry}
    
    def get_density_metrics(model, loader, scaler_stats):
        model.eval()
        means = torch.tensor(scaler_stats['mean'], device=device)
        stds = torch.tensor(scaler_stats['std'], device=device)
        total_abs_err = [0.0, 0.0, 0.0]
        total_count = [0, 0, 0]
        
        with torch.no_grad():
            for X, Y in loader:
                X, Y = X.to(device), Y.to(device)
                pred = model(X)
                
                y_true = Y * stds + means
                y_pred = pred * stds + means
                y_true_total = y_true.sum(dim=-1)
                y_pred_total = y_pred.sum(dim=-1)
                
                abs_err_total = torch.abs(y_true_total - y_pred_total)
                
                # Stratify by MAXIMUM density in the 24-frame INPUT sequence
                x_unscaled = X[..., :2] * stds + means
                x_total = x_unscaled[..., 0] + x_unscaled[..., 1] # Total vehicles: (B, T_in, N)
                input_max_density, _ = x_total.max(dim=1) # Shape: (B, N)
                
                mask_low = (input_max_density < 10).unsqueeze(1).expand_as(abs_err_total)
                mask_med = ((input_max_density >= 10) & (input_max_density <= 25)).unsqueeze(1).expand_as(abs_err_total)
                mask_high = (input_max_density > 25).unsqueeze(1).expand_as(abs_err_total)
                
                total_abs_err[0] += abs_err_total[mask_low].sum().item()
                total_count[0] += mask_low.sum().item()
                total_abs_err[1] += abs_err_total[mask_med].sum().item()
                total_count[1] += mask_med.sum().item()
                total_abs_err[2] += abs_err_total[mask_high].sum().item()
                total_count[2] += mask_high.sum().item()
                
        mae_low = total_abs_err[0] / max(1, total_count[0])
        mae_med = total_abs_err[1] / max(1, total_count[1])
        mae_high = total_abs_err[2] / max(1, total_count[2])
        return mae_low, mae_med, mae_high

    for seed in seeds:
        set_seed(seed)
        for model_name, info in models_registry.items():
            cfg = info['config']
            train_ds = MultiStepDataset(df_train, nodes, cfg.T_IN, cfg.HORIZON)
            scaler = {'mean': train_ds.means, 'std': train_ds.stds}
            test_ds = MultiStepDataset(df_test, nodes, cfg.T_IN, cfg.HORIZON, scaler)
            test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
            
            model = info['build_fn'](cfg).to(device)
            clean_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("=", "_")
            ckpt_path = os.path.join(args.root_dir, 'checkpoints', f"best_{clean_name}_seed_{seed}.pth")
            fallback_path = os.path.join(args.root_dir, 'model', f"best_{clean_name}_seed_{seed}.pth")
            
            loaded = False
            for p in [ckpt_path, fallback_path, f"checkpoints/best_{clean_name}_seed_{seed}.pth", f"model/best_{clean_name}_seed_{seed}.pth"]:
                if os.path.exists(p):
                    model.load_state_dict(torch.load(p, map_location=device))
                    loaded = True
                    break

            if loaded:
                l, m, h = get_density_metrics(model, test_loader, scaler)
                results[model_name]['low'].append(l)
                results[model_name]['med'].append(m)
                results[model_name]['high'].append(h)
                print(f"Seed {seed} | {model_name:15} | Low: {l:.4f} | Med: {m:.4f} | High: {h:.4f}")
            else:
                print(f"   ⚠️ WARNING: Không tìm thấy checkpoint cho {model_name} (Seed {seed})")
                
            del model
            torch.cuda.empty_cache()
            
    print("\n" + "="*80)
    print("📊 TỔNG HỢP KẾT QUẢ STAGE 2 DENSITY ERROR PROPAGATION (Mean ± Std)")
    print("="*80)
    print("| Model | Low (<10) | Medium (10-25) | High (>25) |")
    print("|---|---|---|---|")
    
    md_content = "# Báo cáo Đánh giá Lan truyền Lỗi Theo Mật độ (Stage 2)\n\n"
    md_content += "| Model | Low (<10) | Medium (10-25) | High (>25) |\n"
    md_content += "|---|---|---|---|\n"
    
    for model_name in results:
        l = results[model_name]['low']
        m = results[model_name]['med']
        h = results[model_name]['high']
        if len(l) > 0:
            row_str = f"| {model_name} | {np.mean(l):.4f} ± {np.std(l):.4f} | {np.mean(m):.4f} ± {np.std(m):.4f} | {np.mean(h):.4f} ± {np.std(h):.4f} |"
            print(row_str)
            md_content += row_str + "\n"

    with open("eval_stage2_density_report.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    print("\n✅ Đã lưu kết quả Stage 2 vào eval_stage2_density_report.md")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="g:/nckh")
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 100, 2024, 22, 99], help="Danh sách seeds thử nghiệm")
    # Thêm args giả để không bị lỗi khi bash truyền vào --epochs
    parser.add_argument('--epochs', type=int, default=1)
    args = parser.parse_args()
    
    # Run Stage 1 Evaluation
    evaluate_stage1_density(args)
    
    # Run Stage 2 Evaluation
    evaluate_stage2_density(args)
