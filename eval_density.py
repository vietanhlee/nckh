import os
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torchvision import transforms
from train_counting import VehicleDataset, build_counting_model, Config

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def format_mean_std(data_list):
    if len(data_list) == 0:
        return "-"
    mean_val = np.mean(data_list)
    std_val = np.std(data_list)
    return f"{mean_val:.2f} ± {std_val:.2f}"
    
def format_mean_std_bias(data_list):
    if len(data_list) == 0:
        return "-"
    mean_val = np.mean(data_list)
    std_val = np.std(data_list)
    return f"{mean_val:+.2f} ± {std_val:.2f}"

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Bắt đầu đánh giá phân tầng mật độ cho TẤT CẢ CÁC MÔ HÌNH trên {device}...\n")

    csv_file = Config.CSV_FILE
    image_dir = Config.IMAGE_DIR
    
    models = Config.MODELS 
    seeds = [42, 100, 2024, 22, 99]

    val_transform = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = VehicleDataset(csv_file, image_dir, transform=val_transform)
    total_len = len(full_dataset)
    n_train = int(0.8 * total_len)
    n_val = int(0.1 * total_len)
    n_test = total_len - n_train - n_val

    md_content = "# Báo cáo Đánh giá Phân tầng Mật độ (Density-stratified Evaluation)\n\n"
    md_content += "Báo cáo này trình bày kết quả phân tích lỗi của các mô hình ở 3 mức mật độ giao thông khác nhau: Thấp (<10 xe), Trung bình (10-25 xe), và Cao (>25 xe).\n\n"

    plot_data = [] # Lưu dữ liệu để vẽ biểu đồ

    for model_name in models:
        print(f"\n=======================================================")
        print(f"📌 ĐÁNH GIÁ MÔ HÌNH: {model_name.upper()}")
        print(f"=======================================================")
        
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
                f"checkpoints/best_counting_{model_name}_seed_{seed}.pth",
                f"checkpoints/temp_counting_{model_name}_seed_{seed}.pth",
                f"model/best_counting_{model_name}_seed_{seed}.pth",
                f"checkpoints/overall_best_counting_{model_name}.pth"
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
                
                if count == 0:
                    continue
                
                preds_level = test_preds[mask]
                trues_level = test_trues[mask]

                err = preds_level - trues_level 
                results[level]['mae_car'].append(np.mean(np.abs(err[:, 0])))
                results[level]['mae_moto'].append(np.mean(np.abs(err[:, 1])))
                results[level]['bias_car'].append(np.mean(err[:, 0]))
                results[level]['bias_moto'].append(np.mean(err[:, 1]))

        # In & Lưu Bảng Markdown cho mô hình hiện tại
        md_content += f"## {model_name.upper()}\n\n"
        md_content += "| Mức mật độ | Số lượng ảnh (TB) | MAE Xe ô tô | MAE Xe máy | Độ chệch Ô tô | Độ chệch Xe máy |\n"
        md_content += "|---|---|---|---|---|---|\n"

        for level in ['Low (<10)', 'Medium (10-25)', 'High (>25)']:
            avg_count = int(np.mean(results[level]['counts'])) if results[level]['counts'] else 0
            
            mae_car_str = format_mean_std(results[level]['mae_car'])
            mae_moto_str = format_mean_std(results[level]['mae_moto'])
            bias_car_str = format_mean_std_bias(results[level]['bias_car'])
            bias_moto_str = format_mean_std_bias(results[level]['bias_moto'])
            
            md_content += f"| {level} | ~{avg_count} | {mae_car_str} | {mae_moto_str} | {bias_car_str} | {bias_moto_str} |\n"
            
            # Đẩy dữ liệu vào mảng vẽ biểu đồ (lấy giá trị trung bình)
            if results[level]['mae_car']:
                plot_data.append({'Model': model_name.upper(), 'Density': level, 'Vehicle': 'Car', 'MAE': np.mean(results[level]['mae_car'])})
            if results[level]['mae_moto']:
                plot_data.append({'Model': model_name.upper(), 'Density': level, 'Vehicle': 'Motorcycle', 'MAE': np.mean(results[level]['mae_moto'])})
                
        md_content += "\n"

    # Ghi file Markdown
    with open("density_evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    print("\n✅ Đã lưu kết quả ra file: density_evaluation_report.md")

    # Vẽ biểu đồ
    if plot_data:
        df_plot = pd.DataFrame(plot_data)
        
        # Đặt thứ tự cho trục X
        density_order = ['Low (<10)', 'Medium (10-25)', 'High (>25)']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
        plt.rcParams['font.family'] = 'DejaVu Sans'
        
        sns.barplot(data=df_plot[df_plot['Vehicle'] == 'Car'], x='Density', y='MAE', hue='Model', ax=axes[0], order=density_order)
        axes[0].set_title('Car MAE across Density Levels', fontweight='bold')
        axes[0].set_ylabel('Mean Absolute Error (MAE)')
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)
        
        sns.barplot(data=df_plot[df_plot['Vehicle'] == 'Motorcycle'], x='Density', y='MAE', hue='Model', ax=axes[1], order=density_order)
        axes[1].set_title('Motorcycle MAE across Density Levels', fontweight='bold')
        axes[1].set_ylabel('Mean Absolute Error (MAE)')
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        os.makedirs("paper/fig", exist_ok=True)
        fig_png = "paper/fig/density_mae_comparison.png"
        fig_pdf = "paper/fig/density_mae_comparison.pdf"
        plt.savefig(fig_png)
        plt.savefig(fig_pdf, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"✅ Đã vẽ và lưu biểu đồ so sánh Mật độ ra file:\n   - {fig_png}\n   - {fig_pdf}")

if __name__ == "__main__":
    main()
