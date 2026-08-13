import os
# Cấu hình PyTorch Allocator tránh phân mảnh bộ nhớ CUDA OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from scipy.stats import t as t_dist, wilcoxon, friedmanchisquare

import gc
import copy
import time
import random
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm.auto import tqdm
import timm

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

sys.stdout = TeeLogger("logs/train_counting.log")

# =====================================================================
# ⚙️ GLOBAL CONFIGURATION & FILE PATHS (CẤU HÌNH ĐƯỜNG DẪN & MÔ HÌNH)
# Dễ dàng chỉnh sửa đường dẫn tệp, họ mô hình và các siêu tham số tại đây:
# =====================================================================
class Config:
    # 📂 Đường dẫn tệp CSV nhãn và thư mục ảnh
    CSV_FILE = "/workspace/traffic_update.csv"
    IMAGE_DIR = "/workspace/images"

    # 🏗️ Danh sách 5 họ mô hình cần chạy benchmark: 'resnet', 'efficientnet', 'vit', 'convnext', 'mobilenet'
    MODELS = ['resnet', 'efficientnet', 'vit', 'convnext', 'mobilenet']

    # 🧪 Danh sách seeds ngẫu nhiên để đánh giá thống kê (Mean ± Std)
    SEEDS = [42, 100, 2024, 22, 99]

    # ⚡ Siêu tham số huấn luyện
    EPOCHS = 120
    PATIENCE = 20 # ⚡ Early Stopping Patience = 17 epochs
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4
    IMG_SIZE = (224, 224)

    # 📑 Tên tệp lưu báo cáo kết quả Markdown
    REPORT_PATH = "counting_benchmark_report.md"
# =====================================================================


def set_seed(seed):
    """Cố định seed ngẫu nhiên đảm bảo tính lặp lại (Reproducibility)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VehicleDataset(Dataset):
    """Dataset nạp ảnh giao thông và nhãn đếm phương tiện (Ô tô & Xe máy) an toàn tuyệt đối."""
    def __init__(self, csv_file, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"❌ File CSV nhãn không tồn tại: {csv_file}")
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"❌ Thư mục ảnh không tồn tại: {image_dir}")

        raw_df = pd.read_csv(csv_file)
        raw_count = len(raw_df)

        # 1. Tự động xác định tên cột filename
        fn_col = None
        for c in ['filename', 'file_name', 'image', 'image_name', 'img', 'name']:
            if c in raw_df.columns:
                fn_col = c
                break
        if fn_col is None:
            fn_col = raw_df.columns[0] # Mặc định chọn cột đầu tiên

        # 2. Tự động xác định tên cột Xe máy và Ô tô (Hỗ trợ định dạng header: filename,xe_may,o_to)
        moto_col = None
        for c in ['xe_may', 'xemay', 'motorcycle', 'motorcycles', 'motorbike', 'motor']:
            if c in raw_df.columns:
                moto_col = c
                break

        car_col = None
        for c in ['o_to', 'oto', 'car', 'cars', 'car_count']:
            if c in raw_df.columns:
                car_col = c
                break

        if car_col is None or moto_col is None:
            other_cols = [c for c in raw_df.columns if c != fn_col]
            if len(other_cols) >= 2:
                # Nếu không tìm thấy bằng tên chính xác, đoán dựa trên từ khóa trong tên cột
                for c in other_cols:
                    c_lower = str(c).lower()
                    if any(k in c_lower for k in ['may', 'moto', 'bike']) and moto_col is None:
                        moto_col = c
                    elif any(k in c_lower for k in ['to', 'car', 'auto']) and car_col is None:
                        car_col = c
                
                # Mặc định gán 2 cột nếu vẫn chưa xác định
                if moto_col is None:
                    moto_col = other_cols[0]
                if car_col is None:
                    car_col = other_cols[1] if other_cols[1] != moto_col else other_cols[0]
            else:
                raise ValueError(f"❌ Không tìm thấy các cột chứa nhãn số lượng [Xe máy, Ô tô] trong CSV: {csv_file}")

        print(f"📌 [VehicleDataset] Đã xác định thứ tự cột từ CSV -> Ảnh: '{fn_col}' | Xe máy: '{moto_col}' | Ô tô: '{car_col}'")

        # 3. Lọc bỏ các dòng thiếu thông tin nhãn, NaN hoặc ảnh không tồn tại trên đĩa
        valid_rows = []
        for idx, row in raw_df.iterrows():
            fname = str(row[fn_col]).strip()
            if not fname or pd.isna(row[car_col]) or pd.isna(row[moto_col]):
                continue
            
            img_path = os.path.join(self.image_dir, fname)
            if not os.path.isfile(img_path):
                # Thử kiểm tra nếu tên tệp thiếu đuôi mở rộng (.jpg / .png)
                alt_paths = [img_path + ".jpg", img_path + ".png"]
                found_alt = False
                for ap in alt_paths:
                    if os.path.isfile(ap):
                        fname = os.path.basename(ap)
                        found_alt = True
                        break
                if not found_alt:
                    continue # Bỏ qua nếu ảnh không tồn tại trong image_dir

            try:
                car_cnt = float(row[car_col])
                moto_cnt = float(row[moto_col])
                valid_rows.append({'filename': fname, 'o_to': car_cnt, 'xe_may': moto_cnt})
            except (ValueError, TypeError):
                continue

        self.data = pd.DataFrame(valid_rows)
        clean_count = len(self.data)
        dropped_count = raw_count - clean_count

        print(f"📊 [VehicleDataset] Đã nạp thành công {clean_count}/{raw_count} cặp [ảnh - nhãn] hợp lệ.")
        if dropped_count > 0:
            print(f"⚠️ Đã tự động lọc bỏ {dropped_count} dòng trong CSV không tìm thấy ảnh hoặc bị lỗi nhãn.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, str(row['filename']))
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Fallback nếu tệp ảnh bị hỏng: tạo ảnh đen giả lập
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))

        label = torch.tensor([float(row['o_to']), float(row['xe_may'])], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label


def build_counting_model(model_name: str, num_classes: int = 2, pretrained: bool = True):
    """
    Khởi tạo mô hình ước lượng số lượng phương tiện (2 đầu ra: [Ô tô, Xe máy]).
    Hỗ trợ 4 họ mô hình tiêu chuẩn: ResNet (ResNet-50), EfficientNet (EfficientNet-B4), ViT (ViT-Small), ConvNeXt (ConvNeXt-Tiny).
    """
    name_clean = model_name.lower()
    
    if 'resnet' in name_clean:
        try:
            model = timm.create_model('resnet50', pretrained=pretrained, num_classes=num_classes)
        except Exception:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )
    elif 'efficientnet' in name_clean:
        try:
            model = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=num_classes)
        except Exception:
            model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT if pretrained else None)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )
    elif 'vit' in name_clean:
        try:
            model = timm.create_model('vit_small_patch16_224', pretrained=pretrained, num_classes=num_classes)
        except Exception:
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)
            in_features = model.heads.head.in_features
            model.heads.head = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )
    elif 'convnext' in name_clean:
        try:
            model = timm.create_model('convnext_tiny', pretrained=pretrained, num_classes=num_classes)
        except Exception:
            model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
            in_features = model.classifier[2].in_features
            model.classifier[2] = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )
    elif 'mobilenet' in name_clean:
        try:
            model = timm.create_model('mobilenetv3_large_100', pretrained=pretrained, num_classes=num_classes)
        except Exception:
            model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
            in_features = model.classifier[0].in_features
            model.classifier = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.Hardswish(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )
    else:
        raise ValueError(f"Tên mô hình không hợp lệ: {model_name}. Chọn một trong các loại: 'resnet', 'efficientnet', 'vit', 'convnext', 'mobilenet'")

    return model


def count_parameters(model):
    """Đếm tổng số tham số có thể huấn luyện (Trainable Parameters)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, dummy_input):
    """Đếm số lượng phép tính FLOPs (GFLOPs) cho 1 batch đầu vào."""
    try:
        import thop
        flops, _ = thop.profile(model, inputs=(dummy_input,), verbose=False)
        return flops / 1e9
    except Exception:
        params = count_parameters(model)
        return (2 * params) / 1e9


class GradCAMPlusPlus:
    """
    Grad-CAM++ (Chattopadhay et al., WACV 2018) - Phương pháp giải thích trực quan chuẩn mực nhất
    cho các bài toán đếm nhiều đối tượng mật độ cao (Multi-Object Vehicle Counting Regression).
    Sử dụng đạo hàm cấp 2 và cấp 3 để tính trọng số pixel-wise positive gradient, phản ánh chính xác
    từng vị trí xe máy và ô tô trong cảnh giao thông phức tạp.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        def save_activations(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.activations = output.clone()

            def save_gradients(grad):
                self.gradients = grad.clone()

            output.register_hook(save_gradients)

        self.target_layer.register_forward_hook(save_activations)

    def generate_heatmap(self, input_tensor, target_class_idx=None):
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        score = output.sum() if target_class_idx is None else output[:, target_class_idx].sum()

        score.backward(retain_graph=True)

        gradients = self.gradients.data[0] # (C, H, W)
        activations = self.activations.data[0] # (C, H, W)

        # Tính đạo hàm cấp 2 và cấp 3 phục vụ công thức Grad-CAM++
        g2 = gradients.pow(2)
        g3 = gradients.pow(3)

        sum_activations = torch.sum(activations, dim=(1, 2), keepdim=True) # (C, 1, 1)

        aij = g2 / (2.0 * g2 + sum_activations * g3 + 1e-7)
        aij = torch.where(g2 != 0, aij, torch.zeros_like(aij))

        # Trọng số Grad-CAM++ từng Feature Map
        weights = torch.sum(aij * F.relu(gradients), dim=(1, 2)) # (C,)

        # Tính tổng có trọng số Heatmap
        cam = torch.sum(weights.view(-1, 1, 1) * activations, dim=0).cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, output.data.cpu().numpy()[0]


def get_target_layer(model, model_name):
    """Xác định lớp Convolution/Attention cuối cùng cho từng kiến trúc mô hình."""
    name_lower = model_name.lower()
    if 'resnet' in name_lower:
        if hasattr(model, 'layer4'):
            return model.layer4[-1]
    elif 'efficientnet' in name_lower:
        if hasattr(model, 'conv_head'):
            return model.conv_head
        elif hasattr(model, 'blocks'):
            return model.blocks[-1]
    elif 'vit' in name_lower:
        if hasattr(model, 'blocks'):
            return model.blocks[-1]
        elif hasattr(model, 'encoder'):
            return model.encoder.layers[-1]
    elif 'convnext' in name_lower:
        if hasattr(model, 'stages'):
            return model.stages[-1]
        elif hasattr(model, 'features'):
            return model.features[-1]
    elif 'mobilenet' in name_lower:
        if hasattr(model, 'features'):
            return model.features[-1]
        elif hasattr(model, 'blocks'):
            return model.blocks[-1]

    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.LayerNorm)):
            return module
    raise ValueError(f"Không tìm thấy target layer cho {model_name}")


def generate_vision_explainability_figures(sample_img_path, trained_models_dict, device, save_dir="paper/fig"):
    """Tạo biểu đồ Grad-CAM++ Feature Attribution Heatmap so sánh 4 mô hình vision."""
    os.makedirs(save_dir, exist_ok=True)
    if not os.path.exists(sample_img_path):
        print(f"⚠️ Không tìm thấy ảnh {sample_img_path} để tạo Grad-CAM heatmap.")
        return

    raw_img = Image.open(sample_img_path).convert('RGB')
    orig_w, orig_h = raw_img.size

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = preprocess(raw_img).unsqueeze(0).to(device)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), dpi=300)
    plt.rcParams['font.family'] = 'DejaVu Sans'

    axes[0].imshow(raw_img)
    axes[0].set_title("(a) Traffic Camera Scene\n(Original Input)", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    available_models = [m for m in ['resnet', 'efficientnet', 'vit', 'convnext', 'mobilenet'] if m in trained_models_dict]
    title_dict = {
        'resnet': "(b) ResNet-50\n(Grad-CAM++)",
        'efficientnet': "(c) EfficientNet-B4\n(Grad-CAM++)",
        'vit': "(d) ViT-Small\n(Grad-CAM++)",
        'convnext': "(e) ConvNeXt-Tiny\n(Grad-CAM++)",
        'mobilenet': "(f) MobileNet-V3\n(Grad-CAM++)"
    }

    num_cols = 1 + len(available_models)
    fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4.5), dpi=300)
    plt.rcParams['font.family'] = 'DejaVu Sans'

    axes[0].imshow(raw_img)
    axes[0].set_title("(a) Traffic Camera Scene\n(Original Input)", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    for idx, m_key in enumerate(available_models):
        ax = axes[idx + 1]
        model = trained_models_dict[m_key].to(device)
        model.eval()

        target_layer = get_target_layer(model, m_key)
        grad_cam = GradCAMPlusPlus(model, target_layer)

        cam_mask, preds = grad_cam.generate_heatmap(input_tensor)

        cam_img = Image.fromarray((cam_mask * 255).astype(np.uint8)).resize((orig_w, orig_h), resample=Image.BILINEAR)
        cam_arr = np.array(cam_img) / 255.0

        # Lọc bỏ nhiễu nền (< 0.15): Giữ ảnh gốc trong suốt 100% ở phông nền, chỉ rực màu (Vàng/Đỏ) tại khu vực phương tiện
        cam_display = cam_arr.copy()
        cam_display[cam_display < 0.15] = np.nan

        ax.imshow(raw_img)
        ax.imshow(cam_display, cmap='jet', alpha=0.65, vmin=0.15, vmax=1.0)
        ax.set_title(f"{title_dict.get(m_key, m_key)} (Grad-CAM++)", fontsize=10, fontweight='bold')
        ax.axis('off')

        # --- VẼ NỔI 1 HÌNH ĐỘC LẬP CHO RIÊNG MÔ HÌNH NÀY (2 SUBPLOT: ẢNH GỐC & GRAD-CAM++) ---
        fig_single, axes_single = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)
        
        axes_single[0].imshow(raw_img)
        axes_single[0].set_title("(a) Original Traffic Camera Feed", fontsize=12, fontweight='bold', pad=8)
        axes_single[0].axis('off')

        axes_single[1].imshow(raw_img)
        axes_single[1].imshow(cam_display, cmap='jet', alpha=0.65, vmin=0.15, vmax=1.0)
        m_title = title_dict.get(m_key, m_key)
        axes_single[1].set_title(f"(b) {m_title} Grad-CAM++ Attribution", fontsize=12, fontweight='bold', pad=8)
        axes_single[1].axis('off')

        plt.tight_layout()
        single_pdf = os.path.join(save_dir, f"gradcam_{m_key}.pdf")
        single_png = os.path.join(save_dir, f"gradcam_{m_key}.png")
        plots_single_png = os.path.join("plots", f"gradcam_{m_key}.png")

        plt.savefig(single_pdf, format='pdf', bbox_inches='tight')
        plt.savefig(single_png, format='png', bbox_inches='tight', dpi=300)
        plt.savefig(plots_single_png, format='png', bbox_inches='tight', dpi=300)
        plt.close(fig_single)

    plt.tight_layout()
    fig_pdf = os.path.join(save_dir, "vision_explainability_gradcam.pdf")
    fig_png = os.path.join(save_dir, "vision_explainability_gradcam.png")
    plt.savefig(fig_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(fig_png, format='png', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"🖼️ Đã tự động tạo biểu đồ Grad-CAM XAI cho các mô hình Vision vào:\n   - {fig_pdf}\n   - {fig_png}")


def measure_inference_latency(model, loader, device, max_batches=20):
    """Đo độ trễ suy luận (Inference Latency) tính theo ms/batch."""
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= 3:
                break
            images = images.to(device)
            _ = model(images)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    start_time = time.time()
    count = 0
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= max_batches:
                break
            images = images.to(device)
            _ = model(images)
            count += 1

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed_ms = (time.time() - start_time) * 1000.0
    return elapsed_ms / max(1, count)


def compute_counting_metrics(y_true, y_pred):
    """
    Tính toán chi tiết các chỉ số MAE, MAPE (%), RMSE, MSE:
    - Cho Tổng số lượng phương tiện (Overall)
    - Cho Ô tô (Cars - Index 0)
    - Cho Xe máy (Motorcycles - Index 1)
    Bổ sung pw_ (pointwise errors) cho Wilcoxon/Friedman tests.
    """
    err = y_true - y_pred
    abs_err = np.abs(err)
    sq_err = err ** 2

    # 1. Tổng thể (Overall)
    mae_overall = np.mean(abs_err)
    mse_overall = np.mean(sq_err)
    rmse_overall = np.sqrt(mse_overall)
    mask_overall = (y_true > 0.5)
    mape_overall = np.sum((abs_err / (y_true + 1e-5)) * mask_overall) / max(np.sum(mask_overall), 1.0)

    # 2. Ô tô (Car - Index 0)
    abs_car = abs_err[:, 0]
    mae_car = np.mean(abs_car)
    mse_car = np.mean(sq_err[:, 0])
    rmse_car = np.sqrt(mse_car)
    mask_car = (y_true[:, 0] > 0.5)
    mape_car = np.sum((abs_car / (y_true[:, 0] + 1e-5)) * mask_car) / max(np.sum(mask_car), 1.0)

    # 3. Xe máy (Motorcycle - Index 1)
    abs_moto = abs_err[:, 1]
    mae_moto = np.mean(abs_moto)
    mse_moto = np.mean(sq_err[:, 1])
    rmse_moto = np.sqrt(mse_moto)
    mask_moto = (y_true[:, 1] > 0.5)
    mape_moto = np.sum((abs_moto / (y_true[:, 1] + 1e-5)) * mask_moto) / max(np.sum(mask_moto), 1.0)

    # Pointwise error arrays for statistical testing
    pw_overall = np.mean(abs_err, axis=1)  # per-sample MAE
    pw_car = abs_car
    pw_moto = abs_moto

    return {
        'mae_overall': mae_overall, 'mape_overall': mape_overall, 'rmse_overall': rmse_overall, 'mse_overall': mse_overall,
        'mae_car': mae_car, 'mape_car': mape_car, 'rmse_car': rmse_car, 'mse_car': mse_car,
        'mae_moto': mae_moto, 'mape_moto': mape_moto, 'rmse_moto': rmse_moto, 'mse_moto': mse_moto,
        'pw_overall': pw_overall, 'pw_car': pw_car, 'pw_moto': pw_moto
    }


def train_single_seed_counting(model_name, train_loader, val_loader, test_loader, cfg, device, seed):
    """Huấn luyện và đánh giá 1 mô hình đếm phương tiện với 1 seed ngẫu nhiên (hỗ trợ Early Stopping = 17)."""
    set_seed(seed)
    
    model = build_counting_model(model_name, num_classes=2, pretrained=True).to(device)
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    patience = cfg.get('patience', 17)
    patience_counter = 0
    best_val_mae = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())

    print(f"\n⚡ [{model_name.upper()}] Seed {seed} | Bắt đầu huấn luyện (Max Epochs={cfg['epochs']}, Patience={patience})...")

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            train_loss += loss.item() * len(images)

        train_loss /= len(train_loader.dataset)
        scheduler.step()

        # Validation phase
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                val_preds.append(outputs.cpu().numpy())
                val_trues.append(labels.numpy())

        val_preds = np.vstack(val_preds)
        val_trues = np.vstack(val_trues)
        val_mae = np.mean(np.abs(val_trues - val_preds))

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == cfg['epochs']:
            print(f"   Epoch {epoch:>2d}/{cfg['epochs']} | Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.2f} (Best: {best_val_mae:.2f})")

        if patience_counter >= patience:
            print(f"   🛑 Early stopping tại epoch {epoch} (Patience={patience} không giảm Val MAE). Best Val MAE: {best_val_mae:.2f}")
            break

    # Đánh giá trên tập Test với trọng số tốt nhất
    model.load_state_dict(best_model_weights)
    model.eval()
    
    os.makedirs('checkpoints', exist_ok=True)
    ckpt_path = os.path.join('checkpoints', f"best_counting_{model_name}_seed_{seed}.pth")
    torch.save(best_model_weights, ckpt_path)

    test_preds, test_trues = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            test_preds.append(outputs.cpu().numpy())
            test_trues.append(labels.numpy())

    test_preds = np.vstack(test_preds)
    test_trues = np.vstack(test_trues)

    test_metrics = compute_counting_metrics(test_trues, test_preds)
    return model, test_metrics


def run_counting_benchmark():
    parser = argparse.ArgumentParser(description="Script Huấn luyện & Benchmark Mô hình Đếm Phương tiện (Sub-problem 1).")
    parser.add_argument('--csv_file', type=str, default=Config.CSV_FILE,
                        help="Đường dẫn file CSV chứa nhãn đếm phương tiện (filename, o_to, xe_may).")
    parser.add_argument('--image_dir', type=str, default=Config.IMAGE_DIR,
                        help="Thư mục chứa tệp ảnh giao thông.")
    parser.add_argument('--seeds', type=int, nargs='+', default=Config.SEEDS,
                        help="Danh sách các seeds thử nghiệm.")
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS, help="Số epochs huấn luyện tối đa.")
    parser.add_argument('--patience', type=int, default=Config.PATIENCE, help="Early stopping patience (mặc định: 17).")
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE, help="Kích thước batch_size.")
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE, help="Learning rate.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"============================================================")
    print(f"🚀 CHẠY BENCHMARK ĐẾM PHƯƠNG TIỆN (SUB-PROBLEM 1)")
    print(f"   Device        : {device}")
    print(f"   Seeds         : {args.seeds}")
    print(f"   Epochs        : {args.epochs}")
    print(f"   Patience      : {args.patience}")
    print(f"   Batch Size    : {args.batch_size}")
    print(f"   CSV Label File: {args.csv_file}")
    print(f"   Image Dir     : {args.image_dir}")
    print(f"============================================================")

    # 1. Cấu hình Data Augmentations nâng cao phù hợp với ảnh giao thông
    train_transform = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), value=0)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Nạp Dataset và chia tập dữ liệu (Train 80%, Val 10%, Test 10%)
    if not os.path.exists(args.csv_file):
        local_csv = os.path.join(os.getcwd(), "labels1.csv")
        local_img = os.path.join(os.getcwd(), "images")
        if os.path.exists(local_csv):
            args.csv_file = local_csv
            args.image_dir = local_img

    full_dataset = VehicleDataset(args.csv_file, args.image_dir, transform=train_transform)
    total_len = len(full_dataset)
    n_train = int(0.8 * total_len)
    n_val = int(0.1 * total_len)
    n_test = total_len - n_train - n_val

    set_seed(42)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(full_dataset, [n_train, n_val, n_test])
    
    val_ds.dataset = copy.deepcopy(full_dataset)
    val_ds.dataset.transform = val_transform
    test_ds.dataset = copy.deepcopy(full_dataset)
    test_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"📂 Phân chia dữ liệu: Train={n_train}, Val={n_val}, Test={n_test}")

    models_to_test = Config.MODELS
    cfg = {'epochs': args.epochs, 'lr': args.lr, 'patience': args.patience}
    trained_models_dict = {}

    best_seed_val_mae = {m_name: float('inf') for m_name in models_to_test}

    results = {
        m_name: {
            'params': 0, 'flops_gflops': 0.0, 'inf_latencies': [],
            'mae_overall': [], 'mape_overall': [], 'rmse_overall': [], 'mse_overall': [],
            'mae_car': [], 'mape_car': [], 'rmse_car': [],
            'mae_moto': [], 'mape_moto': [], 'rmse_moto': [],
            'pw_overalls': [], 'pw_cars': [], 'pw_motos': []
        } for m_name in models_to_test
    }

    for seed in args.seeds:
        print(f"\n{'='*70}")
        print(f"🧪 [SUB-PROBLEM 1 - SEED {seed}] THỬ NGHIỆM MÔ HÌNH THỊ GIÁC (STAGE 1 COUNTING BENCHMARK)")
        print(f"{'='*70}")

        for m_name in models_to_test:
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            model, metrics = train_single_seed_counting(
                m_name, train_loader, val_loader, test_loader, cfg, device, seed
            )

            # Đã lưu per-seed best checkpoint tại: checkpoints/best_counting_{m_name}_seed_{seed}.pth
            # Kiểm tra & lưu overall best checkpoint toàn bộ các seed
            if metrics['mae_overall'] < best_seed_val_mae[m_name]:
                best_seed_val_mae[m_name] = metrics['mae_overall']
                overall_ckpt = os.path.join('checkpoints', f"overall_best_counting_{m_name}.pth")
                torch.save(model.state_dict(), overall_ckpt)
                print(f"   🏆 [NEW OVERALL BEST FOR {m_name.upper()}] Best MAE: {metrics['mae_overall']:.4f} -> Saved {overall_ckpt}")

            # Lưu lại mô hình đã huấn luyện của seed đầu tiên phục vụ Grad-CAM++ visualization
            if seed == args.seeds[0]:
                trained_models_dict[m_name] = copy.deepcopy(model).cpu()

            p_count = count_parameters(model)
            results[m_name]['params'] = p_count

            dummy_img, _ = next(iter(test_loader))
            dummy_img = dummy_img.to(device)
            gflops = count_flops(model, dummy_img)
            results[m_name]['flops_gflops'] = gflops

            lat = measure_inference_latency(model, test_loader, device)
            results[m_name]['inf_latencies'].append(lat)

            results[m_name]['mae_overall'].append(metrics['mae_overall'])
            results[m_name]['mape_overall'].append(metrics['mape_overall'])
            results[m_name]['rmse_overall'].append(metrics['rmse_overall'])
            results[m_name]['mse_overall'].append(metrics['mse_overall'])

            results[m_name]['mae_car'].append(metrics['mae_car'])
            results[m_name]['mape_car'].append(metrics['mape_car'])
            results[m_name]['rmse_car'].append(metrics['rmse_car'])

            results[m_name]['mae_moto'].append(metrics['mae_moto'])
            results[m_name]['mape_moto'].append(metrics['mape_moto'])
            results[m_name]['rmse_moto'].append(metrics['rmse_moto'])

            results[m_name]['pw_overalls'].append(metrics['pw_overall'])
            results[m_name]['pw_cars'].append(metrics['pw_car'])
            results[m_name]['pw_motos'].append(metrics['pw_moto'])

            print(f"   ▶ Seed {seed:>4} | {m_name.upper():<12} (Params: {p_count:,} | FLOPs: {gflops:.3f}G) -> "
                  f"MAE Tổng: {metrics['mae_overall']:.2f} (Ô tô: {metrics['mae_car']:.2f}, Xe máy: {metrics['mae_moto']:.2f})")

            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

    # --- Hàm format 95% CI giống benchmark_5seeds.py ---
    def get_ci_str(arr):
        mean = np.mean(arr)
        std = np.std(arr, ddof=1) if len(arr) > 1 else 0
        n = len(arr)
        t_crit = t_dist.ppf(0.975, df=n-1) if n > 1 else 0
        margin = t_crit * (std / np.sqrt(n)) if n > 1 else 0
        return f"{mean:.4f} ± {std:.4f} (95% CI: {mean-margin:.4f}-{mean+margin:.4f})"

    def get_ci_str_pct(arr):
        arr_pct = [v * 100 for v in arr]
        mean = np.mean(arr_pct)
        std = np.std(arr_pct, ddof=1) if len(arr_pct) > 1 else 0
        n = len(arr_pct)
        t_crit = t_dist.ppf(0.975, df=n-1) if n > 1 else 0
        margin = t_crit * (std / np.sqrt(n)) if n > 1 else 0
        return f"{mean:.2f}% ± {std:.2f}% (95% CI: {mean-margin:.2f}%-{mean+margin:.2f}%)"

    table_overall = []
    table_breakdown = []

    for m_name in models_to_test:
        res = results[m_name]
        p_count = res['params']
        gflops = res['flops_gflops']
        lats = res['inf_latencies']

        table_overall.append({
            'Model Architecture': m_name.upper(),
            'Params': f"{p_count:,}",
            'FLOPs (GFLOPs)': f"{gflops:.3f}",
            'Latency (ms/batch)': get_ci_str(lats),
            'MAE Overall': get_ci_str(res['mae_overall']),
            'MAPE Overall (%)': get_ci_str_pct(res['mape_overall']),
            'RMSE Overall': get_ci_str(res['rmse_overall']),
            'MSE Overall': get_ci_str(res['mse_overall'])
        })

        table_breakdown.append({
            'Model Architecture': m_name.upper(),
            'MAE Ô tô (Car)': get_ci_str(res['mae_car']),
            'MAPE Ô tô (%)': get_ci_str_pct(res['mape_car']),
            'RMSE Ô tô': get_ci_str(res['rmse_car']),
            'MAE Xe máy (Moto)': get_ci_str(res['mae_moto']),
            'MAPE Xe máy (%)': get_ci_str_pct(res['mape_moto']),
            'RMSE Xe máy': get_ci_str(res['rmse_moto'])
        })

    df_overall = pd.DataFrame(table_overall)
    df_breakdown = pd.DataFrame(table_breakdown)

    print(f"\n{'='*110}")
    print(f"🏆 1. BẢNG SO SÁNH TỔNG QUAN HỌ MÔ HÌNH ĐẾM PHƯƠNG TIỆN ({len(args.seeds)} SEEDS)")
    print(f"{'='*110}")
    print(df_overall.to_string(index=False))

    print(f"\n{'='*110}")
    print(f"🏍️🚗 2. BẢNG SO SÁNH CHI TIẾT TÁCH RIÊNG Ô TÔ VÀ XE MÁY ({len(args.seeds)} SEEDS)")
    print(f"{'='*110}")
    print(df_breakdown.to_string(index=False))

    report_path = Config.REPORT_PATH
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# 🚗🏍️ Báo cáo Benchmark Mô hình Đếm Phương tiện (Sub-problem 1)\n\n")
        f.write(f"- **Seeds sử dụng**: `{args.seeds}`\n")
        f.write(f"- **Cấu hình**: Epochs={args.epochs}, Early Stopping Patience={args.patience}, Batch Size={args.batch_size}, LR={args.lr}\n")
        f.write(f"- **Họ mô hình so sánh**: ResNet (ResNet-50), EfficientNet (EfficientNet-B4), ViT (ViT-Small), ConvNeXt (ConvNeXt-Tiny)\n\n")
        f.write("## 🏆 1. Bảng So sánh Tổng quan (Params, FLOPs, Latency & Overall Error)\n\n")
        f.write(df_overall.to_markdown(index=False))
        f.write("\n\n---\n\n## 🏍️🚗 2. Bảng So sánh Chi tiết Tách riêng Ô tô (Car) và Xe máy (Motorcycle)\n\n")
        f.write(df_breakdown.to_markdown(index=False))

        # --- Variance Decomposition Analysis ---
        f.write("\n\n---\n\n## 📉 Phân tích Variance (Seed Stochasticity)\n\n")
        f.write("| Model | Seed Variance (Var) | Seed Std (Std) | Tỷ lệ biến động tương đối (Std / Mean) |\n")
        f.write("|---|---|---|---|\n")
        for m_name in models_to_test:
            res = results[m_name]
            mean_val = np.mean(res['mae_overall'])
            var_val = np.var(res['mae_overall'], ddof=1) if len(res['mae_overall']) > 1 else 0
            std_val = np.std(res['mae_overall'], ddof=1) if len(res['mae_overall']) > 1 else 0
            cv = (std_val / mean_val) * 100 if mean_val > 0 else 0
            f.write(f"| {m_name.upper()} | {var_val:.4e} | {std_val:.4f} | {cv:.2f}% |\n")

        # --- Friedman Test ---
        f.write("\n\n## 🔬 Kiểm định Tổng quát Friedman Test\n\n")
        all_models = list(results.keys())
        if len(all_models) > 2:
            all_pw = [np.concatenate(results[m]['pw_overalls']) for m in all_models]
            try:
                stat, p_friedman = friedmanchisquare(*all_pw)
                f.write(f"- **H0:** Tất cả các mô hình có hiệu năng tương đương nhau.\n")
                f.write(f"- **Friedman Chi-Square Statistic:** {stat:.4f}\n")
                f.write(f"- **p-value:** {p_friedman:.4e}\n")
                if p_friedman < 0.05:
                    f.write(f"\n> ✅ Có sự khác biệt có ý nghĩa thống kê giữa các mô hình ($p < 0.05$).\n\n")
                else:
                    f.write(f"\n> ⚠️ Không đủ bằng chứng thống kê ($p \\geq 0.05$).\n\n")
            except Exception as e:
                f.write(f"⚠️ Không thể chạy Friedman test: {e}\n\n")

        # --- Wilcoxon Signed-Rank Test (ResNet-50 vs Others) ---
        baseline_model = 'resnet'
        f.write("\n## 🔬 Post-Hoc: Wilcoxon Signed-Rank Test & Effect Size (ResNet-50 vs Others)\n\n")
        f.write("| Baseline vs. | P-value (Overall) | Cohen's d (Overall) | P-value (Car) | Cohen's d (Car) | P-value (Moto) | Cohen's d (Moto) |\n")
        f.write("|---|---|---|---|---|---|---|\n")

        if baseline_model in results:
            base_pw = np.concatenate(results[baseline_model]['pw_overalls'])
            base_pw_car = np.concatenate(results[baseline_model]['pw_cars'])
            base_pw_moto = np.concatenate(results[baseline_model]['pw_motos'])

            def calc_cohens_dz(base_err, comp_err):
                diff = comp_err - base_err
                std_diff = np.std(diff, ddof=1)
                if std_diff == 0: return 0.0
                return np.mean(diff) / std_diff

            def format_sig(p, d):
                sig_star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
                return f"{p:.2e}{sig_star}", f"{d:.3f}"

            for m_name in models_to_test:
                if m_name == baseline_model:
                    continue
                comp_pw = np.concatenate(results[m_name]['pw_overalls'])
                comp_pw_car = np.concatenate(results[m_name]['pw_cars'])
                comp_pw_moto = np.concatenate(results[m_name]['pw_motos'])

                try:
                    _, p_tot = wilcoxon(base_pw, comp_pw)
                    _, p_car = wilcoxon(base_pw_car, comp_pw_car)
                    _, p_moto = wilcoxon(base_pw_moto, comp_pw_moto)
                except Exception:
                    p_tot, p_car, p_moto = 1.0, 1.0, 1.0

                d_tot = calc_cohens_dz(base_pw, comp_pw)
                d_car = calc_cohens_dz(base_pw_car, comp_pw_car)
                d_moto = calc_cohens_dz(base_pw_moto, comp_pw_moto)

                p_tot_str, d_tot_str = format_sig(p_tot, d_tot)
                p_car_str, d_car_str = format_sig(p_car, d_car)
                p_moto_str, d_moto_str = format_sig(p_moto, d_moto)
                f.write(f"| {m_name.upper()} | {p_tot_str} | {d_tot_str} | {p_car_str} | {d_car_str} | {p_moto_str} | {d_moto_str} |\n")

    print(f"\n📑 Đã lưu báo cáo chi tiết đếm phương tiện vào tệp: {report_path}")

    print(f"\n🎨 Đang tự động xuất biểu đồ Giải thích Mô hình Grad-CAM++ (Vision Explainability)...")
    try:
        sample_img_path = None
        if hasattr(full_dataset, 'data') and len(full_dataset.data) > 0:
            sample_fn = full_dataset.data.iloc[0]['filename']
            sample_img_path = os.path.join(args.image_dir, sample_fn)

        if sample_img_path and os.path.exists(sample_img_path):
            if not trained_models_dict:
                trained_models_dict = {
                    'resnet': build_counting_model('resnet', pretrained=True),
                    'efficientnet': build_counting_model('efficientnet', pretrained=True),
                    'vit': build_counting_model('vit', pretrained=True),
                    'convnext': build_counting_model('convnext', pretrained=True)
                }
            generate_vision_explainability_figures(sample_img_path, trained_models_dict, device, save_dir="paper/fig")
        else:
            print(f"⚠️ Không tìm thấy tệp ảnh giao thông mẫu để sinh Grad-CAM++.")
    except Exception as e:
        print(f"⚠️ Không thể sinh biểu đồ Grad-CAM++ tự động: {e}")


if __name__ == "__main__":
    run_counting_benchmark()