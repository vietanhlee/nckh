#!/usr/bin/env bash
# ==============================================================================
# 📸 AUTOMATED GRAD-CAM++ CHECKPOINT VISUALIZER (IEEE PAPER SUITE)
# ==============================================================================
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import glob
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import timm

# Thiết lập Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ------------------------------------------------------------------------------
# 1. Định nghĩa Lớp Grad-CAM++ (Chattopadhay et al., WACV 2018)
# ------------------------------------------------------------------------------
class GradCAMPlusPlus:
    """
    Grad-CAM++ (Chattopadhay et al., WACV 2018) - Phương pháp giải thích trực quan chuẩn mực nhất
    cho các bài toán đếm nhiều đối tượng mật độ cao (Multi-Object Vehicle Counting Regression).
    Sử dụng đạo hàm cấp 2 và cấp 3 để tính trọng số pixel-wise positive gradient.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class_idx=None):
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        score = output.sum() if target_class_idx is None else output[:, target_class_idx].sum()

        score.backward(retain_graph=True)

        gradients = self.gradients.data[0] # (C, H, W)
        activations = self.activations.data[0] # (C, H, W)

        # Tính đạo hàm cấp 2 và cấp 3
        g2 = gradients.pow(2)
        g3 = gradients.pow(3)

        sum_activations = torch.sum(activations, dim=(1, 2), keepdim=True) # (C, 1, 1)

        aij = g2 / (2.0 * g2 + sum_activations * g3 + 1e-7)
        aij = torch.where(g2 != 0, aij, torch.zeros_like(aij))

        # Trọng số Grad-CAM++ từng Feature Map
        weights = torch.sum(aij * F.relu(gradients), dim=(1, 2)) # (C,)

        # Tính tổng có trọng số Heatmap
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=input_tensor.device)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = F.relu(cam)
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.detach().cpu().numpy(), output.detach().cpu().numpy()[0]


# ------------------------------------------------------------------------------
# 2. Định nghĩa Kiến trúc Mô hình & Target Layer
# ------------------------------------------------------------------------------
def build_counting_model(model_name, num_classes=2, pretrained=True):
    model_name = model_name.lower()
    if model_name in ['resnet', 'resnet50']:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name in ['efficientnet', 'efficientnet_b4']:
        model = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=num_classes)
    elif model_name in ['vit', 'vit_small']:
        model = timm.create_model('vit_small_patch16_224', pretrained=pretrained, num_classes=num_classes)
    elif model_name in ['convnext', 'convnext_tiny']:
        model = timm.create_model('convnext_tiny', pretrained=pretrained, num_classes=num_classes)
    elif model_name in ['mobilenet', 'mobilenet_v3']:
        model = timm.create_model('mobilenetv3_large_100', pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Mô hình '{model_name}' không được hỗ trợ.")
    return model


def get_target_layer(model, model_name):
    model_name = model_name.lower()
    if model_name in ['resnet', 'resnet50']:
        return model.layer4[-1]
    elif model_name in ['efficientnet', 'efficientnet_b4']:
        return getattr(model, 'conv_head', model.blocks[-1])
    elif model_name in ['vit', 'vit_small']:
        return model.blocks[-1].norm1
    elif model_name in ['convnext', 'convnext_tiny']:
        return model.stages[-1].blocks[-1]
    elif model_name in ['mobilenet', 'mobilenet_v3']:
        return getattr(model, 'conv_head', model.blocks[-1])
    else:
        raise ValueError(f"Không xác định được target layer cho mô hình '{model_name}'")


# ------------------------------------------------------------------------------
# 3. Hàm tìm kiếm Checkpoint xuất sắc nhất
# ------------------------------------------------------------------------------
def find_best_checkpoint(model_name):
    patterns = [
        f"checkpoints/overall_best_counting_{model_name}.pth",
        f"checkpoints/best_counting_{model_name}_seed_*.pth",
        f"checkpoints/*{model_name}*.pth",
        f"model/*{model_name}*.pth",
        f"*{model_name}*.pth"
    ]
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            files.sort(key=os.path.getmtime, reverse=True)
            return files[0]
    return None


# ------------------------------------------------------------------------------
# 4. Sinh Biểu đồ Grad-CAM++ Chất lượng Xuất bản IEEE
# ------------------------------------------------------------------------------
def generate_vision_explainability_figures(image_path, save_dir="paper/fig", device='cuda'):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    img_raw = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img_raw.size

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img_raw).unsqueeze(0).to(device)

    models_config = [
        ('resnet', 'ResNet-50'),
        ('efficientnet', 'EfficientNet-B4'),
        ('vit', 'ViT-Small'),
        ('convnext', 'ConvNeXt-Tiny'),
        ('mobilenet', 'MobileNet-V3')
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), dpi=300)
    axes = axes.flatten()

    # Ảnh gốc (Subplot a)
    axes[0].imshow(img_raw)
    axes[0].set_title("(a) Camera View Scene (Raw Input)", fontsize=11, fontweight='bold', pad=8)
    axes[0].axis('off')

    plt.rcParams['font.family'] = 'DejaVu Sans'

    for idx, (m_key, m_name) in enumerate(models_config, start=1):
        ax = axes[idx]
        print(f"📸 Nạp mô hình [{m_name}] và khởi tạo Grad-CAM++...")
        
        model = build_counting_model(m_key, num_classes=2, pretrained=True).to(device)
        ckpt_path = find_best_checkpoint(m_key)
        
        if ckpt_path and os.path.exists(ckpt_path):
            print(f"   - Đã nạp checkpoint trọng số: {ckpt_path}")
            try:
                state_dict = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"   ⚠️ Lỗi nạp checkpoint ({e}), sử dụng weights mặc định.")
        else:
            print(f"   ℹ️ Không tìm thấy checkpoint cho {m_name}, sử dụng trọng số pretrained ImageNet.")

        target_layer = get_target_layer(model, m_key)
        grad_cam = GradCAMPlusPlus(model, target_layer)

        try:
            cam_map, preds = grad_cam.generate_heatmap(input_tensor)
            cam_resized = Image.fromarray((cam_map * 255).astype(np.uint8)).resize((orig_w, orig_h), resample=Image.BILINEAR)
            cam_norm = np.array(cam_resized) / 255.0

            ax.imshow(img_raw)
            ax.imshow(cam_norm, cmap='jet', alpha=0.48)
            
            sub_labels = ['(b)', '(c)', '(d)', '(e)', '(f)']
            ax.set_title(f"{sub_labels[idx-1]} {m_name} (Grad-CAM++)", fontsize=11, fontweight='bold', pad=8)

            # --- VẼ NỔI 1 HÌNH ĐỘC LẬP CHO RIÊNG MÔ HÌNH NÀY (2 SUBPLOT: ẢNH GỐC & GRAD-CAM++) ---
            fig_single, axes_single = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)
            
            # (a) Ảnh gốc
            axes_single[0].imshow(img_raw)
            axes_single[0].set_title("(a) Original Traffic Camera Feed", fontsize=12, fontweight='bold', pad=8)
            axes_single[0].axis('off')

            # (b) Grad-CAM++
            axes_single[1].imshow(img_raw)
            axes_single[1].imshow(cam_norm, cmap='jet', alpha=0.48)
            axes_single[1].set_title(f"(b) {m_name} Grad-CAM++ Feature Attribution", fontsize=12, fontweight='bold', pad=8)
            axes_single[1].axis('off')

            plt.tight_layout()
            single_pdf = os.path.join(save_dir, f"gradcam_{m_key}.pdf")
            single_png = os.path.join(save_dir, f"gradcam_{m_key}.png")
            plots_single_png = os.path.join("plots", f"gradcam_{m_key}.png")

            plt.savefig(single_pdf, format='pdf', bbox_inches='tight')
            plt.savefig(single_png, format='png', bbox_inches='tight', dpi=300)
            plt.savefig(plots_single_png, format='png', bbox_inches='tight', dpi=300)
            plt.close(fig_single)
            print(f"   🖼️ Đã lưu hình riêng mô hình {m_name} vào: {single_png} & {single_pdf}")

        except Exception as e:
            print(f"   ⚠️ Lỗi sinh Heatmap cho {m_name}: {e}")
            ax.imshow(img_raw)
            ax.set_title(f"{m_name} (Render Error)", fontsize=11, fontweight='bold')

        ax.axis('off')

    plt.tight_layout()
    pdf_path = os.path.join(save_dir, "vision_explainability_gradcam.pdf")
    png_path = os.path.join(save_dir, "vision_explainability_gradcam.png")
    plots_png_path = os.path.join("plots", "vision_explainability_gradcam.png")

    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    plt.savefig(plots_png_path, format='png', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"\n============================================================")
    print(f"🖼️ Đã xuất biểu đồ Grad-CAM++ chất lượng IEEE công phu vào:")
    print(f"   - PDF (IEEE Paper) : {pdf_path}")
    print(f"   - PNG (IEEE Paper) : {png_path}")
    print(f"   - PNG (Plots)      : {plots_png_path}")
    print(f"============================================================")


# ------------------------------------------------------------------------------
# 5. Hàm thực thi chính
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Tự động nạp Checkpoint mô hình Stage 1 và xuất ảnh Grad-CAM++ cho Bài báo IEEE.")
    parser.add_argument('--image_path', type=str, default=None, help="Đường dẫn đến file ảnh giao thông cần vẽ Grad-CAM++.")
    parser.add_argument('--image_dir', type=str, default="/workspace/GRAPH/images", help="Thư mục chứa tệp ảnh giao thông.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    target_img = args.image_path
    if not target_img or not os.path.exists(target_img):
        # Tự động tìm kiếm file ảnh mẫu trong hệ thống
        candidates = []
        for d in [args.image_dir, "images", "data", "/workspace/images", "."]:
            found = glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png"))
            if found:
                candidates.extend(found)
        if candidates:
            target_img = candidates[0]
            print(f"🔍 Tự động phát hiện tệp ảnh giao thông mẫu: {target_img}")
        else:
            print("❌ Không tìm thấy ảnh giao thông (.jpg / .png) nào trong hệ thống.")
            return

    generate_vision_explainability_figures(target_img, save_dir="paper/fig", device=device)

if __name__ == '__main__':
    main()
