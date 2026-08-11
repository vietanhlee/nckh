import os
# Cấu hình PyTorch Allocator tránh phân mảnh bộ nhớ CUDA OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import random
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

# Import tiện ích khởi tạo mô hình từ train_counting
from train_counting import build_counting_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GradCAMPlusPlus:
    """
    Grad-CAM++ (Chattopadhay et al., WACV 2018) - Phương pháp giải thích trực quan chuẩn mực nhất
    cho bài toán đếm nhiều đối tượng mật độ cao (Multi-Object Vehicle Counting Regression).
    Sử dụng đạo hàm cấp 2 và cấp 3 để tính trọng số pixel-wise positive gradient, phản ánh chính xác
    từng vị trí xe máy và ô tô trong cảnh giao thông phức tạp.
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
    """Xác định lớp Convolution cuối cùng cho từng kiến trúc mô hình."""
    name_lower = model_name.lower()

    if 'resnet' in name_lower:
        if hasattr(model, 'layer4'):
            return model.layer4[-1]
        elif hasattr(model, 'stages'):
            return model.stages[-1]
    elif 'efficientnet' in name_lower:
        if hasattr(model, 'conv_head'):
            return model.conv_head
        elif hasattr(model, 'blocks'):
            return model.blocks[-1]
        elif hasattr(model, 'features'):
            return model.features[-1]
    elif 'convnext' in name_lower:
        if hasattr(model, 'stages'):
            return model.stages[-1]
        elif hasattr(model, 'features'):
            return model.features[-1]

    # Fallback cho timm models
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            return module

    raise ValueError(f"Không tìm thấy target layer cho {model_name}")


def generate_vision_explainability_figures(image_path, models_dict, device, save_dir="paper/fig"):
    """
    Tạo biểu đồ so sánh Grad-CAM Feature Attribution Heatmap giữa 3 mô hình vision.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Tiền xử lý ảnh giao thông
    raw_img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = raw_img.size

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = preprocess(raw_img).unsqueeze(0).to(device)

    # 2. Khởi tạo Figure lưới hiển thị 4 cột: Ảnh gốc, ResNet-50, EfficientNet-B4, ConvNeXt-Tiny
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=300)
    plt.rcParams['font.family'] = 'DejaVu Sans'

    # (Col 0) Ảnh giao thông thực tế gốc
    axes[0].imshow(raw_img)
    axes[0].set_title("(a) Traffic Camera Scene\n(Original Input)", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    model_names = ['resnet', 'efficientnet', 'convnext']
    titles = [
        "(b) ResNet-50\n(Grad-CAM++ Attribution)",
        "(c) EfficientNet-B4\n(Grad-CAM++ Attribution)",
        "(d) ConvNeXt-Tiny (Ours)\n(Grad-CAM++ Attribution)"
    ]

    for idx, m_key in enumerate(model_names):
        ax = axes[idx + 1]
        model = models_dict[m_key].to(device)
        model.eval()

        target_layer = get_target_layer(model, m_key)
        grad_cam = GradCAMPlusPlus(model, target_layer)

        cam_mask, preds = grad_cam.generate_heatmap(input_tensor)

        # Resize Heatmap về kích thước ảnh gốc
        cam_img = Image.fromarray((cam_mask * 255).astype(np.uint8)).resize((orig_w, orig_h), resample=Image.BILINEAR)
        cam_arr = np.array(cam_img) / 255.0

        # Phủ Heatmap JET mờ lên ảnh gốc
        ax.imshow(raw_img)
        ax.imshow(cam_arr, cmap='jet', alpha=0.55)
        ax.set_title(f"{titles[idx]}\nPreds: Cars={preds[0]:.1f}, Bikes={preds[1]:.1f}", fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()

    fig_pdf = os.path.join(save_dir, "vision_explainability_gradcam.pdf")
    fig_png = os.path.join(save_dir, "vision_explainability_gradcam.png")
    plt.savefig(fig_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(fig_png, format='png', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"🖼️ Đã tạo thành công biểu đồ Giải thích Mô hình Thị giác Grad-CAM:")
    print(f"   - PDF: {fig_pdf}")
    print(f"   - PNG: {fig_png}")


def main():
    parser = argparse.ArgumentParser(description="Tạo biểu đồ XAI Grad-CAM so sánh tính giải thích cho 3 mô hình Vision Giai đoạn 1.")
    parser.add_argument('--image_path', type=str, default="/kaggle/input/datasets/canhdoo/lane-vehicle/images/test_001.jpg", help="Đường dẫn tới 1 ảnh giao thông minh họa.")
    parser.add_argument('--save_dir', type=str, default="paper/fig", help="Thư mục lưu ảnh bài báo.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(42)

    # Khởi tạo 3 mô hình
    models_dict = {
        'resnet': build_counting_model('resnet', pretrained=True),
        'efficientnet': build_counting_model('efficientnet', pretrained=True),
        'convnext': build_counting_model('convnext', pretrained=True)
    }

    # Nếu có file ảnh demo hợp lệ, tạo đồ thị
    if os.path.exists(args.image_path):
        generate_vision_explainability_figures(args.image_path, models_dict, device, args.save_dir)
    else:
        print(f"⚠️ Không tìm thấy ảnh {args.image_path}. Script đã sẵn sàng chạy trên Kaggle GPU với đường dẫn ảnh thực tế.")


if __name__ == "__main__":
    main()
