import os
import argparse
import torch

from train_counting import build_counting_model, generate_vision_explainability_figures, set_seed

def main():
    parser = argparse.ArgumentParser(description="Tạo biểu đồ XAI Grad-CAM so sánh tính giải thích cho 4 mô hình Vision Giai đoạn 1.")
    parser.add_argument('--image_path', type=str, default="/kaggle/input/datasets/canhdoo/lane-vehicle/images/test_001.jpg", help="Đường dẫn tới 1 ảnh giao thông minh họa.")
    parser.add_argument('--save_dir', type=str, default="paper/fig", help="Thư mục lưu ảnh bài báo.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(42)

    # Khởi tạo 4 mô hình vision
    models_dict = {
        'resnet': build_counting_model('resnet', pretrained=True),
        'efficientnet': build_counting_model('efficientnet', pretrained=True),
        'vit': build_counting_model('vit', pretrained=True),
        'convnext': build_counting_model('convnext', pretrained=True)
    }

    if os.path.exists(args.image_path):
        generate_vision_explainability_figures(args.image_path, models_dict, device, args.save_dir)
    else:
        print(f"⚠️ Không tìm thấy ảnh {args.image_path}. Script đã sẵn sàng chạy trên Kaggle GPU với đường dẫn ảnh thực tế.")

if __name__ == "__main__":
    main()
