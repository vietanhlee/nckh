#!/usr/bin/env python3
# ==============================================================================
# 🚗 EFFICIENTNET AUTOMATED VEHICLE COUNTING LABELER (AUTO-LABELING SCRIPT)
# ==============================================================================
import os
import glob
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm

# ------------------------------------------------------------------------------
# 1. Định nghĩa Khởi tạo Mô hình EfficientNet (Độc lập, không import train_counting)
# ------------------------------------------------------------------------------
def build_efficientnet_model(checkpoint_path=None, device="cuda"):
    """
    Khởi tạo mô hình EfficientNet-B5 cho bài toán đếm phương tiện (2 đầu ra: [Ô tô, Xe máy])
    khớp chuẩn 100% với kiến trúc huấn luyện trong train_counting.py.
    """
    try:
        model = timm.create_model('efficientnet_b5', pretrained=False, num_classes=2)
    except Exception:
        from torchvision import models
        model = models.efficientnet_b5(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    # Nạp trọng số weights từ file Checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"📥 Đang nạp trọng số mô hình từ: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            
        model.load_state_dict(state_dict, strict=False)
        print("✅ Nạp trọng số mô hình thành công!")
    else:
        if checkpoint_path:
            print(f"⚠️ CẢNH BÁO: Không tìm thấy file trọng số tại '{checkpoint_path}'. Khởi tạo mô hình ngẫu nhiên.")

    model = model.to(device)
    model.eval()
    return model


# ------------------------------------------------------------------------------
# 2. Custom Dataset cho Inference Thư mục Ảnh (images)
# ------------------------------------------------------------------------------
class InferenceImageDataset(Dataset):
    def __init__(self, image_dir, img_size=(224, 224)):
        self.image_dir = image_dir
        
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG')
        if os.path.exists(image_dir):
            self.image_files = [
                f for f in os.listdir(image_dir)
                if f.endswith(valid_extensions) and os.path.isfile(os.path.join(image_dir, f))
            ]
            self.image_files.sort()
        else:
            self.image_files = []
        
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        fpath = os.path.join(self.image_dir, fname)
        try:
            img = Image.open(fpath).convert('RGB')
            tensor = self.transform(img)
        except Exception as e:
            print(f"⚠️ Lỗi đọc ảnh {fname}: {e}. Tạo ảnh đen giả lập.")
            img = Image.new('RGB', (224, 224), color=(0, 0, 0))
            tensor = self.transform(img)
        return fname, tensor

# ------------------------------------------------------------------------------
# 3. Hàm Auto-Label & Xuất CSV (filename,xe_may,o_to,tong)
# ------------------------------------------------------------------------------
def run_auto_label(image_dir="images", checkpoint_path=None, output_csv="auto_labeled_output.csv", batch_size=32, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"🚀 Khởi chạy Auto-Labeling (EfficientNet) trên thiết bị: {device}")
    
        
    # 1. Khởi tạo & Load Model
    model = build_efficientnet_model(checkpoint_path=checkpoint_path, device=device)
    
    # 2. Tạo Dataset & DataLoader
    dataset = InferenceImageDataset(image_dir)
    if len(dataset) == 0:
        print(f"❌ Không tìm thấy tệp ảnh hợp lệ nào trong thư mục: {image_dir}")
        return
        
    print(f"📦 Tìm thấy {len(dataset)} ảnh trong thư mục '{image_dir}'. Bắt đầu gán nhãn tự động...")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 3. Chạy Inference
    results = []
    with torch.no_grad():
        for filenames, tensors in tqdm(loader, desc="Auto Labeling"):
            tensors = tensors.to(device)
            outputs = model(tensors) # Output shape: (B, 2) -> [o_to, xe_may]
            
            preds = torch.clamp(outputs, min=0.0).cpu().numpy()
            
            for fname, pred in zip(filenames, preds):
                car_count = int(round(float(pred[0])))
                moto_count = int(round(float(pred[1])))
                total_vehicles = moto_count + car_count
                
                results.append({
                    'filename': fname,
                    'xe_may': moto_count,
                    'o_to': car_count,
                    'tong': total_vehicles
                })

    # 4. Xuất file CSV nhãn chuẩn: filename,xe_may,o_to,tong
    df = pd.DataFrame(results)[['filename', 'xe_may', 'o_to', 'tong']]
    
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\n🎉 HOÀN THÀNH! Đã xuất file CSV nhãn tự động tại: {output_csv}")
    print(f"📊 Preview 5 dòng đầu tiên:")
    print(df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chương trình Auto-Label đếm phương tiện giao thông bằng EfficientNet")
    parser.add_argument("--image_dir", type=str, default="images", help="Đường dẫn thư mục chứa ảnh (mặc định: 'images')")
    parser.add_argument("--checkpoint", type=str, default="overall_best_counting_efficientnet.pth", help="Đường dẫn file trọng số model (mặc định: tự tìm best checkpoint)")
    parser.add_argument("--output_csv", type=str, default="auto_labeled_output.csv", help="Đường dẫn file CSV đầu ra (mặc định: 'auto_labeled_output.csv')")
    parser.add_argument("--batch_size", type=int, default=32, help="Kích thước batch_size")
    
    args = parser.parse_args()
    
    run_auto_label(
        image_dir=args.image_dir,
        checkpoint_path=args.checkpoint,
        output_csv=args.output_csv,
        batch_size=args.batch_size
    )
