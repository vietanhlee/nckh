import os
# Cấu hình PyTorch Allocator tránh phân mảnh bộ nhớ CUDA OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import copy
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm.auto import tqdm
import timm

# =====================================================================
# ⚙️ GLOBAL CONFIGURATION & FILE PATHS (CẤU HÌNH ĐƯỜNG DẪN & MÔ HÌNH)
# Dễ dàng chỉnh sửa đường dẫn tệp, họ mô hình và các siêu tham số tại đây:
# =====================================================================
class Config:
    # 📂 Đường dẫn tệp CSV nhãn và thư mục ảnh
    CSV_FILE = "/kaggle/input/datasets/canhdoo/csv-images/traffic_update.csv"
    IMAGE_DIR = "/kaggle/input/datasets/canhdoo/lane-vehicle/images"

    # 🏗️ Danh sách họ mô hình cần chạy benchmark: 'resnet', 'efficientnet', 'convnext'
    MODELS = ['resnet', 'efficientnet', 'convnext']

    # 🧪 Danh sách seeds ngẫu nhiên để đánh giá thống kê (Mean ± Std)
    SEEDS = [42, 100, 2024]

    # ⚡ Siêu tham số huấn luyện
    EPOCHS = 60
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
    Hỗ trợ 3 họ mô hình tiêu chuẩn: ResNet (ResNet-50), EfficientNet (EfficientNet-B0), ConvNeXt (ConvNeXt-Tiny).
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
    else:
        raise ValueError(f"Tên mô hình không hợp lệ: {model_name}. Chọn một trong các loại: 'resnet', 'efficientnet', 'convnext'")

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

    return {
        'mae_overall': mae_overall, 'mape_overall': mape_overall, 'rmse_overall': rmse_overall, 'mse_overall': mse_overall,
        'mae_car': mae_car, 'mape_car': mape_car, 'rmse_car': rmse_car, 'mse_car': mse_car,
        'mae_moto': mae_moto, 'mape_moto': mape_moto, 'rmse_moto': rmse_moto, 'mse_moto': mse_moto
    }


def train_single_seed_counting(model_name, train_loader, val_loader, test_loader, cfg, device, seed):
    """Huấn luyện và đánh giá 1 mô hình đếm phương tiện với 1 seed ngẫu nhiên."""
    set_seed(seed)
    
    model = build_counting_model(model_name, num_classes=2, pretrained=True).to(device)
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * cfg['epochs'])
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_val_mae = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())

    print(f"\n⚡ [{model_name.upper()}] Seed {seed} | Bắt đầu huấn luyện ({cfg['epochs']} Epochs)...")

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

            scheduler.step()
            train_loss += loss.item() * len(images)

        train_loss /= len(train_loader.dataset)

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
            best_model_weights = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0 or epoch == cfg['epochs']:
            print(f"   Epoch {epoch:>2d}/{cfg['epochs']} | Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.2f} (Best: {best_val_mae:.2f})")

    # Đánh giá trên tập Test với trọng số tốt nhất
    model.load_state_dict(best_model_weights)
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
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS, help="Số epochs huấn luyện.")
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE, help="Kích thước batch_size.")
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE, help="Learning rate.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"============================================================")
    print(f"🚀 CHẠY BENCHMARK ĐẾM PHƯƠNG TIỆN (SUB-PROBLEM 1)")
    print(f"   Device        : {device}")
    print(f"   Seeds         : {args.seeds}")
    print(f"   Epochs        : {args.epochs}")
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
        # Fallback đường dẫn cục bộ nếu không chạy trên Kaggle
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

    # Đảm bảo chia tập nhất quán
    set_seed(42)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(full_dataset, [n_train, n_val, n_test])
    
    # Gán transform tách biệt cho Val & Test (không áp dụng Augmentation khi đánh giá)
    val_ds.dataset = copy.deepcopy(full_dataset)
    val_ds.dataset.transform = val_transform
    test_ds.dataset = copy.deepcopy(full_dataset)
    test_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"📂 Phân chia dữ liệu: Train={n_train}, Val={n_val}, Test={n_test}")

    models_to_test = Config.MODELS
    cfg = {'epochs': args.epochs, 'lr': args.lr}

    results = {
        m_name: {
            'params': 0, 'flops_gflops': 0.0, 'inf_latencies': [],
            'mae_overall': [], 'mape_overall': [], 'rmse_overall': [], 'mse_overall': [],
            'mae_car': [], 'mape_car': [], 'rmse_car': [],
            'mae_moto': [], 'mape_moto': [], 'rmse_moto': []
        } for m_name in models_to_test
    }

    for seed in args.seeds:
        print(f"\n{'='*70}")
        print(f"🧪 [SUB-PROBLEM 1 - SEED {seed}] THỬ NGHIỆM 3 MÔ HÌNH (RESNET, EFFICIENTNET, CONVNEXT)")
        print(f"{'='*70}")

        for m_name in models_to_test:
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            model, metrics = train_single_seed_counting(
                m_name, train_loader, val_loader, test_loader, cfg, device, seed
            )

            # Đo số tham số, FLOPs và Latency
            p_count = count_parameters(model)
            results[m_name]['params'] = p_count

            dummy_img, _ = next(iter(test_loader))
            dummy_img = dummy_img.to(device)
            gflops = count_flops(model, dummy_img)
            results[m_name]['flops_gflops'] = gflops

            lat = measure_inference_latency(model, test_loader, device)
            results[m_name]['inf_latencies'].append(lat)

            # Thu thập chỉ số Test
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

            print(f"   ▶ Seed {seed:>4} | {m_name.upper():<12} (Params: {p_count:,} | FLOPs: {gflops:.3f}G) -> "
                  f"MAE Tổng: {metrics['mae_overall']:.2f} (Ô tô: {metrics['mae_car']:.2f}, Xe máy: {metrics['mae_moto']:.2f})")

            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

    # 3. Tổng hợp Bảng so sánh kết quả
    table_overall = []
    table_breakdown = []

    for m_name in models_to_test:
        res = results[m_name]
        p_count = res['params']
        gflops = res['flops_gflops']
        lats = res['inf_latencies']

        # Bảng Tổng quan
        table_overall.append({
            'Model Architecture': m_name.upper(),
            'Params': f"{p_count:,}",
            'FLOPs (GFLOPs)': f"{gflops:.3f}",
            'Latency (ms/batch)': f"{np.mean(lats):.2f} ± {np.std(lats):.2f}",
            'MAE Overall': f"{np.mean(res['mae_overall']):.2f} ± {np.std(res['mae_overall']):.2f}",
            'MAPE Overall (%)': f"{np.mean(res['mape_overall'])*100:.2f}% ± {np.std(res['mape_overall'])*100:.2f}%",
            'RMSE Overall': f"{np.mean(res['rmse_overall']):.2f} ± {np.std(res['rmse_overall']):.2f}",
            'MSE Overall': f"{np.mean(res['mse_overall']):.2f} ± {np.std(res['mse_overall']):.2f}"
        })

        # Bảng Tách riêng Ô tô vs Xe máy
        table_breakdown.append({
            'Model Architecture': m_name.upper(),
            'MAE Ô tô (Car)': f"{np.mean(res['mae_car']):.2f} ± {np.std(res['mae_car']):.2f}",
            'MAPE Ô tô (%)': f"{np.mean(res['mape_car'])*100:.2f}% ± {np.std(res['mape_car'])*100:.2f}%",
            'RMSE Ô tô': f"{np.mean(res['rmse_car']):.2f} ± {np.std(res['rmse_car']):.2f}",
            'MAE Xe máy (Moto)': f"{np.mean(res['mae_moto']):.2f} ± {np.std(res['mae_moto']):.2f}",
            'MAPE Xe máy (%)': f"{np.mean(res['mape_moto'])*100:.2f}% ± {np.std(res['mape_moto'])*100:.2f}%",
            'RMSE Xe máy': f"{np.mean(res['rmse_moto']):.2f} ± {np.std(res['rmse_moto']):.2f}"
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

    # Ghi báo cáo ra file Markdown
    report_path = Config.REPORT_PATH
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# 🚗🏍️ Báo cáo Benchmark Mô hình Đếm Phương tiện (Sub-problem 1)\n\n")
        f.write(f"- **Seeds sử dụng**: `{args.seeds}`\n")
        f.write(f"- **Cấu hình**: Epochs={args.epochs}, Batch Size={args.batch_size}, LR={args.lr}\n")
        f.write(f"- **Họ mô hình so sánh**: ResNet (ResNet-50), EfficientNet (EfficientNet-B0), ConvNeXt (ConvNeXt-Tiny)\n\n")
        f.write("## 🏆 1. Bảng So sánh Tổng quan (Params, FLOPs, Latency & Overall Error)\n\n")
        f.write(df_overall.to_markdown(index=False))
        f.write("\n\n---\n\n## 🏍️🚗 2. Bảng So sánh Chi tiết Tách riêng Ô tô (Car) và Xe máy (Motorcycle)\n\n")
        f.write(df_breakdown.to_markdown(index=False))

    print(f"\n📑 Đã lưu báo cáo chi tiết đếm phương tiện vào tệp: {report_path}")


if __name__ == "__main__":
    run_counting_benchmark()