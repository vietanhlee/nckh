"""
coral_focal.py — Phân loại mức độ tắc nghẽn giao thông
Phiên bản 1: ConvNeXt-Tiny + Spatial Attention Head + CORAL Ordinal Focal Loss

Kiến trúc:
  Input (B, 3, 256, 512)
  → ConvNeXt-Tiny backbone → f32 (B, 768, 8, 16)
  → Spatial Attention Head → attn_map (B, 1, 8, 16)  [density-like map]
  → Weighted Average Pooling → (B, 768)
  → CORAL ordinal head → 4 logits → Sigmoid → mức 1-5

Attention map visualize trực tiếp (không cần Grad-CAM/thư viện ngoài)
để kiểm chứng model nhìn vào xe hay background.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import timm
from torch.nn import functional as F
import seaborn as sns

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

# ============================================================
# Cấu hình
# ============================================================
INPUT_H = 256
INPUT_W = 512
NUM_CLASSES = 5
BATCH_SIZE = 64
NUM_EPOCHS = 100
PATIENCE = 20
LR_BACKBONE = 2e-5
LR_HEAD = 2e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.05
NUM_WORKERS = 4  # Đặt 0 nếu gặp lỗi multiprocessing trên Windows


# ============================================================
# CORAL Ordinal Utilities
# ============================================================

def labels_to_ordinal_targets(labels, num_classes=NUM_CLASSES):
    """CORAL-style targets: t_k = 1 if y > k else 0.

    labels: LongTensor shape [N] with values in [0..C-1]
    returns FloatTensor shape [N, C-1]
    """
    labels = labels.long().view(-1)
    thresholds = torch.arange(0, num_classes - 1, device=labels.device).view(1, -1)
    return (labels.view(-1, 1) > thresholds).float()


def ordinal_predict_from_probs(probs, thresholds):
    """Predict class label from per-threshold probabilities.

    probs: Tensor [N, C-1] where probs[:, k] = P(y > k)
    thresholds: list/np.array/Tensor length (C-1)
    returns: LongTensor [N] with predicted labels in [0..C-1]
    """
    thr = torch.as_tensor(thresholds, device=probs.device, dtype=probs.dtype).view(1, -1)
    return (probs > thr).sum(dim=1).long()


def tune_thresholds_by_accuracy(y_true, y_probs, initial=None, grid=None, max_passes=3):
    """Tune per-head thresholds to maximize overall multiclass accuracy on validation.

    This uses coordinate-ascent over a small grid for speed.
    y_true: np.array [N] labels 0..C-1
    y_probs: np.array [N, C-1] sigmoid probs for heads (y > k)
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    num_heads = y_probs.shape[1]

    if initial is None:
        thr = np.full(num_heads, 0.5, dtype=np.float32)
    else:
        thr = np.asarray(initial, dtype=np.float32).copy()

    if grid is None:
        # vừa đủ mịn để tối ưu, không quá nặng
        grid = np.linspace(0.05, 0.95, 19, dtype=np.float32)
    else:
        grid = np.asarray(grid, dtype=np.float32)

    def acc_for(thresholds):
        pred = (y_probs > thresholds[None, :]).sum(axis=1).astype(np.int32)
        return float((pred == y_true).mean())

    best_acc = acc_for(thr)
    for _ in range(int(max_passes)):
        improved = False
        for k in range(num_heads):
            best_t = float(thr[k])
            for t in grid:
                cand = thr.copy()
                cand[k] = float(t)
                a = acc_for(cand)
                if a > best_acc + 1e-12:
                    best_acc = a
                    best_t = float(t)
                    improved = True
            thr[k] = best_t
        if not improved:
            break

    return thr


# ============================================================
# Ordinal Focal Loss
# ============================================================

class OrdinalFocalLoss(nn.Module):
    """Focal Loss cho CORAL ordinal regression.

    Kết hợp:
    - Focal modulating factor (1-p_t)^gamma → focus vào hard examples
    - Per-head pos_weight → xử lý class imbalance ở mỗi ngưỡng
    - Label smoothing nhẹ → tránh overconfident
    """

    def __init__(self, pos_weight=None, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if pos_weight is not None:
            self.register_buffer('pos_weight', pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits, targets):
        # Label smoothing: target 1 → 1-ε/2, target 0 → ε/2
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # BCE numerically stable (logsigmoid)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Focal modulating factor
        p = torch.sigmoid(logits)
        p_t = targets * p + (1 - targets) * (1 - p)
        focal_weight = (1 - p_t).pow(self.gamma)

        # Per-head pos_weight cho imbalance
        if self.pos_weight is not None:
            alpha = targets * self.pos_weight.unsqueeze(0) + (1 - targets)
            bce = bce * alpha

        return (focal_weight * bce).mean()


# ============================================================
# EarlyStopping
# ============================================================

class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_acc):
        if self.best_score is None:
            self.best_score = current_acc
        elif current_acc < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_acc
            self.counter = 0


# ============================================================
# Dataset
# ============================================================

class VehicleDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        raw_data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        # Lọc sớm các mẫu lỗi để DataLoader không bị dừng giữa chừng.
        valid_rows = []
        skipped_missing = 0
        skipped_label = 0

        for _, row in raw_data.iterrows():
            try:
                label = int(row['phan_loai'])
            except Exception:
                skipped_label += 1
                continue

            if label < 1 or label > 5:
                skipped_label += 1
                continue

            filename = str(row['filename'])
            img_path = os.path.join(self.image_dir, filename)
            if not os.path.exists(img_path):
                skipped_missing += 1
                continue

            valid_rows.append(row)

        self.data = pd.DataFrame(valid_rows).reset_index(drop=True)

        if skipped_missing or skipped_label:
            print(
                f"Filtered dataset: kept {len(self.data)}/{len(raw_data)} samples | "
                f"missing images={skipped_missing}, invalid labels={skipped_label}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')

        # Chuyển nhãn 1-5 thành 0-4
        label = int(row['phan_loai']) - 1

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

    def clone_with_transform(self, transform):
        """Tạo bản sao dataset với transform khác (tránh đọc lại CSV)."""
        new_ds = VehicleDataset.__new__(VehicleDataset)
        new_ds.data = self.data  # Chia sẻ DataFrame (read-only)
        new_ds.image_dir = self.image_dir
        new_ds.transform = transform
        return new_ds


# ============================================================
# Model — Spatial Attention Head
# ============================================================

class SpatialAttentionHead(nn.Module):
    """Sinh bản đồ mật độ chú ý (density-like attention map) từ feature map.

    Output: sigmoid attention map (B, 1, H', W') ∈ [0, 1]
    Vùng có xe → giá trị cao, vùng background → giá trị thấp.
    Có thể upsample và chồng lên ảnh gốc để kiểm chứng.
    """

    def __init__(self, in_channels=768, mid_channels=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.conv(x))


class TrafficCongestionNet(nn.Module):
    """ConvNeXt-Tiny + Spatial Attention Head + CORAL ordinal regression.

    Forward output: dict with keys:
        - 'logits': (B, C-1) ordinal logits
        - 'attn_map': (B, 1, H', W') spatial attention map
    """

    def __init__(self, num_classes=NUM_CLASSES, hidden_dim=256, dropout=0.3, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1

        # Lấy components từ ConvNeXt-Tiny pretrained
        full_model = timm.create_model('convnext_tiny', pretrained=pretrained)
        self.stem = full_model.stem
        self.stages = full_model.stages
        self.norm_pre = full_model.norm_pre
        feat_dim = full_model.num_features  # 768
        del full_model

        # Spatial Attention Head → density-like attention map
        self.attn_head = SpatialAttentionHead(feat_dim, mid_channels=128)

        # CORAL ordinal classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, self.num_thresholds),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        f = self.norm_pre(x)  # (B, 768, H/32, W/32)

        # Spatial attention map
        attn_map = self.attn_head(f)  # (B, 1, H/32, W/32)

        # Weighted Average Pooling: chú ý vùng xe, bỏ qua background
        weighted = f * attn_map  # (B, 768, H/32, W/32)
        attn_sum = attn_map.sum(dim=(2, 3)) + 1e-6  # (B, 1)
        pooled = weighted.sum(dim=(2, 3)) / attn_sum  # (B, 768)

        logits = self.classifier(pooled)  # (B, C-1)

        return {'logits': logits, 'attn_map': attn_map}


# ============================================================
# Data Split
# ============================================================

def get_split_indices(dataset, val_ratio=0.2, seed=42):
    """Stratified split → trả về (train_indices, val_indices).

    Nếu một lớp chỉ có 1 mẫu, mẫu đó sẽ được đưa vào train.
    """
    labels = (dataset.data['phan_loai'].values - 1).astype(int)
    num_classes = int(labels.max() + 1) if labels.size > 0 else 0

    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(dataset))

    train_indices = []
    val_indices = []

    for c in range(num_classes):
        cls_idx = all_indices[labels == c]
        if cls_idx.size == 0:
            continue
        rng.shuffle(cls_idx)

        n = int(cls_idx.size)
        if n <= 1:
            n_val = 0
        else:
            n_val = int(round(val_ratio * n))
            n_val = max(1, n_val)
            n_val = min(n_val, n - 1)

        val_indices.extend(cls_idx[:n_val].tolist())
        train_indices.extend(cls_idx[n_val:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


# ============================================================
# Transforms — Scene-Global Safe (KHÔNG CẮT ẢNH)
# ============================================================

def get_train_transform():
    """Augmentation an toàn: chỉ các phép KHÔNG thay đổi nội dung ngữ nghĩa.

    ⚠️  TUYỆT ĐỐI KHÔNG dùng RandomCrop/RandomResizedCrop!
    Nhãn (mức ùn tắc) là của TOÀN BỘ ảnh.
    """
    return transforms.Compose([
        transforms.Resize((INPUT_H, INPUT_W)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2
        ),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform():
    """Transform cho val/test: chỉ resize + normalize."""
    return transforms.Compose([
        transforms.Resize((INPUT_H, INPUT_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ============================================================
# Training & Evaluation
# ============================================================

def train_epoch(epoch, model, loader, loss_func, optimizer, scheduler, device, scaler):
    model.train()
    epoch_loss = 0.0
    running_correct = 0
    running_total = 0
    default_thr = [0.5] * (NUM_CLASSES - 1)

    if tqdm is not None:
        iterator = tqdm(loader, desc=f"Train epoch {epoch + 1}", leave=False)
    else:
        iterator = loader

    for i, (images, labels) in enumerate(iterator):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            output = model(images)
            logits = output['logits']
            targets = labels_to_ordinal_targets(labels)
            loss = loss_func(logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        epoch_loss += loss.item()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = ordinal_predict_from_probs(probs, thresholds=default_thr)
            running_correct += (preds == labels).sum().item()
            running_total += labels.numel()

        if tqdm is not None:
            current_lr = optimizer.param_groups[0]['lr']
            batch_acc = running_correct / max(1, running_total)
            iterator.set_postfix({
                'loss': f"{(epoch_loss / (i + 1)):.4f}",
                'acc': f"{batch_acc:.3f}",
                'lr': f"{current_lr:.2e}",
            })

    avg_loss = epoch_loss / max(1, len(loader))
    train_acc = running_correct / max(1, running_total)
    return avg_loss, train_acc


def test_epoch(model, loader, loss_func, device):
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []
    total_loss = 0.0
    total_samples = 0
    default_thr = [0.5] * (NUM_CLASSES - 1)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            output = model(images)
            logits = output['logits']
            targets = labels_to_ordinal_targets(labels)
            loss = loss_func(logits, targets)
            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            probs = torch.sigmoid(logits)
            predicted = ordinal_predict_from_probs(probs, thresholds=default_thr)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / max(1, total_samples)
    return avg_loss, np.array(y_true), np.array(y_pred), np.array(y_probs)


# ============================================================
# Visualization — Bản đồ mật độ chú ý
# ============================================================

def visualize_attention_map(model, dataset, indices, device, thresholds=None,
                            num_images=10, save_path='attention_heatmap_samples.png'):
    """Vẽ bản đồ mật độ chú ý (attention map) chồng lên ảnh gốc.

    Nếu điểm nóng trùng vùng xe → model đúng.
    Nếu điểm nóng ở bầu trời/biển hiệu → model đang đoán từ bối cảnh.
    """
    if thresholds is None:
        thresholds = [0.5] * (NUM_CLASSES - 1)

    model.eval()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    num_to_show = min(num_images, len(indices))
    fig, axes = plt.subplots(num_to_show, 2, figsize=(14, 4 * num_to_show))
    if num_to_show == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for i in range(num_to_show):
            idx = indices[i]
            img_tensor, label = dataset[idx]
            x = img_tensor.unsqueeze(0).to(device)

            output = model(x)
            logits = output['logits']
            attn = output['attn_map'][0, 0].cpu()  # (H', W')

            probs = torch.sigmoid(logits)
            pred = ordinal_predict_from_probs(probs, thresholds).item()

            # Un-normalize image
            img_np = img_tensor.permute(1, 2, 0).numpy()
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)

            # Upsample attention map về kích thước ảnh gốc
            heat = F.interpolate(
                attn.unsqueeze(0).unsqueeze(0),
                size=img_np.shape[:2],
                mode='bilinear', align_corners=False
            )[0, 0].numpy()

            # Ảnh gốc
            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title(f"Ảnh gốc — Nhãn thực: {label.item() + 1}", fontsize=10)
            axes[i, 0].axis('off')

            # Heatmap chồng lên ảnh
            axes[i, 1].imshow(img_np)
            axes[i, 1].imshow(heat, alpha=0.55, cmap='jet')
            axes[i, 1].set_title(f"Bản đồ chú ý — Dự đoán: {pred + 1}", fontsize=10)
            axes[i, 1].axis('off')

    plt.suptitle("Kiểm chứng: Model nhìn vào xe hay background?", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Đã lưu attention heatmap tại: {save_path}")


# ============================================================
# Visualization — Confusion Matrix + Misclassifications
# ============================================================

def visualize_misclassifications(model, loader, device, thresholds=None, num_images=20):
    model.eval()
    all_images = []
    all_preds = []
    all_labels = []

    if thresholds is None:
        thresholds = [0.5] * (NUM_CLASSES - 1)

    with torch.no_grad():
        for images, labels in loader:
            images_dev = images.to(device)
            output = model(images_dev)
            logits = output['logits']
            probs = torch.sigmoid(logits)
            predicted = ordinal_predict_from_probs(probs, thresholds=thresholds)

            all_images.append(images.cpu())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_images = torch.cat(all_images)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Hiển thị 1-indexed (Lớp 1..5)
    display_preds = all_preds + 1
    display_labels = all_labels + 1

    # 1. Confusion Matrix
    cm = confusion_matrix(display_labels, display_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Lớp {i}' for i in range(1, 6)],
                yticklabels=[f'Lớp {i}' for i in range(1, 6)])
    plt.xlabel('Dự đoán (Predicted)')
    plt.ylabel('Thực tế (Actual)')
    plt.title('Confusion Matrix - Phân tích lỗi nhầm lẫn')
    plt.show()

    # 2. Ảnh bị đoán sai
    misclassified_idx = np.where(all_preds != all_labels)[0]
    print(f"Tổng số mẫu bị đoán sai: {len(misclassified_idx)}/{len(all_labels)}")

    if len(misclassified_idx) > 0:
        plt.figure(figsize=(30, 25))
        num_to_show = min(len(misclassified_idx), num_images)

        ncols = 5
        nrows = (num_to_show + ncols - 1) // ncols

        for i in range(num_to_show):
            idx = misclassified_idx[i]
            img = all_images[idx].permute(1, 2, 0).numpy()

            # Un-normalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)

            plt.subplot(nrows, ncols, i + 1)
            plt.imshow(img)
            plt.title(f"A: {display_labels[idx]} | P: {display_preds[idx]}", color='red')
            plt.axis('off')

        plt.suptitle("Các mẫu dự đoán sai tiêu biểu", fontsize=16)
        plt.tight_layout()
        plt.show()


# ============================================================
# EDA
# ============================================================

def run_eda(dataset, image_dir, num_preview_per_class=2):
    raw_labels = pd.to_numeric(dataset.data['phan_loai'], errors='coerce')
    filenames = dataset.data['filename'].astype(str).tolist()
    num_classes = NUM_CLASSES
    class_names = [f'Lớp {i}' for i in range(1, num_classes + 1)]
    valid_mask = raw_labels.between(1, num_classes)
    labels = (raw_labels[valid_mask].astype(int) - 1).to_numpy()
    invalid_label_count = int((~valid_mask).sum())

    print("\n" + "=" * 60)
    print("EDA: Dataset overview")
    print(f"Total samples: {len(dataset)}")
    print(f"Total classes: {num_classes}")
    print("Columns:", list(dataset.data.columns))
    print("Missing values per column:")
    print(dataset.data.isna().sum().to_string())
    print(f"Invalid/out-of-range labels (expected 1..{num_classes}): {invalid_label_count}")

    duplicate_files = dataset.data['filename'].duplicated().sum()
    print(f"Duplicate filenames: {int(duplicate_files)}")

    if labels.size > 0:
        class_counts = np.bincount(labels, minlength=num_classes)
    else:
        class_counts = np.zeros(num_classes, dtype=int)
    class_pct = class_counts / max(1, len(labels)) * 100.0
    print("Class distribution:")
    for idx, (count, pct) in enumerate(zip(class_counts, class_pct), start=1):
        print(f"  - Class {idx}: {int(count)} samples ({pct:.2f}%)")

    missing_images = [fn for fn in filenames if not os.path.exists(os.path.join(image_dir, fn))]
    print(f"Missing image files: {len(missing_images)}")
    if missing_images:
        print("First missing files:", missing_images[:10])

    plt.figure(figsize=(8, 4))
    sns.barplot(x=class_names, y=class_counts, palette='viridis')
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('eda_class_distribution.png')
    plt.show()

    preview_images = []
    preview_titles = []
    seen_per_class = {c: 0 for c in range(num_classes)}
    for _, row in dataset.data.iterrows():
        cls_raw = pd.to_numeric(pd.Series([row['phan_loai']]), errors='coerce').iloc[0]
        if pd.isna(cls_raw):
            continue
        cls = int(cls_raw) - 1
        if cls < 0 or cls >= num_classes:
            continue
        if seen_per_class[cls] >= num_preview_per_class:
            continue
        img_path = os.path.join(image_dir, str(row['filename']))
        if not os.path.exists(img_path):
            continue
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            continue
        preview_images.append(image)
        preview_titles.append(f"{class_names[cls]}\n{row['filename']}")
        seen_per_class[cls] += 1
        if all(v >= num_preview_per_class for v in seen_per_class.values()):
            break

    if preview_images:
        n = len(preview_images)
        ncols = min(4, n)
        nrows = int(np.ceil(n / ncols))
        plt.figure(figsize=(4 * ncols, 4 * nrows))
        for i, (img, title) in enumerate(zip(preview_images, preview_titles)):
            plt.subplot(nrows, ncols, i + 1)
            plt.imshow(img)
            plt.title(title, fontsize=9)
            plt.axis('off')
        plt.suptitle('EDA: Sample images per class', fontsize=14)
        plt.tight_layout()
        plt.savefig('eda_sample_images.png')
        plt.show()

    print("=" * 60)


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    csv_file = '/kaggle/input/datasets/huecute/csv-images/traffic_final_windows_order.csv'
    image_dir = '/kaggle/input/datasets/huecute/images/images'

    # ── 1. Dataset & EDA ──────────────────────────────────
    base_dataset = VehicleDataset(csv_file, image_dir, transform=None)

    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model: ConvNeXt-Tiny + Spatial Attention Head")
    print(f"Input size: {INPUT_H}×{INPUT_W}")
    print(f"Dataset size: {len(base_dataset)}")

    run_eda(base_dataset, image_dir)

    # ── 2. Split & Transforms ─────────────────────────────
    train_indices, val_indices = get_split_indices(base_dataset)

    train_dataset = base_dataset.clone_with_transform(get_train_transform())
    val_dataset = base_dataset.clone_with_transform(get_val_transform())

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    # ── 3. Class weights (tính từ TRAIN split) ────────────
    labels_all = (base_dataset.data['phan_loai'].values - 1).astype(int)
    train_labels = labels_all[np.array(train_indices)]
    val_labels = labels_all[np.array(val_indices)]

    train_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    val_counts = np.bincount(val_labels, minlength=NUM_CLASSES)

    # pos_weight cho từng head CORAL (y > k)
    pos_weight = []
    for k in range(NUM_CLASSES - 1):
        y_bin = (train_labels > k).astype(np.int32)
        pos = float(y_bin.sum())
        neg = float(len(y_bin) - pos)
        pw = neg / pos if pos > 0 else 1.0
        pos_weight.append(pw)
    pos_weight = torch.tensor(pos_weight, dtype=torch.float, device=device)

    print(f"Train samples per class: {train_counts}")
    print(f"Val samples per class:   {val_counts}")
    print(f"CORAL pos_weight:        {np.round(pos_weight.cpu().numpy(), 4)}")
    print("=" * 60)

    # ── 4. DataLoaders ────────────────────────────────────
    pin_mem = device.type == 'cuda'
    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin_mem, drop_last=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin_mem,
    )

    # ── 5. Model ──────────────────────────────────────────
    model = TrafficCongestionNet(
        num_classes=NUM_CLASSES, hidden_dim=256, dropout=0.3, pretrained=True
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # ── 6. Loss, Optimizer, Scheduler ─────────────────────
    loss_func = OrdinalFocalLoss(pos_weight=pos_weight, gamma=FOCAL_GAMMA)

    # Differential LR: backbone thấp (giữ pretrained), head cao (học nhanh)
    backbone_params = list(model.stem.parameters()) + \
                      list(model.stages.parameters()) + \
                      list(model.norm_pre.parameters())
    head_params = list(model.attn_head.parameters()) + \
                  list(model.classifier.parameters())

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': LR_BACKBONE},
        {'params': head_params, 'lr': LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    # Warmup + Cosine schedule
    steps_per_epoch = len(train_loader)
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    total_steps = NUM_EPOCHS * steps_per_epoch

    warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_steps])

    # AMP (Mixed Precision)
    amp_enabled = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)

    # ── 7. Training Loop ──────────────────────────────────
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_acc_tuned': [],
    }
    best_acc = 0.0
    save_path = ""
    best_thresholds = np.full(NUM_CLASSES - 1, 0.5, dtype=np.float32)

    for epoch in range(NUM_EPOCHS):
        print(f"\n[Epoch {epoch + 1}/{NUM_EPOCHS}]")

        avg_train_loss, train_acc = train_epoch(
            epoch, model, train_loader, loss_func, optimizer, scheduler, device, scaler
        )
        val_loss, y_true, y_pred, y_probs = test_epoch(model, val_loader, loss_func, device)

        acc = accuracy_score(y_true, y_pred)
        tuned_thresholds = tune_thresholds_by_accuracy(y_true, y_probs)
        y_pred_tuned = ordinal_predict_from_probs(
            torch.from_numpy(y_probs).float(), thresholds=tuned_thresholds
        ).cpu().numpy()
        acc_tuned = accuracy_score(y_true, y_pred_tuned)

        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc: {acc:.4f} | Val Acc (tuned): {acc_tuned:.4f}")

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(acc)
        history['val_acc_tuned'].append(acc_tuned)

        if acc_tuned > best_acc:
            best_acc = acc_tuned
            save_path = f"best_model_convnext_attn_{acc_tuned:.2f}.pth"
            torch.save(model.state_dict(), save_path)
            best_thresholds = tuned_thresholds
            print(f"  ✅ Saved best model (ValAcc_tuned={acc_tuned:.4f})")

        early_stopping(acc_tuned)
        if early_stopping.early_stop:
            print(f"\n🛑 Early stopping triggered tại epoch {epoch + 1}.")
            break

    # ── 8. Training Curves ────────────────────────────────
    print("\nTraining finished. Plotting results...")
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', color='red')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss', color='orange')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train Acc', color='purple')
    plt.plot(epochs_range, history['val_acc'], label='Val Acc', color='blue')
    plt.plot(epochs_range, history['val_acc_tuned'], label='Val Acc (tuned)', color='green')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics_convnext_attn.png')
    plt.show()

    # ── 9. Final Evaluation ───────────────────────────────
    print("\nFinal Evaluation:")
    print(classification_report(
        y_true, y_pred,
        target_names=[f'Lớp {i}' for i in range(1, NUM_CLASSES + 1)]
    ))

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        best_val_loss, best_y_true, best_y_pred, best_y_probs = test_epoch(
            model, val_loader, loss_func, device
        )

        # Tune thresholds trên best model
        thresholds = tune_thresholds_by_accuracy(
            best_y_true, best_y_probs, initial=best_thresholds
        )
        y_pred_opt = ordinal_predict_from_probs(
            torch.from_numpy(best_y_probs).float(), thresholds=thresholds
        ).cpu().numpy()

        print(f"\nBest model — Thresholds: {np.round(thresholds, 4)}")
        print(f"Best model — Accuracy:   {accuracy_score(best_y_true, y_pred_opt):.4f}")
        print("\nClassification Report (best thresholds):")
        print(classification_report(
            best_y_true, y_pred_opt,
            target_names=[f'Lớp {i}' for i in range(1, NUM_CLASSES + 1)]
        ))

        # Confusion Matrix & Misclassifications
        visualize_misclassifications(model, val_loader, device, thresholds=thresholds)

        # ── 10. Attention Heatmap ─────────────────────────
        print("\n🔍 Visualizing attention maps...")
        visualize_attention_map(
            model, val_dataset, val_indices, device,
            thresholds=thresholds, num_images=10,
            save_path='attention_heatmap_samples.png'
        )
    else:
        print("Không tìm thấy file lưu model tốt nhất.")


if __name__ == '__main__':
    main()