import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import timm



class VehicleDataset(Dataset):
  def __init__ (self, csv_file, image_dir, transforms = None):
    self.data = pd.read_csv(csv_file)
    self.image_dir = image_dir
    self.transforms = transforms

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    row = self.data.iloc[idx]
    img_path = os.path.join(self.image_dir, row['filename'])
    image = Image.open(img_path).convert('RGB')

    label = torch.tensor([row['o_to'], row['xe_may']], dtype = torch.float32)
    if self.transforms:
      image = self.transforms(image)

    return image, label



class VehicleModel(nn.Module):
    def __init__(self):
        super(VehicleModel, self).__init__()
        base_model = timm.create_model('vit_base_patch16_224', pretrained=True)

        in_features = base_model.head.in_features
        base_model.head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # đầu ra 2 số thực: [ô tô, xe máy]
        )

        self.model = base_model

    def forward(self, x):
        return self.model(x)


# class VehicleModel(nn.Module):
#   def __init__(self):
#     super(VehicleModel, self).__init__()
#     base_model = models.resnet50(pretrained = True)
#     # base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
#     base_model.fc = nn.Sequential(
#         nn.Linear(base_model.fc.in_features, 128),
#         nn.ReLU(),
#         nn.Linear(128, 2)
#     )
#     self.model = base_model

#   def forward(self, x):
#     return self.model(x)




def prepare_data(dataset):
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, [train_size, test_size])



def train_epoch(epoch, model, loader, loss_func, optimizer, scheduler, device):
  model.train()
  running_loss = 0.0
  epoch_loss = 0.0 
  reporting_step = 60

  for i , (images, labels) in enumerate(loader):
    images = images.to(device)
    labels  = labels.to(device)

    out_put = model(images)
    loss = loss_func(out_put, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    
    scheduler.step()

    running_loss += loss.item()
    epoch_loss += loss.item() 

    if i % reporting_step == reporting_step - 1:
      current_lr = optimizer.param_groups[0]['lr']
      print(f"Epoch: {epoch} step: {i}  ave_loss: {running_loss/reporting_step:.4f} LR: {current_lr:.6f}")
      running_loss = 0.0

  return epoch_loss / len(loader)



def test_epoch(epoch, model, loader, device):

  y_pred = []
  y_true = []

  with torch.no_grad():
    model.eval()
    for i , (images, labels) in enumerate(loader):
      images = images.to(device)
      labels = labels.to(device)

      output = model(images)

      output = output.cpu().numpy()
      label = labels.cpu().numpy()


      y_pred +=list(output)
      y_true +=list(label)

    return y_pred, y_true



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    csv_file = '/content/drive/MyDrive/train_resnet/labels1.csv'
    image_dir = '/content/drive/MyDrive/train_resnet/images'

    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
    ])

    dataset = VehicleDataset(csv_file, image_dir, transform)
    train_dataset, test_dataset = prepare_data(dataset)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    best_mae = float('inf')
    model = VehicleModel().to(device)
    loss_func = nn.SmoothL1Loss()
    # optimizer = optim.SGD(model.parameters(), lr = 0.005, momentum = 0.9, weight_decay = 5e-4) # resnet
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)



    num_epochs = 45
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * num_epochs)

    history = {
        'train_loss': [],
        'val_mse': [],
        'val_mae': [],
        'mae_oto': [],
        'mae_xe_may': []
    }

    for epoch in range(num_epochs):
      avg_train_loss = train_epoch(epoch, model, train_loader, loss_func, optimizer, scheduler, device)
      y_pred, y_true = test_epoch(epoch, model, test_loader, device)

      y_true_np = np.array(y_true)
      y_pred_np = np.array(y_pred)

      
      if np.isnan(y_pred_np).any():
          idx = np.where(np.isnan(y_pred_np))
          print("NaN found at indices:", idx)

      mse = mean_squared_error(y_true_np, y_pred_np)
      mae = mean_absolute_error(y_true_np, y_pred_np)

      mae_oto = mean_absolute_error(y_true_np[:, 0], y_pred_np[:, 0])
      mae_xe_may = mean_absolute_error(y_true_np[:, 1], y_pred_np[:, 1])

      print(f"Epoch {epoch} Evaluation:")
      print(f"  - Train Loss: {avg_train_loss:.4f}")
      print(f"  - MSE Tổng thể: {mse:.2f}")
      print(f"  - MAE Tổng thể: {mae:.2f} (Ô tô: {mae_oto:.2f}, Xe máy: {mae_xe_may:.2f})")

      history['train_loss'].append(avg_train_loss)
      history['val_mse'].append(mse)
      history['val_mae'].append(mae)
      history['mae_oto'].append(mae_oto)
      history['mae_xe_may'].append(mae_xe_may)

      if mae < best_mae:
            best_mae = mae
            save_path = f"best_model_resnet50_noscale_mae_{mae:.2f}.pth"
            torch.save(model.state_dict(), save_path)
            print(f" Saved best model (MAE={mae:.2f}) at epoch {epoch+1} -> {save_path}")


    print("\nTraining finished. Plotting results...")
    epochs_range = range(1, num_epochs + 1)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', color='red')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history['val_mse'], label='Validation MSE', color='blue')
    plt.title('Mean Squared Error (MSE)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history['val_mae'], label='Total MAE', color='green', linewidth=2)
    plt.plot(epochs_range, history['mae_oto'], label='MAE Oto', color='orange', linestyle='--')
    plt.plot(epochs_range, history['mae_xe_may'], label='MAE Xe May', color='purple', linestyle='--')
    plt.title('Mean Absolute Error (MAE)')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_metrics.png') # Lưu ảnh
    plt.show() 

main()