import torch
from torch.utils.data import DataLoader, random_split
from data_augmentation import train_transform, val_transform
from dataset import RareDiseaseDataset
from model import PretrainedModel
from transformers import AutoModel

# Load dataset
dataset = RareDiseaseDataset(root_dir="rare_disease", transform=train_transform, advanced=True)

# Chia train/validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Thay đổi transform cho validation
val_dataset.dataset.transform = val_transform

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Khởi tạo mô hình pretrained
num_classes = len(dataset.classes)
model = PretrainedModel("facebook/deit-base-distilled-patch16-224", num_classes)

# Cấu hình loss function và optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_one_epoch():
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Loss: {loss.item():.4f}")

# Chạy training 1 epoch
train_one_epoch()
