import torch, torchvision as tv, torch.nn as nn, torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
data_dir = ROOT/'data'/'test-leaf-dataset'
assert data_dir.exists(), "test-leaf-dataset not found in data/"

# Enhanced data augmentation for small dataset
tf_train = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(45),  # Increased rotation range
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tf_val = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_ds = tv.datasets.ImageFolder(root=str(data_dir), transform=None)
num_classes = len(full_ds.classes)
print(f"Found {num_classes} classes: {full_ds.classes}")
print(f"Total images: {len(full_ds)}")

# Split 80/20 for better validation
n = len(full_ds)
val_size = int(0.2 * n)
train_size = n - val_size
train_subset, val_subset = random_split(full_ds, [train_size, val_size])

# Wrap with transforms
class WrapDS(torch.utils.data.Dataset):
    def __init__(self, base, tf): self.base, self.tf = base, tf
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        x = tf_train(x) if self.tf=='train' else tf_val(x)
        return x, y

train_ds = WrapDS(train_subset, 'train')
val_ds   = WrapDS(val_subset, 'val')

# Larger batch size, use 0 workers for Windows compatibility
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = tv.models.resnet18(weights=tv.models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

# Better optimizer and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# Training tracking
train_losses = []
val_losses = []
val_accuracies = []

best_acc, best_path = 0.0, ROOT/'backend'/'models'/'leaf_classifier_enhanced.pt'
for epoch in range(50):  # Increased epochs
    # Training phase
    model.train()
    epoch_train_loss = 0.0
    for x,y in train_loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation phase
    model.eval()
    epoch_val_loss = 0.0
    correct = total = 0
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            epoch_val_loss += loss.item()
            pred = logits.argmax(1)
            correct += (pred==y).sum().item()
            total += y.numel()

    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    acc = correct/total if total else 0
    val_accuracies.append(acc)

    scheduler.step()

    print(".3f")

    if acc > best_acc:
        best_acc = acc
        best_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model':model.state_dict(),'classes':full_ds.classes}, best_path)

print(".3f")

# Save training curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
plt.bar(range(num_classes), [0]*num_classes)  # Placeholder for confusion matrix
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix (Placeholder)')
plt.xticks(range(num_classes), full_ds.classes, rotation=45)

plt.tight_layout()
plt.savefig(ROOT/'backend'/'models'/'training_curves.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Enhanced classifier saved to {best_path}")
print(f"Training curves saved to {ROOT/'backend'/'models'/'training_curves.png'}")
