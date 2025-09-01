import torch, torchvision as tv, torch.nn as nn, torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
data_dir = ROOT/'data'/'test-leaf-dataset'
assert data_dir.exists(), "test-leaf-dataset not found in data/"

tf_train = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(12),
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor()
])
tf_val = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

full_ds = tv.datasets.ImageFolder(root=str(data_dir), transform=None)
num_classes = len(full_ds.classes)
print(f"Found {num_classes} classes: {full_ds.classes}")
print(f"Total images: {len(full_ds)}")

# Split 85/15 if no separate train/val provided
n = len(full_ds)
val_size = int(0.15 * n)
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

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = tv.models.resnet18(weights=tv.models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

best_acc, best_path = 0.0, ROOT/'backend'/'models'/'leaf_classifier_test.pt'
for epoch in range(5):  # Reduced epochs for testing
    model.train()
    for x,y in train_loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward(); optimizer.step()

    # quick val
    model.eval(); correct=total=0
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item(); total += y.numel()
    acc = correct/total if total else 0
    print(f"epoch {epoch}: val acc={acc:.3f}")
    if acc>best_acc:
        best_acc=acc
        best_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model':model.state_dict(),'classes':full_ds.classes}, best_path)
print(f"Saved test classifier to {best_path} with acc {best_acc:.3f}")
