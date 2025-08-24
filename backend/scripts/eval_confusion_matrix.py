import torch, torchvision as tv
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ckpt_path = ROOT/'backend'/'models'/'leaf_classifier.pt'
assert ckpt_path.exists(), "leaf_classifier.pt not found. Train the classifier first."

ckpt = torch.load(ckpt_path, map_location='cpu')
classes = ckpt['classes']

data_dir = ROOT/'data'/'pear-leaf-disease'
tf = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
ds = tv.datasets.ImageFolder(str(data_dir), transform=tf)
loader = DataLoader(ds, batch_size=32, shuffle=False)

import torchvision.models as M, torch.nn as nn
model = M.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(ckpt['model']); model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for x,y in loader:
        logits = model(x)
        y_true += y.tolist()
        y_pred += logits.argmax(1).tolist()

print(classification_report(y_true, y_pred, target_names=classes))
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(xticks_rotation=45)
plt.tight_layout()
out_path = ROOT/'frontend'/'confusion_matrix.png'
plt.savefig(out_path, dpi=150)
print(f"Saved confusion matrix to {out_path}")
