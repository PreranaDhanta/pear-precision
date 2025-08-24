from ultralytics import YOLO
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
data_yaml = ROOT/'data'/'pear640.yaml'
weights = 'yolov8n.pt'

model = YOLO(weights)
results = model.train(
    data=str(data_yaml),
    epochs=80,
    imgsz=640,
    batch=16,
    lr0=0.01,
    patience=20,
    mosaic=1.0,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4
)

# Find best.pt and copy to backend/models/yolo_pear.pt
runs = ROOT/'runs'/'detect'
best = None
if runs.exists():
    # pick most recent run with weights/best.pt
    candidates = sorted(runs.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        best = candidates[0]

out = ROOT/'backend'/'models'/'yolo_pear.pt'
out.parent.mkdir(parents=True, exist_ok=True)
if best and best.exists():
    shutil.copy(best, out)
    print(f"Saved detector to {out}")
else:
    print("Could not locate best.pt; check runs/ directory.")
