from ultralytics import YOLO
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
weights = ROOT/'backend'/'models'/'yolo_pear.pt'
imgs_dir = ROOT/'data'/'pear640'/'images'/'val'  # adjust as needed

model = YOLO(str(weights))
rows=[]
for img in imgs_dir.rglob("*.jpg"):
    res = model.predict(source=str(img), imgsz=640, conf=0.25, verbose=False)[0]
    n = int((res.boxes.cls==0).sum().item())
    rows.append({'image': img.name, 'count': n})
out = ROOT/'data'/'counts_val.json'
out.write_text(json.dumps(rows, indent=2))
print(f"Wrote {out}")
