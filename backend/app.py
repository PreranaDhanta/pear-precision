from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io, torch, numpy as np, cv2
from ultralytics import YOLO
import torchvision as tv, torch.nn as nn
from utils.spray_rules import get_recommendations
from utils.boxes_model import counts_to_boxes, YieldConfig

app = FastAPI(title="Pear Precision API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load models (place trained .pt files in backend/models)
try:
    yolo_model = YOLO("backend/models/yolo_pear.pt")
except Exception:
    yolo_model = None

clf = None
leaf_classes = []
try:
    ckpt = torch.load("backend/models/leaf_classifier.pt", map_location="cpu")
    leaf_classes = ckpt["classes"]
    clf = tv.models.resnet18(weights=None)
    clf.fc = nn.Linear(clf.fc.in_features, len(leaf_classes))
    clf.load_state_dict(ckpt["model"])
    clf.eval()
except Exception:
    pass

def pil_from_upload_bytes(b: bytes)->Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

@app.post("/api/count")
async def api_count(file: UploadFile = File(...), conf: float = Form(0.25)):
    if yolo_model is None:
        return {"error": "YOLO model not loaded. Train it first and place backend/models/yolo_pear.pt"}
    img_bytes = await file.read()
    res = yolo_model.predict(source=io.BytesIO(img_bytes), imgsz=640, conf=conf, verbose=False)[0]
    n = int((res.boxes.cls==0).sum().item())
    plotted = res.plot()  # BGR numpy
    ok, buff = cv2.imencode(".jpg", plotted)
    if not ok:
        return {"error": "Failed to encode result image.", "count": n}
    return {"count": n, "image_b64": buff.tobytes().hex()}

@app.post("/api/disease")
async def api_disease(file: UploadFile = File(...)):
    if clf is None:
        return {"error": "Leaf classifier not loaded. Train it first and place backend/models/leaf_classifier.pt"}
    img_bytes = await file.read()
    img = pil_from_upload_bytes(img_bytes).resize((256,256))
    x = tv.transforms.ToTensor()(img).unsqueeze(0)
    with torch.no_grad():
        logits = clf(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy().tolist()
        idx = int(np.argmax(probs))
    return {"top_class": leaf_classes[idx], "probs": dict(zip(leaf_classes, [float(p) for p in probs]))}

class BoxesReq(BaseModel):
    detected_count: int
    pears_per_box: float = 100.0
    detection_recall: float = 0.9

@app.post("/api/boxes")
async def api_boxes(req: BoxesReq):
    boxes = counts_to_boxes(req.detected_count, YieldConfig(req.pears_per_box, req.detection_recall))
    return {"estimated_boxes": round(boxes, 2)}

@app.get("/api/spray")
async def api_spray(stage: str = "all"):
    return {"recommendations": get_recommendations(stage)}

@app.get("/api/health")
async def api_health():
    return {"ok": True}
