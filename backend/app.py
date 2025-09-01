import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io, torch, numpy as np, cv2
from ultralytics import YOLO
import torchvision as tv, torch.nn as nn
from backend.utils.spray_rules import get_recommendations
from backend.utils.boxes_model import counts_to_boxes, YieldConfig

app = FastAPI(title="Pear Precision API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load models (place trained .pt files in backend/models)
try:
    yolo_model = YOLO("backend/models/yolo_pear.pt")
except Exception:
    yolo_model = None

clf = None
leaf_classes = []
memory_clf = None

# Try memory classifier first (most reliable for small dataset)
try:
    import pickle
    with open("backend/models/memory_classifier.pkl", "rb") as f:
        memory_model_data = pickle.load(f)
    memory_clf = memory_model_data['classifier']
    leaf_classes = memory_model_data['classes']
    print("Loaded memory classifier successfully")
except Exception as e:
    print(f"Could not load memory classifier: {e}")

# Fallback to simple classifier
if memory_clf is None:
    try:
        with open("backend/models/simple_classifier.pkl", "rb") as f:
            simple_model_data = pickle.load(f)
        memory_clf = simple_model_data['classifier']
        leaf_classes = simple_model_data['classes']
        print("Loaded simple classifier as fallback")
    except Exception as e:
        print(f"Could not load simple classifier: {e}")

# Fallback to neural network
if memory_clf is None:
    try:
        ckpt = torch.load("backend/models/leaf_classifier_enhanced.pt", map_location="cpu")
        leaf_classes = ckpt["classes"]
        clf = tv.models.resnet18(weights=None)
        clf.fc = nn.Linear(clf.fc.in_features, len(leaf_classes))
        clf.load_state_dict(ckpt["model"])
        clf.eval()
        print("Loaded neural network classifier as fallback")
    except Exception as e:
        print(f"Could not load enhanced classifier: {e}")
        # Final fallback to original model
        try:
            ckpt = torch.load("backend/models/leaf_classifier.pt", map_location="cpu")
            leaf_classes = ckpt["classes"]
            clf = tv.models.resnet18(weights=None)
            clf.fc = nn.Linear(clf.fc.in_features, len(leaf_classes))
            clf.load_state_dict(ckpt["model"])
            clf.eval()
            print("Loaded original neural network classifier")
        except Exception as e:
            print(f"Could not load any classifier: {e}")

def pil_from_upload_bytes(b: bytes)->Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

@app.post("/api/count")
async def api_count(file: UploadFile = File(...), conf: float = Form(0.25)):
    if yolo_model is None:
        return {"error": "YOLO model not loaded. Train it first and place backend/models/yolo_pear.pt"}
    img_bytes = await file.read()

    res = None
    img_bytes_io = io.BytesIO(img_bytes)
    try:
        # Try passing bytes directly
        res = yolo_model.predict(source=img_bytes_io, imgsz=640, conf=conf, verbose=False)[0]
    except TypeError:
        # Convert bytes to numpy array and then to BGR image for YOLO
        img_np = np.frombuffer(img_bytes, np.uint8)
        img_cv = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        res = yolo_model.predict(source=img_cv, imgsz=640, conf=conf, verbose=False)[0]

    n = int((res.boxes.cls==0).sum().item())
    plotted = res.plot()  # BGR numpy
    ok, buff = cv2.imencode(".jpg", plotted)
    if not ok:
        return {"error": "Failed to encode result image.", "count": n}
    return {"count": n, "image_b64": buff.tobytes().hex()}

def extract_features(image_path):
    """Extract simple features from image"""
    import cv2
    img = cv2.imread(str(image_path))
    img = cv2.resize(img, (256, 256))

    # Color histogram features
    hist_features = []
    for channel in range(3):  # RGB channels
        hist = cv2.calcHist([img], [channel], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.extend(hist)

    # Texture features (simple statistics)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    texture_features = [
        gray.mean(),      # Mean intensity
        gray.std(),       # Standard deviation
        gray.var(),       # Variance
        cv2.Laplacian(gray, cv2.CV_64F).var()  # Laplacian variance (texture)
    ]

    # Shape features
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        shape_features = [area, perimeter, area/(perimeter+1)]  # Compactness
    else:
        shape_features = [0, 0, 0]

    return np.array(hist_features + texture_features + shape_features)

@app.post("/api/disease")
async def api_disease(file: UploadFile = File(...)):
    if memory_clf is None and clf is None:
        return {"error": "No leaf classifier loaded. Train models first."}

    img_bytes = await file.read()

    # Try memory classifier first (most reliable for small dataset)
    if memory_clf is not None:
        try:
            # Save temp image for feature extraction
            import tempfile
            import os
            import cv2

            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                tmp_file.write(img_bytes)
                tmp_path = tmp_file.name

            # Extract features (same as training)
            img = cv2.imread(str(tmp_path))
            if img is not None:
                img = cv2.resize(img, (64, 64))

                features = []
                features.extend(img.mean(axis=(0,1)))  # Mean RGB
                features.extend(img.std(axis=(0,1)))   # Std RGB
                features.extend(img.flatten()[:100])   # First 100 pixels

                features = np.array(features).reshape(1, -1)
                probs = memory_clf.predict_proba(features)[0]
                idx = int(np.argmax(probs))

                # Clean up temp file
                os.unlink(tmp_path)

                return {"top_class": leaf_classes[idx], "probs": dict(zip(leaf_classes, [float(p) for p in probs]))}
        except Exception as e:
            print(f"Memory classifier failed: {e}")

    # Fallback to neural network
    if clf is not None:
        img = pil_from_upload_bytes(img_bytes).resize((256,256))
        x = tv.transforms.ToTensor()(img).unsqueeze(0)
        with torch.no_grad():
            logits = clf(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).numpy().tolist()
            idx = int(np.argmax(probs))
        return {"top_class": leaf_classes[idx], "probs": dict(zip(leaf_classes, [float(p) for p in probs]))}

    return {"error": "All classifiers failed"}

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
