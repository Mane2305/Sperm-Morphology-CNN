"""
SpermAI — Clinical Morphology Analysis API
FastAPI backend with single/batch prediction, session logging, and Grad-CAM support
"""

import io
import os
import sys
import uuid
import json
import logging
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import get_model

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH  = r"C:\Users\admin\Desktop\sperm-morphology-cnn\models\fold_5_best.pth"
MODEL_TYPE  = "efficientnet_b0"
NUM_CLASSES = 3
DROPOUT     = 0.5
CLASS_NAMES = ["Abnormal", "Non-Sperm", "Normal"]
IMAGE_SIZE  = 224
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD   = 0.70

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
SESSION_LOG = LOG_DIR / "sessions.jsonl"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spermai")

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
model = None

def load_model():
    global model
    m = get_model(model_type=MODEL_TYPE, num_classes=NUM_CLASSES, pretrained=False, dropout=DROPOUT)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    m.load_state_dict(state_dict)
    m.to(DEVICE)
    m.eval()
    model = m
    logger.info(f"✅ Model loaded — {MODEL_TYPE} on {DEVICE}")

transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # type: ignore
    ToTensorV2(),
])

# ─────────────────────────────────────────────
# APP LIFESPAN
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(
    title="SpermAI Clinical API",
    version="2.0.0",
    description="Sperm morphology classification — Research Use Only",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def read_image(contents: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        return np.array(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {e}")


def run_inference(img_np: np.ndarray) -> dict:
    tensor = transform(image=img_np)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]  # type: ignore
    conf, idx = torch.max(probs, dim=0)
    prediction = CLASS_NAMES[idx.item()]  # type: ignore
    confidence = round(float(conf), 4)
    probabilities = {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)}
    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "flagged_for_review": confidence < THRESHOLD,
        "morphology_index": round(probabilities.get("Normal", 0) * 100, 1),  # % normal
    }


def log_session(session_id: str, entries: list):
    record = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "count": len(entries),
        "entries": entries,
    }
    with open(SESSION_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "service": "SpermAI Clinical API", "version": "2.0.0"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": MODEL_TYPE,
        "device": DEVICE,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "threshold": THRESHOLD,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/predict")
async def predict(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: Optional[str] = None,
):
    """Single image prediction"""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    img_np = read_image(contents)

    t0 = datetime.now()
    result = run_inference(img_np)
    ms = round((datetime.now() - t0).total_seconds() * 1000, 1)

    result["processing_time_ms"] = ms
    result["filename"] = file.filename
    result["image_size"] = list(img_np.shape[:2])
    result["session_id"] = session_id or str(uuid.uuid4())[:8]
    result["timestamp"] = datetime.now().isoformat()

    if session_id:
        background_tasks.add_task(log_session, session_id, [result])

    return result


@app.post("/batch-predict")
async def batch_predict(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    """Batch image prediction — up to 50 images"""
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 images per batch")

    session_id = str(uuid.uuid4())[:8]
    predictions = []
    summary = {c: 0 for c in CLASS_NAMES}
    flagged_count = 0

    for i, file in enumerate(files):
        try:
            contents = await file.read()
            img_np = read_image(contents)
            t0 = datetime.now()
            result = run_inference(img_np)
            ms = round((datetime.now() - t0).total_seconds() * 1000, 1)

            result["filename"] = file.filename
            result["index"] = i + 1
            result["processing_time_ms"] = ms
            summary[result["prediction"]] += 1
            if result["flagged_for_review"]:
                flagged_count += 1
            predictions.append(result)

        except Exception as e:
            predictions.append({
                "filename": file.filename,
                "index": i + 1,
                "error": str(e),
            })

    total = len([p for p in predictions if "prediction" in p])
    avg_confidence = (
        sum(p.get("confidence", 0) for p in predictions if "confidence" in p) / max(total, 1)
    )
    normal_rate = round((summary.get("Normal", 0) / max(total, 1)) * 100, 1)

    batch_result = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "total_processed": total,
        "total_submitted": len(files),
        "predictions": predictions,
        "summary": summary,
        "analytics": {
            "average_confidence": round(avg_confidence, 4),
            "flagged_count": flagged_count,
            "normal_rate_percent": normal_rate,
            "abnormal_rate_percent": round(
                (summary.get("Abnormal", 0) / max(total, 1)) * 100, 1
            ),
            "non_sperm_rate_percent": round(
                (summary.get("Non-Sperm", 0) / max(total, 1)) * 100, 1
            ),
        },
    }

    background_tasks.add_task(log_session, session_id, predictions)
    return batch_result


@app.get("/sessions")
def list_sessions(limit: int = 20):
    """Return recent analysis sessions"""
    if not SESSION_LOG.exists():
        return {"sessions": []}
    sessions = []
    with open(SESSION_LOG) as f:
        lines = f.readlines()[-limit:]
        for line in lines:
            try:
                sessions.append(json.loads(line))
            except Exception:
                pass
    return {"sessions": list(reversed(sessions))}


@app.get("/stats")
def overall_stats():
    """Aggregate statistics across all sessions"""
    if not SESSION_LOG.exists():
        return {"message": "No sessions yet"}
    total_images = 0
    class_counts = {c: 0 for c in CLASS_NAMES}
    session_count = 0
    with open(SESSION_LOG) as f:
        for line in f:
            try:
                s = json.loads(line)
                session_count += 1
                for entry in s.get("entries", []):
                    if "prediction" in entry:
                        total_images += 1
                        class_counts[entry["prediction"]] = class_counts.get(entry["prediction"], 0) + 1
            except Exception:
                pass
    return {
        "total_sessions": session_count,
        "total_images_analyzed": total_images,
        "class_distribution": class_counts,
        "normal_rate": round(class_counts.get("Normal", 0) / max(total_images, 1) * 100, 1),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)