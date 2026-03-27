import os
import random
import pickle

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict

import numpy as np

from utils.helper import ModelManager, save_heatmap, save_image, clear_temp_storage
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration — all paths injectable via env vars for Helm/OpenShift
# ---------------------------------------------------------------------------

CKPT_PATH       = os.getenv("CKPT_PATH",       "/app/outputs/checkpoints/best_cxr_multilabel_3.pt")
VAL_LOADER_PATH = os.getenv("VAL_LOADER_PATH", "/app/outputs/val_samples.pkl")
TEMP_DIR        = os.getenv("TEMP_DIR",         "/app/static/heatmaps")
AMP_ENABLED     = os.getenv("AMP_ENABLED", "true").lower() == "true"

os.makedirs(TEMP_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="MONAI Inference Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file mounts
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

VAL_SAMPLES_DIR = os.path.join(PROJECT_ROOT, "outputs", "val_samples")
if os.path.exists(VAL_SAMPLES_DIR):
    app.mount("/images", StaticFiles(directory=VAL_SAMPLES_DIR), name="val_samples")
else:
    print(f"Warning: {VAL_SAMPLES_DIR} not found.")

STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------------------------------------------------------------------------
# Global state — initialised once at startup
# ---------------------------------------------------------------------------

manager: ModelManager | None = None
val_items: list = []


@app.on_event("startup")
async def startup_event():
    global manager, val_items

    # Load model once — includes CUDA warm-up before first real request
    manager = ModelManager(ckpt_path=CKPT_PATH, amp=AMP_ENABLED)
    print(
        f"Model loaded on {manager.device} | AMP={manager._amp} | "
        f"classes={manager.labels} | img_size={manager.img_size}"
    )

    # Load validation sample list
    if os.path.exists(VAL_LOADER_PATH):
        with open(VAL_LOADER_PATH, "rb") as f:
            val_items = pickle.load(f)
            if not isinstance(val_items, list):
                print(f"Warning: Expected list in {VAL_LOADER_PATH}, got {type(val_items)}")
    else:
        print(f"Warning: {VAL_LOADER_PATH} not found.")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SampleResponse(BaseModel):
    image_path: str
    labels: List[float]
    class_names: List[str]


class InferRequest(BaseModel):
    image_path: str
    class_name: str


class InferResponse(BaseModel):
    probabilities: Dict[str, float]
    heatmap_url: str
    input_image_url: str
    prediction: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": manager is not None}


@app.get("/classes")
async def get_classes():
    if manager is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {"class_names": manager.labels}


@app.get("/sample", response_model=SampleResponse)
async def get_sample():
    if not val_items:
        raise HTTPException(status_code=500, detail="Validation items not loaded.")
    item = random.choice(val_items)
    lbl = item["label"]
    if hasattr(lbl, "tolist"):
        lbl = lbl.tolist()
    return {"image_path": item["image"], "labels": lbl, "class_names": manager.labels}


@app.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest):
    if manager is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if request.class_name not in manager.label_to_idx:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid class name. Must be one of: {manager.labels}",
        )

    clear_temp_storage(TEMP_DIR)

    abs_image_path = os.path.join(VAL_SAMPLES_DIR, os.path.basename(request.image_path))
    class_idx = manager.label_to_idx[request.class_name]

    try:
        probs, cam, img = manager.infer(abs_image_path, class_idx)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return _build_infer_response(probs, cam, img)


@app.post("/infer_upload", response_model=InferResponse)
async def infer_upload(file: UploadFile = File(...), class_name: str = Form(...)):
    if manager is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if class_name not in manager.label_to_idx:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid class name. Must be one of: {manager.labels}",
        )

    clear_temp_storage(TEMP_DIR)

    upload_path = os.path.join(TEMP_DIR, f"upload_{file.filename}")
    try:
        with open(upload_path, "wb") as buf:
            buf.write(await file.read())

        class_idx = manager.label_to_idx[class_name]
        probs, cam, img = manager.infer(upload_path, class_idx)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return _build_infer_response(probs, cam, img)


# ---------------------------------------------------------------------------
# Shared response builder
# ---------------------------------------------------------------------------

def _build_infer_response(probs: np.ndarray, cam: np.ndarray, img: np.ndarray) -> dict:
    suffix = random.randint(1000, 9999)
    heatmap_filename = f"heatmap_{suffix}.jpg"
    input_filename   = f"input_{suffix}.jpg"

    save_heatmap(img, cam, os.path.join(TEMP_DIR, heatmap_filename))
    save_image(img,         os.path.join(TEMP_DIR, input_filename))

    topk = np.argsort(-probs)[:5]
    prob_dict = {manager.labels[i]: float(probs[i]) for i in topk}
    preds = [manager.labels[i] for i, p in enumerate(probs) if p > 0.5]

    return {
        "probabilities":   prob_dict,
        "heatmap_url":     f"/static/heatmaps/{heatmap_filename}",
        "input_image_url": f"/static/heatmaps/{input_filename}",
        "prediction":      ", ".join(preds) if preds else "No Finding",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
