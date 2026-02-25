# src/app.py
import io
import os

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.modeling import build_classifier, preprocess_input_array


APP_MODEL_PATH = os.getenv("MODEL_PATH", "models/model.weights.h5")
APP_MODEL_LOAD_MODE = os.getenv("MODEL_LOAD_MODE", "auto").strip().lower()
APP_IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
APP_THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
APP_MODEL_BACKBONE = os.getenv("MODEL_BACKBONE", "efficientnetb0")
APP_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]
APP_ALLOWED_ORIGIN_REGEX = os.getenv("ALLOWED_ORIGIN_REGEX") or None

app = FastAPI(title="Skin Mole Classifier", version="1.0.0")

# allows apps on different ports/domains to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=APP_ALLOWED_ORIGINS,
    allow_origin_regex=APP_ALLOWED_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


def resolve_model_load_mode(model_path: str, mode: str) -> str:
    value = (mode or "auto").strip().lower()
    if value not in {"auto", "weights", "full"}:
        raise RuntimeError("MODEL_LOAD_MODE must be one of: auto, weights, full")
    if value != "auto":
        return value

    path_lower = model_path.lower()
    if path_lower.endswith(".keras"):
        return "full"
    if path_lower.endswith(".weights.h5"):
        return "weights"
    if path_lower.endswith(".h5"):
        # Treat generic .h5 as a full model by default.
        return "full"
    return "weights"


# check if model is loaded already, if not load it and return it.
# This way we only load the model once when the app starts, and subsequent requests will use
# the already loaded model, improving performance.
def get_model():
    global _model
    if _model is None:
        if not os.path.exists(APP_MODEL_PATH):
            raise RuntimeError(f"Model not found at {APP_MODEL_PATH}")

        load_mode = resolve_model_load_mode(APP_MODEL_PATH, APP_MODEL_LOAD_MODE)
        if load_mode == "full":
            _model = tf.keras.models.load_model(APP_MODEL_PATH, compile=False)
        else:
            _model = build_classifier(
                image_size=(APP_IMG_SIZE, APP_IMG_SIZE),
                base_weights=None,
                backbone=APP_MODEL_BACKBONE,
            )
            _model.load_weights(APP_MODEL_PATH)
    return _model


def preprocess_pil(img: Image.Image, img_size: int):
    img = img.convert("RGB")
    img = img.resize((img_size, img_size))
    arr = np.asarray(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)

    load_mode = resolve_model_load_mode(APP_MODEL_PATH, APP_MODEL_LOAD_MODE)
    if load_mode == "full":
        # Full saved models may already include preprocessing layers/ops in the graph.
        return arr

    return preprocess_input_array(arr, backbone=APP_MODEL_BACKBONE)


# check server is running
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": APP_MODEL_PATH,
        "model_load_mode": resolve_model_load_mode(APP_MODEL_PATH, APP_MODEL_LOAD_MODE),
        "backbone": APP_MODEL_BACKBONE,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # reject non-image files
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    x = preprocess_pil(img, APP_IMG_SIZE)
    try:
        model = get_model()
    except RuntimeError as exc:
        # Surface deployment/runtime model issues clearly to API clients.
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model could not be loaded: {exc}")
    prob = float(model.predict(x, verbose=0).reshape(-1)[0])
    pred = int(prob >= APP_THRESHOLD)

    return {
        "prob_malignant": prob,
        "pred_label": "malignant" if pred == 1 else "benign",
        "threshold": APP_THRESHOLD,
    }
