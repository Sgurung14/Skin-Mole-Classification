# src/app.py
import io
import os

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image

from src.modeling import build_classifier


APP_MODEL_PATH = os.getenv("MODEL_PATH", "models/model.weights.h5")
APP_IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
APP_THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

app = FastAPI(title="Skin Mole Classifier", version="1.0.0")

_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(APP_MODEL_PATH):
            raise RuntimeError(f"Model not found at {APP_MODEL_PATH}")
        _model = build_classifier(image_size=(APP_IMG_SIZE, APP_IMG_SIZE), base_weights=None)
        _model.load_weights(APP_MODEL_PATH)
    return _model


def preprocess_pil(img: Image.Image, img_size: int):
    img = img.convert("RGB")
    img = img.resize((img_size, img_size))
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.get("/health")
def health():
    return {"status": "ok", "model_path": APP_MODEL_PATH}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    x = preprocess_pil(img, APP_IMG_SIZE)
    model = get_model()
    prob = float(model.predict(x, verbose=0).reshape(-1)[0])
    pred = int(prob >= APP_THRESHOLD)

    return {
        "prob_malignant": prob,
        "pred_label": "malignant" if pred == 1 else "benign",
        "threshold": APP_THRESHOLD,
    }
