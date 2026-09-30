import os
import io
import time
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_model_path() -> str:
    candidates = [
        os.getenv("MODEL_PATH"),
        os.path.join(SCRIPT_DIR, "cifake_model.keras"),
        "cifake_model.keras",
        os.path.join(SCRIPT_DIR, "..", "..", "cifake_model.keras")
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return os.path.abspath(p)
    return os.path.join(SCRIPT_DIR, "cifake_model.keras")

MODEL_PATH = resolve_model_path()
IMG_SIZE = 96
_model = None

def get_model():
    global _model
    if _model is not None:
        return _model

    if os.path.exists(MODEL_PATH):
        try:
            import keras
            _model = keras.models.load_model(MODEL_PATH)
            return _model
        except Exception:
            pass

        try:
            import tensorflow as tf
            _model = tf.keras.models.load_model(MODEL_PATH)
            return _model
        except Exception as e:
            print(f"[ERROR] Gagal memuat model dari {MODEL_PATH}: {e}")
            return None
    return None

def preprocess_image(image_bytes: bytes, target_size: int = IMG_SIZE) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Berkas citra tidak valid atau rusak: {e}")
    
    img = img.resize((target_size, target_size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)

def predict_image(image_bytes: bytes):
    t0 = time.time()
    input_tensor = preprocess_image(image_bytes)
    model = get_model()

    if model is not None:
        raw_score = float(model.predict(input_tensor, verbose=0)[0][0])
    else:
        hash_val = sum(image_bytes[:50]) % 100
        raw_score = 0.85 if hash_val > 45 else 0.15

    latency_ms = round((time.time() - t0) * 1000, 2)
    is_real = raw_score >= 0.5
    label = "REAL" if is_real else "FAKE"
    confidence = raw_score if is_real else (1.0 - raw_score)

    return {
        "label": label,
        "class_id": 1 if is_real else 0,
        "confidence": round(float(confidence), 4),
        "raw_score": round(float(raw_score), 4),
        "is_simulated": model is None,
        "inference_time_ms": latency_ms
    }
