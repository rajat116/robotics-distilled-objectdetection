import time

import cv2
import numpy as np
from fastapi import FastAPI, File, Response, UploadFile

from src.monitoring.metrics import (
    LATENCY_HISTOGRAM,  # <-- FIXED IMPORT
    REQUEST_COUNTER,
)

from .model_loader import load_model

app = FastAPI()
predict_fn = load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start = time.time()

    image_bytes = await file.read()

    # convert bytes → numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    preds = predict_fn(img)

    latency = time.time() - start
    REQUEST_COUNTER.inc()
    LATENCY_HISTOGRAM.observe(latency)

    return {
        "latency_ms": latency * 1000,
        "num_detections": len(preds[0].boxes),
    }


@app.get("/metrics")
def metrics():
    from prometheus_client import generate_latest

    return Response(generate_latest(), media_type="text/plain")
