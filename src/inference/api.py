import time

import cv2
import numpy as np
from fastapi import FastAPI, File, Response, UploadFile
from .model_loader import load_model, load_shadow_model

from src.monitoring.metrics import (
    LATENCY_HISTOGRAM,  # <-- FIXED IMPORT
    REQUEST_COUNTER,
)

app = FastAPI()
predict_primary = load_model()
predict_shadow = load_shadow_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # start = time.time()

    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # PRIMARY MODEL
    t0 = time.time()
    primary_preds = predict_primary(img)
    primary_latency = time.time() - t0
    num_primary = len(primary_preds[0].boxes)

    result = {
        # OLD field names (required for CI tests)
        "latency_ms": primary_latency * 1000,
        "num_detections": num_primary,
        # NEW field names (your updated API)
        "latency_primary_ms": primary_latency * 1000,
        "num_detections_primary": num_primary,
    }

    # SHADOW MODEL
    if predict_shadow is not None:
        t1 = time.time()
        shadow_preds = predict_shadow(img)
        shadow_latency = time.time() - t1
        num_shadow = len(shadow_preds[0].boxes)

        # Simple IoU comparison (Option A)
        iou_mean = None
        try:

            b1 = primary_preds[0].boxes.xyxy.cpu().numpy()
            b2 = shadow_preds[0].boxes.xyxy.cpu().numpy()

            if len(b1) > 0 and len(b2) > 0:
                from shapely.geometry import Polygon

                ious = []
                for p in b1:
                    poly1 = Polygon(
                        [(p[0], p[1]), (p[2], p[1]), (p[2], p[3]), (p[0], p[3])]
                    )
                    best_iou = 0
                    for q in b2:
                        poly2 = Polygon(
                            [(q[0], q[1]), (q[2], q[1]), (q[2], q[3]), (q[0], q[3])]
                        )
                        best_iou = max(
                            best_iou,
                            poly1.intersection(poly2).area / poly1.union(poly2).area,
                        )
                    ious.append(best_iou)
                iou_mean = float(np.mean(ious))
        except Exception:
            iou_mean = None

        # Add to returned result
        result.update(
            {
                "latency_shadow_ms": shadow_latency * 1000,
                "num_detections_shadow": num_shadow,
                "iou_mean": iou_mean,
            }
        )

    REQUEST_COUNTER.inc()
    LATENCY_HISTOGRAM.observe(primary_latency)

    return result


@app.get("/metrics")
def metrics():
    from prometheus_client import generate_latest

    return Response(generate_latest(), media_type="text/plain")
