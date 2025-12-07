# tests/test_api.py
import cv2
import numpy as np
from fastapi.testclient import TestClient

from src.inference.api import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_dummy():
    # Create a 10×10 black RGB image using numpy + OpenCV
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    success, buf = cv2.imencode(".jpg", img)
    assert success

    files = {"file": ("test.jpg", buf.tobytes(), "image/jpeg")}

    r = client.post("/predict", files=files)

    assert r.status_code == 200
    data = r.json()
    # From your api.py: returns latency_ms and num_detections
    assert "latency_ms" in data
    assert "num_detections" in data
