import csv
import os
import time
import numpy as np
from pathlib import Path

import mlflow
from ultralytics import YOLO

from src.utils.logger import get_logger

logger = get_logger("benchmark")

# ============================================================
# CONFIG + DATASET AUTO-DETECTION
# ============================================================


def resolve_dataset_images():
    """
    Automatically find COCO128 train images.
    Checks both repo-local and /workspaces/ paths.
    """
    candidates = [
        "datasets/coco128/images/train2017",
        "/workspaces/datasets/coco128/images/train2017",
    ]

    for c in candidates:
        p = Path(c)
        if p.exists() and any(p.glob("*.jpg")):
            print(f"[benchmark] Using dataset images → {p}")
            return p

    raise FileNotFoundError(
        "Could not find COCO128 train images. Checked:\n"
        + "\n".join([f"  - {x}" for x in candidates])
    )


VAL_IMAGES = resolve_dataset_images()

# ============================================================
# MODEL PATHS
# ============================================================

MODELS = {
    "teacher": "runs/teacher/yolov8s_coco128/weights/best.pt",
    "student": "runs/student/yolov8n_coco128_student/weights/best.pt",
    "student_kd": "runs/student/yolov8n_coco128_student_KD/weights/best.pt",
}

N = 10  # number of inference runs
CSV_OUT = "benchmark_results.csv"

os.environ.setdefault("MLFLOW_TRACKING_URI", "file:./mlruns")

# ============================================================
# BENCHMARKING
# ============================================================


def benchmark_model(name, weights):
    """Run warm-up + N timed inference runs for one model."""

    logger.info(f"Benchmarking {name} → {weights}")

    model = YOLO(weights)

    # Model size in MB
    model_size = Path(weights).stat().st_size / (1024 * 1024)

    # Pick sample image
    sample_image = next(Path(VAL_IMAGES).glob("*.jpg"))
    print(f"[benchmark] Running on image → {sample_image}")
    img_bytes = sample_image.read_bytes()

    # Convert bytes -> OpenCV image array
    import cv2

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Warm-up
    for _ in range(2):
        model.predict(img, imgsz=320, verbose=False)

    # Timed inference runs
    latencies = []
    for _ in range(N):
        t0 = time.time()
        model.predict(img, imgsz=320, verbose=False)
        latencies.append(time.time() - t0)

    avg = float(np.mean(latencies))
    p95 = float(np.percentile(latencies, 95))

    return {
        "model": name,
        "weights": weights,
        "model_size_mb": model_size,
        "avg_latency_ms": avg * 1000,
        "p95_latency_ms": p95 * 1000,
    }


# ============================================================
# CSV EXPORT
# ============================================================


def save_csv(results):
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Saved CSV → {CSV_OUT}")


# ============================================================
# MLflow Logging
# ============================================================


def log_to_mlflow(metrics, model_name):
    mlflow.set_experiment("benchmark_experiment")

    with mlflow.start_run(run_name=f"benchmark_{model_name}"):
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)
            else:
                mlflow.log_param(k, v)


# ============================================================
# MAIN
# ============================================================


def main():
    results = []

    for name, weights in MODELS.items():
        metrics = benchmark_model(name, weights)
        results.append(metrics)
        log_to_mlflow(metrics, name)

    save_csv(results)


if __name__ == "__main__":
    main()
