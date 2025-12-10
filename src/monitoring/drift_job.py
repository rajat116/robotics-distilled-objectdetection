import os
import json
from datetime import datetime
import numpy as np
import cv2

# Path to dataset ON EC2
DATA_FOLDER = (
    "/home/ubuntu/robotics-distilled-objectdetection/datasets/coco128/images/train2017"
)


def run_drift_job():
    """Compute simple drift metrics on EC2 dataset."""
    if not os.path.exists(DATA_FOLDER):
        raise Exception(f"❌ Dataset path not found on EC2: {DATA_FOLDER}")

    pixels = []

    for fname in os.listdir(DATA_FOLDER):
        if not fname.endswith(".jpg"):
            continue
        img = cv2.imread(os.path.join(DATA_FOLDER, fname))
        if img is None:
            continue
        pixels.append(img.mean())

    if len(pixels) == 0:
        raise Exception("❌ No valid images found for drift computation!")

    current_mean = float(np.mean(pixels))

    # Baseline stats (example)
    baseline_mean = 110.0
    baseline_std = 55.0

    drift_score = abs(current_mean - baseline_mean) / baseline_std
    drift_score = min(1.0, drift_score)

    # Save to EC2 reports directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_dir = (
        f"/home/ubuntu/robotics-distilled-objectdetection/reports/drift/{timestamp}"
    )
    os.makedirs(out_dir, exist_ok=True)

    summary = {
        "drift_score": drift_score,
        "current_mean": current_mean,
        "baseline_mean": baseline_mean,
        "timestamp": timestamp,
    }

    json_path = os.path.join(out_dir, "report.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"📊 Drift score = {drift_score}")
    print(f"JSON saved → {json_path}")

    return drift_score


if __name__ == "__main__":
    run_drift_job()
