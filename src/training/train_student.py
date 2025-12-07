import os

import mlflow
from ultralytics import YOLO

from src.utils.config import MLFLOW_EXPERIMENT_NAME
from src.utils.logger import get_logger

logger = get_logger("train_student")

# -------------------------
# CONFIG
# -------------------------
STUDENT_MODEL_NAME = "yolov8n.pt"  # small student model
DATA_CONFIG = "coco128.yaml"  # same small dataset as teacher
EPOCHS = 5
IMGSZ = 320
BATCH = 8

# Disable Ultralytics' built-in MLflow; we use our own
os.environ["YOLO_MLFLOW"] = "False"
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"


def train_student():
    """
    Train a small YOLOv8n student model on coco128 and log everything to MLflow.
    This is a clean, minimal baseline (no custom KD internals).
    """

    print(">>> Initializing student model:", STUDENT_MODEL_NAME)
    model = YOLO(STUDENT_MODEL_NAME)

    logger.info("Starting STUDENT training on COCO128 (yolov8n fine-tune)")

    # Set MLflow experiment (same as teacher, so runs are grouped)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="student_yolov8n_coco128"):

        # -----------------------------
        # LOG HYPERPARAMETERS
        # -----------------------------
        mlflow.log_param("role", "student")
        mlflow.log_param("model_name", "yolov8n_student")
        mlflow.log_param("base_weights", STUDENT_MODEL_NAME)
        mlflow.log_param("data", DATA_CONFIG)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("imgsz", IMGSZ)
        mlflow.log_param("batch", BATCH)

        # -----------------------------
        # TRAINING
        # -----------------------------
        results = model.train(
            data=DATA_CONFIG,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            project="runs/student",
            name="yolov8n_coco128_student",
        )

        # -----------------------------
        # METRIC LOGGING (same pattern as teacher)
        # -----------------------------
        try:
            metrics = results.results_dict  # this is a dict in your version
            for key, value in metrics.items():
                safe_key = (
                    key.replace("/", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("-", "_")
                )
                try:
                    mlflow.log_metric(safe_key, float(value))
                except Exception:
                    logger.warning(f"MLflow could not log {safe_key}={value}")
        except Exception as e:
            logger.warning(f"Could not access results_dict from Ultralytics: {e}")

        # -----------------------------
        # SAVE BEST CHECKPOINT
        # -----------------------------
        best_ckpt = "runs/student/yolov8n_coco128_student/weights/best.pt"
        if os.path.exists(best_ckpt):
            mlflow.log_artifact(best_ckpt)
            logger.info(f"Logged student best checkpoint: {best_ckpt}")
        else:
            logger.warning("Student best.pt not found at expected location!")

        logger.info("Student training complete.")


if __name__ == "__main__":
    print(">>> train_student.py STARTED")
    train_student()
