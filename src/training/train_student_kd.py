import os

import mlflow
from ultralytics import YOLO

from src.utils.config import MLFLOW_EXPERIMENT_NAME
from src.utils.logger import get_logger

logger = get_logger("train_student_kd")

# Config
STUDENT_MODEL_NAME = "yolov8n.pt"  # same small model as baseline
DATA_CONFIG = "coco128.yaml"  # same dataset, but labels now = teacher predictions
EPOCHS = 5
IMGSZ = 320
BATCH = 8

os.environ["YOLO_MLFLOW"] = "False"
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"


def train_student_kd():
    """
    Train student model on teacher-generated pseudo-labels
    (distilled training).
    """

    print(">>> train_student_kd.py STARTED")
    print(">>> Initializing student model:", STUDENT_MODEL_NAME)
    model = YOLO(STUDENT_MODEL_NAME)

    logger.info("Starting STUDENT KD training on COCO128 (teacher pseudo-labels)")

    # Group under same experiment as teacher & baseline student
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="student_yolov8n_coco128_KD"):
        # Params
        mlflow.log_param("role", "student_KD")
        mlflow.log_param("model_name", "yolov8n_student_KD")
        mlflow.log_param("base_weights", STUDENT_MODEL_NAME)
        mlflow.log_param("data", DATA_CONFIG)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("imgsz", IMGSZ)
        mlflow.log_param("batch", BATCH)
        mlflow.log_param("distillation", "teacher_pseudo_labels")

        # Training
        results = model.train(
            data=DATA_CONFIG,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            project="runs/student",
            name="yolov8n_coco128_student_KD",
        )

        # Metrics
        try:
            metrics = results.results_dict  # dict in your version
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

        # Save best checkpoint
        best_ckpt = "runs/student/yolov8n_coco128_student_KD/weights/best.pt"
        if os.path.exists(best_ckpt):
            mlflow.log_artifact(best_ckpt)
            logger.info(f"Logged KD student best checkpoint: {best_ckpt}")
        else:
            logger.warning("KD student best.pt not found at expected location!")

        logger.info("Student KD training complete.")


if __name__ == "__main__":
    train_student_kd()
