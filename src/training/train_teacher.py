import os
import mlflow
from ultralytics import YOLO

from src.utils.config import MLFLOW_EXPERIMENT_NAME
from src.utils.logger import get_logger

os.environ["YOLO_MLFLOW"] = "False"
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"

logger = get_logger("train_teacher")


def train_teacher():

    model = YOLO("yolov8s.pt")
    data_config = "coco128.yaml"

    epochs = 5
    imgsz = 320
    batch = 8

    logger.info("Starting teacher training on COCO128 (small model)")

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="teacher_yolov8s_coco128"):

        mlflow.log_param("model_name", "yolov8n")
        mlflow.log_param("data", data_config)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("imgsz", imgsz)
        mlflow.log_param("batch", batch)

        results = model.train(
            data=data_config,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project="runs/teacher",
            name="yolov8s_coco128",
        )

        # -----------------------------
        # METRIC LOGGING (works always)
        # -----------------------------
        metrics = results.results_dict  # <-- dict

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

        # -----------------------------
        # SAVE BEST WEIGHT MANUALLY
        # -----------------------------
        best_ckpt = "runs/teacher/yolov8s_coco128/weights/best.pt"
        if os.path.exists(best_ckpt):
            mlflow.log_artifact(best_ckpt)
            logger.info(f"Logged best checkpoint: {best_ckpt}")
        else:
            logger.warning("best.pt not found at expected location!")

        logger.info("Teacher training complete.")


if __name__ == "__main__":
    train_teacher()
