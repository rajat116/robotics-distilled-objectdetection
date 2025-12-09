import os
import mlflow
from mlflow.tracking import MlflowClient
from ultralytics import YOLO

from src.utils.config import MLFLOW_EXPERIMENT_NAME
from src.utils.logger import get_logger

logger = get_logger("train_student_kd")

# Config
STUDENT_MODEL_NAME = "yolov8n.pt"
DATA_CONFIG = "coco128.yaml"
EPOCHS = 5
IMGSZ = 320
BATCH = 8

# Disable Ultralytics built-in MLflow
os.environ["YOLO_MLFLOW"] = "False"
os.environ.setdefault("MLFLOW_TRACKING_URI", "file:./mlruns")

# MLflow registry model name
REGISTERED_MODEL_NAME = "yolo-student-kd"


def train_student_kd():
    print(">>> train_student_kd.py STARTED")
    print(">>> Initializing student model:", STUDENT_MODEL_NAME)

    model = YOLO(STUDENT_MODEL_NAME)

    logger.info("Starting STUDENT KD training on COCO128 (teacher pseudo-labels)")

    # Ensure MLflow tracking URI + experiment
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="student_yolov8n_coco128_KD") as run:

        run_id = run.info.run_id

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
            metrics = results.results_dict
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
            mlflow.log_artifact(best_ckpt, artifact_path="model")  # <-- REQUIRED
            logger.info(f"Logged KD student best checkpoint: {best_ckpt}")
        else:
            logger.warning("KD student best.pt not found at expected location!")

        # ------------------------------------------------------------------
        # REGISTER KD MODEL IN MLflow MODEL REGISTRY
        # ------------------------------------------------------------------
        client = MlflowClient()

        # Create registry entry if missing
        try:
            client.create_registered_model(REGISTERED_MODEL_NAME)
        except Exception:
            pass  # already exists

        source = f"runs:/{run_id}/artifacts/model"

        mv = client.create_model_version(
            name=REGISTERED_MODEL_NAME,
            source=source,
            run_id=run_id,
        )

        # Add new version to STAGING
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=mv.version,
            stage="Staging",
            archive_existing_versions=False,
        )

        logger.info(
            f"Registered KD student model: {REGISTERED_MODEL_NAME} v{mv.version} (Staging)"
        )
        logger.info("Student KD training complete.")


if __name__ == "__main__":
    train_student_kd()
