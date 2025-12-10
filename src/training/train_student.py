import os
import mlflow
from mlflow.tracking import MlflowClient
from ultralytics import YOLO

from src.utils.config import MLFLOW_EXPERIMENT_NAME
from src.utils.logger import get_logger

logger = get_logger("train_student")

# -------------------------
# CONFIG
# -------------------------
STUDENT_MODEL_NAME = "yolov8n.pt"  # small student model
DATA_CONFIG = "coco128.yaml"
EPOCHS = 5
IMGSZ = 320
BATCH = 8

# MLflow settings
os.environ["YOLO_MLFLOW"] = "False"
os.environ["MLFLOW_TRACKING_URI"] = os.getenv(
    "MLFLOW_TRACKING_URI", "http://98.88.77.30:5000"
)

REGISTERED_MODEL_NAME = "yolo-student"  # <-- REGISTRY NAME


def train_student():
    print(">>> Initializing student model:", STUDENT_MODEL_NAME)
    model = YOLO(STUDENT_MODEL_NAME)

    logger.info("Starting STUDENT training on COCO128")

    # --- ensure tracking URI and experiment ---
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="student_yolov8n_coco128") as run:

        run_id = run.info.run_id

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

        # -----------------------------
        # SAVE BEST CHECKPOINT
        # -----------------------------
        best_ckpt = "runs/student/yolov8n_coco128_student/weights/best.pt"
        if os.path.exists(best_ckpt):
            mlflow.log_artifact(best_ckpt, artifact_path="model")  # <--- REQUIRED
            logger.info(f"Logged student best checkpoint: {best_ckpt}")
        else:
            logger.warning("Student best.pt not found at expected location!")

        # -----------------------------
        # REGISTER MODEL IN MLflow REGISTRY
        # -----------------------------
        client = MlflowClient()

        # create registry entry if not exists
        try:
            client.create_registered_model(REGISTERED_MODEL_NAME)
        except Exception:
            pass  # already exists

        # model source path inside MLflow run
        source = f"runs:/{run_id}/artifacts/model"

        mv = client.create_model_version(
            name=REGISTERED_MODEL_NAME,
            source=source,
            run_id=run_id,
        )

        # set stage to STAGING
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=mv.version,
            stage="Staging",
            archive_existing_versions=False,
        )

        logger.info(
            f"Registered student model: {REGISTERED_MODEL_NAME} v{mv.version} (Staging)"
        )
        logger.info("Student training complete.")


if __name__ == "__main__":
    print(">>> train_student.py STARTED")
    train_student()
