import os
import socket
import mlflow
from mlflow.tracking import MlflowClient
from ultralytics import YOLO

from src.utils.config import MLFLOW_EXPERIMENT_NAME
from src.utils.logger import get_logger

logger = get_logger("train_teacher")


def is_ec2():
    # EC2 always has this file
    if os.path.exists("/sys/hypervisor/uuid"):
        return True
    # EC2 hostnames start with "ip-"
    return socket.gethostname().startswith("ip-")


# Name for the MLflow Model Registry entry
REGISTERED_MODEL_NAME = "yolo-teacher"


def train_teacher():

    # ----------------------
    # Dataset config selector
    # ----------------------
    if is_ec2():
        data_config = "data/coco128_ec2.yaml"
        mlflow_uri = "http://52.0.54.129:5000"
    else:
        data_config = "data/coco128_local.yaml"
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://52.0.54.129:5000")

    os.environ["YOLO_MLFLOW"] = "False"
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    model = YOLO("yolov8s.pt")

    epochs = 5
    imgsz = 320
    batch = 8

    logger.info(f"Training started using dataset {data_config}")

    # Capture run handle so we can use run_id later
    with mlflow.start_run(run_name="teacher_yolov8s_coco128") as run:

        run_id = run.info.run_id

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
            # ⬅️ IMPORTANT: put it under a known artifact path: "model"
            mlflow.log_artifact(best_ckpt, artifact_path="model")
            logger.info(f"Logged best checkpoint: {best_ckpt}")
        else:
            logger.warning("best.pt not found at expected location!")

        # -----------------------------
        # REGISTER MODEL IN MLFLOW REGISTRY
        # -----------------------------
        client = MlflowClient()

        # Make sure the registered model exists (ignore if already created)
        try:
            client.create_registered_model(REGISTERED_MODEL_NAME)
        except Exception:
            # Typically AlreadyExistsError; safe to ignore
            pass

        # Source path in this run's artifacts
        # We logged as artifact_path="model", so the directory is "model"
        source = f"runs:/{run_id}/artifacts/model"

        mv = client.create_model_version(
            name=REGISTERED_MODEL_NAME,
            source=source,
            run_id=run_id,
        )

        # Optionally set stage to "Staging" for now
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=mv.version,
            stage="Staging",
            archive_existing_versions=False,
        )

        logger.info(
            f"Registered teacher model in MLflow: {REGISTERED_MODEL_NAME} v{mv.version} (Staging)"
        )
        logger.info("Teacher training complete.")


if __name__ == "__main__":
    train_teacher()
