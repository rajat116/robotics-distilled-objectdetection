import os
from ultralytics import YOLO
import mlflow
from mlflow.tracking import MlflowClient
from mlflow import artifacts as mlflow_artifacts

# ----------------------------------------------------------------------
# LOCAL MODEL PATHS (fallback mode)
# ----------------------------------------------------------------------
MODEL_PATHS = {
    "teacher": "runs/teacher/yolov8s_coco128/weights/best.pt",
    "student": "runs/student/yolov8n_coco128_student/weights/best.pt",
    "student_kd": "runs/student/yolov8n_coco128_student_KD/weights/best.pt",
}

DEFAULT_MODEL = "student_kd"


# ----------------------------------------------------------------------
# MLflow model loading
# ----------------------------------------------------------------------
def _load_mlflow_model():
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow import artifacts as mlflow_artifacts
    import os

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    model_name = os.getenv("YOLO_MODEL_NAME", "yolo-student-kd")
    model_stage = os.getenv("YOLO_MODEL_STAGE", "Staging")

    client = MlflowClient()

    versions = client.get_latest_versions(model_name, stages=[model_stage])
    if not versions:
        raise RuntimeError(f"No model versions found at stage {model_stage}")

    mv = versions[0]
    print(f"[MLflow] Loading {model_name} v{mv.version} ({model_stage})")

    # ------------------------------------------------------------
    # ✔ Download ENTIRE artifact directory for the run
    # ------------------------------------------------------------
    local_artifacts = mlflow_artifacts.download_artifacts(
        run_id=mv.run_id, artifact_path=""  # download full artifacts folder
    )

    # ------------------------------------------------------------
    # ✔ Now search for best.pt anywhere under this folder
    # ------------------------------------------------------------
    best_path = None
    for root, dirs, files in os.walk(local_artifacts):
        if "best.pt" in files:
            best_path = os.path.join(root, "best.pt")
            break

    if best_path is None:
        raise FileNotFoundError(
            f"Could not find best.pt under MLflow artifacts for run {mv.run_id}"
        )

    print(f"[MLflow] Using weights at: {best_path}")

    model = YOLO(best_path)

    def predictor(image):
        return model.predict(image, imgsz=320, verbose=False)

    return predictor


def load_shadow_model():
    """
    Optional shadow model for A/B testing.
    Loads from MLflow only if YOLO_SHADOW=True.
    """
    import os

    if os.getenv("YOLO_SHADOW", "False") != "True":
        print("[Shadow] Shadow mode disabled.")
        return None

    model_name = os.getenv("SHADOW_MODEL_NAME")
    model_stage = os.getenv("SHADOW_MODEL_STAGE")

    print("DEBUG ENV YOLO_SHADOW =", os.getenv("YOLO_SHADOW"))
    print("DEBUG ENV SHADOW_MODEL_NAME =", os.getenv("SHADOW_MODEL_NAME"))
    print("DEBUG ENV SHADOW_MODEL_STAGE =", os.getenv("SHADOW_MODEL_STAGE"))

    if not model_name or not model_stage:
        print("[Shadow] Missing SHADOW_MODEL_NAME or SHADOW_MODEL_STAGE")
        return None

    print(f"[Shadow] Loading shadow model {model_name}:{model_stage}")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=[model_stage])
    if not versions:
        print("[Shadow] No shadow model found!")
        return None

    mv = versions[0]
    print(f"[Shadow] Found shadow version v{mv.version}")

    # Download entire artifact folder (robust)
    local_artifacts = mlflow_artifacts.download_artifacts(
        run_id=mv.run_id, artifact_path=""
    )

    # Locate best.pt
    best_path = None
    for root, dirs, files in os.walk(local_artifacts):
        if "best.pt" in files:
            best_path = os.path.join(root, "best.pt")
            break

    if best_path is None:
        print("[Shadow] Could not find best.pt for shadow model")
        return None

    print(f"[Shadow] Using weights: {best_path}")

    from ultralytics import YOLO

    shadow_model = YOLO(best_path)

    def predictor(image):
        return shadow_model.predict(image, imgsz=320, verbose=False)

    return predictor


# ----------------------------------------------------------------------
# LOCAL fallback loader
# ----------------------------------------------------------------------
def _load_local_model():
    name = os.getenv("MODEL_NAME", DEFAULT_MODEL)
    if name not in MODEL_PATHS:
        raise ValueError(f"Unknown MODEL_NAME={name}")

    weights = MODEL_PATHS[name]
    print(f"[model_loader] Loading LOCAL model '{name}' -> {weights}")

    model = YOLO(weights)

    def predictor(image):
        return model.predict(image, imgsz=320, verbose=False)

    return predictor


# ----------------------------------------------------------------------
# PUBLIC loader
# ----------------------------------------------------------------------
def load_model():
    use_mlflow = os.getenv("YOLO_MLFLOW", "False").lower() == "true"

    if use_mlflow:
        return _load_mlflow_model()

    return _load_local_model()
