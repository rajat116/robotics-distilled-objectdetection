from pathlib import Path

# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"

# MLflow settings
MLFLOW_EXPERIMENT_NAME = "robotics_distilled_detection"
