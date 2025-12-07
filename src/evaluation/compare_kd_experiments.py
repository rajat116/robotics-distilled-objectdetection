import os
from pathlib import Path
from typing import Dict, List

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.config import MLFLOW_EXPERIMENT_NAME
from src.utils.logger import get_logger

logger = get_logger("compare_kd_experiments")

# Use the same tracking URI as training scripts
os.environ.setdefault("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


# ---------------------------------------------------------
# UTILS
# ---------------------------------------------------------


def get_repo_root() -> Path:
    """Resolve repo root relative to this file."""
    return Path(__file__).resolve().parents[2]


def get_experiment_id(experiment_name: str) -> str:
    """Return experiment ID for the given name, or raise if not found."""
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(
            f"MLflow experiment '{experiment_name}' not found. "
            "Make sure you've run the training scripts and that "
            "MLFLOW_TRACKING_URI is correct."
        )
    logger.info(f"Using experiment '{experiment_name}' (id={exp.experiment_id})")
    return exp.experiment_id


def get_run_by_name(
    client: MlflowClient,
    experiment_id: str,
    run_name: str,
):
    """
    Return the latest run with the given run_name (mlflow.runName tag).
    """
    # Filter by MLflow default run name tag
    # NOTE: tags.mlflow.runName is the standard name field used by mlflow.start_run(run_name=...)
    filter_str = f"tags.mlflow.runName = '{run_name}'"

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_str,
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )

    if not runs:
        raise RuntimeError(
            f"No MLflow run found with run_name='{run_name}' in experiment {experiment_id}. "
            f"Did you run the corresponding training script?"
        )

    run = runs[0]
    logger.info(f"Found run for '{run_name}': run_id={run.info.run_id}")
    return run


def extract_metrics(run) -> Dict[str, float]:
    """
    Extract metrics as a simple dict: {metric_name: value}.
    """
    return dict(run.data.metrics)


def pretty_print_metrics_table(
    metric_names: List[str],
    teacher_metrics: Dict[str, float],
    student_metrics: Dict[str, float],
    kd_metrics: Dict[str, float],
):
    """
    Print a formatted side-by-side table of metric values.
    """
    print("\n================ YOLO KD METRIC COMPARISON ================\n")
    header = (
        f"{'metric':35s} | {'teacher':>12s} | {'student':>12s} | {'student_KD':>12s}"
    )
    print(header)
    print("-" * len(header))

    for m in metric_names:
        t = teacher_metrics.get(m, float("nan"))
        s = student_metrics.get(m, float("nan"))
        k = kd_metrics.get(m, float("nan"))
        print(f"{m:35s} | {t:12.5f} | {s:12.5f} | {k:12.5f}")

    print("\n===========================================================\n")


def select_intersection_metrics(
    teacher_metrics: Dict[str, float],
    student_metrics: Dict[str, float],
    kd_metrics: Dict[str, float],
) -> List[str]:
    """
    Only keep metrics that are present in all three runs.
    """
    common = (
        set(teacher_metrics.keys())
        & set(student_metrics.keys())
        & set(kd_metrics.keys())
    )
    # Sort them for consistent output
    return sorted(common)


def suggest_key_metrics(metric_names: List[str]) -> List[str]:
    """
    Try to guess the most interesting metrics to focus on for detection:
    - mAP metrics
    - precision / recall metrics
    """
    key_like = []
    for m in metric_names:
        lower = m.lower()
        if "map" in lower or "precision" in lower or "recall" in lower:
            key_like.append(m)

    # Deduplicate and sort
    return sorted(set(key_like))


def save_bar_plot(
    metric_names: List[str],
    teacher_metrics: Dict[str, float],
    student_metrics: Dict[str, float],
    kd_metrics: Dict[str, float],
    out_path: Path,
):
    """
    Save a simple bar chart comparing a small set of key metrics.
    """
    if not metric_names:
        logger.warning("No key metrics found for plotting. Skipping plot.")
        return

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib/numpy not installed, skipping plot.")
        return

    x = np.arange(len(metric_names))
    width = 0.25

    teacher_vals = [teacher_metrics[m] for m in metric_names]
    student_vals = [student_metrics[m] for m in metric_names]
    kd_vals = [kd_metrics[m] for m in metric_names]

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, teacher_vals, width, label="teacher")
    plt.bar(x, student_vals, width, label="student")
    plt.bar(x + width, kd_vals, width, label="student_KD")

    plt.xticks(x, metric_names, rotation=45, ha="right")
    plt.ylabel("metric value")
    plt.title("Teacher vs Student vs Student-KD (key metrics)")
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()

    logger.info(f"Saved comparison plot to: {out_path}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------


def main():
    print(">>> compare_kd_experiments.py STARTED")

    experiment_id = get_experiment_id(MLFLOW_EXPERIMENT_NAME)
    client = MlflowClient()

    # Names must match your training scripts
    teacher_run_name = "teacher_yolov8s_coco128"
    student_run_name = "student_yolov8n_coco128"
    kd_run_name = "student_yolov8n_coco128_KD"

    # Fetch runs
    teacher_run = get_run_by_name(client, experiment_id, teacher_run_name)
    student_run = get_run_by_name(client, experiment_id, student_run_name)
    kd_run = get_run_by_name(client, experiment_id, kd_run_name)

    # Extract metrics
    teacher_metrics = extract_metrics(teacher_run)
    student_metrics = extract_metrics(student_run)
    kd_metrics = extract_metrics(kd_run)

    if not teacher_metrics:
        logger.warning("Teacher run has no metrics logged.")
    if not student_metrics:
        logger.warning("Student baseline run has no metrics logged.")
    if not kd_metrics:
        logger.warning("Student KD run has no metrics logged.")

    # Only compare metrics that exist in all three
    common_metrics = select_intersection_metrics(
        teacher_metrics, student_metrics, kd_metrics
    )

    if not common_metrics:
        raise RuntimeError(
            "No common metrics found across the three runs. "
            "Check that your training scripts logged overlapping metric keys."
        )

    # Print full comparison table
    pretty_print_metrics_table(
        common_metrics, teacher_metrics, student_metrics, kd_metrics
    )

    # Try to find interesting key metrics for plotting
    key_metrics = suggest_key_metrics(common_metrics)
    if key_metrics:
        print("Key metrics selected for plotting:")
        for m in key_metrics:
            print("  -", m)
    else:
        print(
            "No specific key metrics detected (mAP/precision/recall). Will use all common metrics for plotting."
        )
        key_metrics = common_metrics

    # Save bar plot
    repo_root = get_repo_root()
    out_plot = repo_root / "figures" / "yolo_kd_comparison.png"
    save_bar_plot(key_metrics, teacher_metrics, student_metrics, kd_metrics, out_plot)

    print(">>> compare_kd_experiments.py DONE")


if __name__ == "__main__":
    main()
