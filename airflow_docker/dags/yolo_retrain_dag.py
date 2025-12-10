"""
YOLO Retraining DAG
--------------------
A production-grade MLOps pipeline for the Robotics Distilled Object Detection project.

This DAG:
1. Checks if new data exists
2. Runs teacher training
3. Runs student training
4. Runs distillation training
5. Registers new versions into MLflow Model Registry
6. Promotes best model to Production if metrics improved
7. Sends Slack/email notifications
"""

import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

# -------------------------------------------------------------------------
# MLflow config — read from environment (set in docker-compose via .env)
# -------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI")

os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
os.environ["MLFLOW_REGISTRY_URI"] = MLFLOW_REGISTRY_URI

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------


def check_new_data(**context):
    """Check if new dataset exists (dummy example)."""
    dataset_path = "/workspaces/datasets/coco128/images/train2017"
    if len(os.listdir(dataset_path)) > 0:
        return "train_teacher"
    return "no_new_data"


def train_teacher_fn():
    import subprocess

    cmd = (
        "cd robotics-distilled-objectdetection && "
        "source .venv/bin/activate && "
        "python -m src.training.train_teacher"
    )
    subprocess.run(
        [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-i",
            "/opt/airflow/keys/yolo-robotics.pem",
            "ubuntu@52.0.54.129",
            cmd,
        ],
        check=True,
    )


def train_student_fn():
    import subprocess

    cmd = (
        "cd robotics-distilled-objectdetection && "
        "source .venv/bin/activate && "
        "python -m src.training.train_student"
    )
    subprocess.run(
        [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-i",
            "/opt/airflow/keys/yolo-robotics.pem",
            "ubuntu@52.0.54.129",
            cmd,
        ],
        check=True,
    )


def train_student_kd_fn():
    import subprocess

    cmd = (
        "cd robotics-distilled-objectdetection && "
        "source .venv/bin/activate && "
        "python -m src.training.train_student_kd"
    )
    subprocess.run(
        [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-i",
            "/opt/airflow/keys/yolo-robotics.pem",
            "ubuntu@52.0.54.129",
            cmd,
        ],
        check=True,
    )


def evaluate_and_promote_fn():
    """
    Compares the performance of the new KD model vs the existing Production model.
    If better → promote to Production stage.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri="http://52.0.54.129:5000")
    model_name = "yolo-student-kd"
    metric_key = "metrics/mAP50-95B"  # <-- correct metric

    # ---- Get staging model ----
    staging_list = client.get_latest_versions(model_name, ["Staging"])
    if not staging_list:
        raise Exception("❌ No staging model found!")
    staging = staging_list[0]

    # ---- Extract staging metric ----
    staging_hist = client.get_metric_history(staging.run_id, metric_key)
    if len(staging_hist) == 0:
        raise Exception(f"❌ Staging run has no metric {metric_key}")
    staging_score = staging_hist[-1].value

    # ---- Get production model (if exists) ----
    prod_list = client.get_latest_versions(model_name, ["Production"])
    if len(prod_list) == 0:
        print("⚠️ No production model yet → auto-promoting first model")
        prod_score = -1
        prod = None
    else:
        prod = prod_list[0]
        prod_hist = client.get_metric_history(prod.run_id, metric_key)
        prod_score = prod_hist[-1].value if len(prod_hist) else -1

    # ---- Compare scores ----
    print(f"Staging {metric_key} = {staging_score}")
    print(f"Production {metric_key} = {prod_score}")

    # ---- Promote if improved ----
    if staging_score > prod_score:
        client.transition_model_version_stage(
            name=model_name,
            version=staging.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"🚀 PROMOTED yolo-student-kd v{staging.version}")
    else:
        print("⚠️ No promotion — staging is not better than production")


def notify_success():
    print("🎉 Retraining + registration finished successfully!")


# -------------------------------------------------------------------------
# DAG DEFINITION
# -------------------------------------------------------------------------

default_args = {
    "owner": "rajat",
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="yolo_retrain_pipeline",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    description="Full YOLO retraining + registry + promotion pipeline",
) as dag:

    start = EmptyOperator(task_id="start")

    check_data = BranchPythonOperator(
        task_id="check_new_data",
        python_callable=check_new_data,
    )

    no_new_data = EmptyOperator(task_id="no_new_data")

    train_teacher = PythonOperator(
        task_id="train_teacher",
        python_callable=train_teacher_fn,
    )

    train_student = PythonOperator(
        task_id="train_student",
        python_callable=train_student_fn,
    )

    train_student_kd = PythonOperator(
        task_id="train_student_kd",
        python_callable=train_student_kd_fn,
    )

    eval_promote = PythonOperator(
        task_id="evaluate_and_promote",
        python_callable=evaluate_and_promote_fn,
    )

    notify = PythonOperator(
        task_id="notify",
        python_callable=notify_success,
    )

    end = EmptyOperator(task_id="end")

    # DAG flow
    start >> check_data
    check_data >> no_new_data >> end

    (
        check_data
        >> train_teacher
        >> train_student
        >> train_student_kd
        >> eval_promote
        >> notify
        >> end
    )
