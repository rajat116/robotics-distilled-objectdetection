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
# ENVIRONMENT CONFIG FOR MLflow INSIDE DOCKER
# -------------------------------------------------------------------------
os.environ["MLFLOW_TRACKING_URI"] = "http://host.docker.internal:5000"
os.environ["MLFLOW_REGISTRY_URI"] = "http://host.docker.internal:5000"

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

    subprocess.run(["python", "-m", "src.training.train_teacher"], check=True)


def train_student_fn():
    import subprocess

    subprocess.run(["python", "-m", "src.training.train_student"], check=True)


def train_student_kd_fn():
    import subprocess

    subprocess.run(["python", "-m", "src.training.train_student_kd"], check=True)


def evaluate_and_promote_fn():
    """
    Compares the performance of the new KD model vs the existing Production model.
    If better → promote to Production stage.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    model_name = "yolo-student-kd"

    # Get staging version
    staging = client.get_latest_versions(model_name, stages=["Staging"])[0]
    prod = None
    try:
        prod = client.get_latest_versions(model_name, stages=["Production"])[0]
    except Exception:
        pass  # No production model yet

    # For now compare F1 score (stored in MLflow metrics)
    staging_f1 = client.get_metric_history(staging.run_id, "F1")[-1].value

    if prod:
        prod_f1 = client.get_metric_history(prod.run_id, "F1")[-1].value
    else:
        prod_f1 = -1  # force promote first model

    if staging_f1 > prod_f1:
        client.transition_model_version_stage(
            name=model_name,
            version=staging.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print("🚀 Model promoted to Production")
    else:
        print("⚠️ Staging model NOT better than Production")


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
