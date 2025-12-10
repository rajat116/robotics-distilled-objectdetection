import json
import subprocess
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

EC2_HOST = "ubuntu@52.0.54.129"
SSH_KEY = "/opt/airflow/keys/yolo-robotics.pem"
DRIFT_DIR = "/home/ubuntu/robotics-distilled-objectdetection/reports/drift"


def run_drift_ec2():
    """Run drift job on EC2."""
    cmd = (
        "cd robotics-distilled-objectdetection && "
        "source .venv/bin/activate && "
        "python -m src.monitoring.drift_job"
    )

    subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-i", SSH_KEY, EC2_HOST, cmd],
        check=True,
    )


def decide_retrain(**context):
    """Read drift score directly from EC2 via SSH."""
    # List drift folders on EC2
    cmd_list = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-i",
        SSH_KEY,
        EC2_HOST,
        f"ls -1 {DRIFT_DIR}",
    ]

    result = subprocess.run(cmd_list, capture_output=True, text=True)
    folders = result.stdout.strip().split("\n")
    folders = [f for f in folders if f]

    if not folders:
        raise Exception("❌ No drift reports found on EC2")

    latest = sorted(folders)[-1]

    # Read JSON via SSH
    cmd_json = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-i",
        SSH_KEY,
        EC2_HOST,
        f"cat {DRIFT_DIR}/{latest}/report.json",
    ]

    json_output = subprocess.run(cmd_json, capture_output=True, text=True).stdout
    data = json.loads(json_output)
    drift_score = data["drift_score"]

    print("📊 Drift score =", drift_score)

    # Threshold decision
    if drift_score > 0.3:
        print("⚠️ Drift above threshold → retraining triggered")
        return "trigger_retrain"
    else:
        print("👌 Drift normal → no retraining")
        return "no_retrain"


def trigger_training():
    """Trigger the retraining DAG from Airflow."""
    from airflow.api.client.local_client import Client

    c = Client(None, None)
    c.trigger_dag(dag_id="yolo_retrain_pipeline")
    print("🚀 Triggered yolo_retrain_pipeline")


default_args = {
    "owner": "rajat",
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}

with DAG(
    "drift_monitoring_pipeline",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    description="Drift detection and auto-retraining trigger",
):

    start = EmptyOperator(task_id="start")

    run_drift = PythonOperator(
        task_id="run_drift",
        python_callable=run_drift_ec2,
    )

    decide = BranchPythonOperator(
        task_id="decide_retrain",
        python_callable=decide_retrain,
    )

    no_retrain = EmptyOperator(task_id="no_retrain")
    trigger = PythonOperator(
        task_id="trigger_retrain",
        python_callable=trigger_training,
    )

    end = EmptyOperator(task_id="end")

    start >> run_drift >> decide
    decide >> no_retrain >> end
    decide >> trigger >> end
