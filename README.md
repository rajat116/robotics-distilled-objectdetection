# **Robotics Object Detection with YOLOv8 Knowledge Distillation + End-to-End MLOps on AWS**

This project demonstrates a **production-grade machine learning system** for deploying, monitoring, and retraining an edge-ready **object detection model** for robotics applications.

Robotics hardware imposes strict constraints:

* ✔ **low latency**
* ✔ **small model size**
* ✔ **consistent tail-latency (p95)**
* ✔ **high detection accuracy**
* ✔ **ability to adapt to drift over time**

To address this, we build a **full MLOps pipeline** that:

* Trains a **YOLOv8s teacher** and a lightweight **YOLOv8n student**
* Applies **label-based knowledge distillation** to create a much stronger **YOLOv8n-KD student**
* Benchmarks latency, accuracy, and robustness
* Deploys via **FastAPI + Docker**
* Uses **MLflow Model Registry** (running on AWS EC2 + S3)
* Automates retraining & promotion via **Airflow**
* Monitors data drift with a dedicated **Drift Detection DAG**
* Provides model comparison plots, inference visualizations, and performance summaries

This repository showcases **all components of the ML lifecycle**—training → registry → inference → monitoring → retraining → redeployment.

---

# 🔍 **1. Problem Motivation**

Modern robotic systems (autonomous mobile robots, UAVs, warehouse bots) rely heavily on **real-time object detection**.
However:

* High-capacity models (YOLOv8s, YOLOv8m) are accurate but **too slow/heavy** for embedded devices
* Tiny models (YOLOv8n) are fast but **lose significant accuracy**
* The model must run **24/7 on hardware**, meaning:

  * strict **latency budgets**
  * **limited compute and memory**
  * **high reliability**
  * **resistance to data drift**

Therefore, we need a model that is:

* **small**
* **fast**
* **accurate**
* **maintainable in production**
* **automatically retrained when drift occurs**

---

# 🧠 **2. Solution: Knowledge Distillation + Full MLOps Pipeline**

We implement **label-based knowledge distillation (KD)**:

* The **teacher** (YOLOv8s) generates high-quality pseudo-labels
* The **student** (YOLOv8n) trains on those labels
* The distilled student becomes **much more accurate** while remaining **lightweight and fast**

Then we integrate this training process into a **complete MLOps system**:

### ✔ Training on AWS EC2 with MLflow logging

### ✔ Model Registry with versioning (Staging → Production)

### ✔ Retraining DAG triggered by Airflow

### ✔ Drift Detection DAG to monitor incoming data

### ✔ CI/CD with GitHub Actions

### ✔ Dockerized inference (FastAPI)

### ✔ Benchmarking suite

### ✔ Automated model promotion and rollback logic

This is the same workflow used in professional robotics, autonomous driving, and real-time AI systems.

---

# 🧩 **3. Architecture Overview**

## **High-Level Pipeline**

```
                  ┌────────────────────┐
                  │   Raw Dataset      │
                  └─────────┬──────────┘
                            ▼
                  ┌────────────────────┐
                  │   Teacher Model     │ (YOLOv8s)
                  │ GT + Predictions    │
                  └─────────┬──────────┘
                            ▼
                  ┌────────────────────┐
                  │  KD Label Generator │
                  └─────────┬──────────┘
                            ▼
        ┌───────────────────────────────────────┐
        │ Student (YOLOv8n)    ──────────────┐ │
        │ KD-Student (YOLOv8n-KD) ◄───────────┘ │
        └───────────────────────────────────────┘
                            ▼
                  ┌────────────────────┐
                  │ MLflow Tracking     │
                  │ & Model Registry    │
                  └─────────┬──────────┘
                            ▼
             ┌─────────────────────────────┐
             │ Airflow Retrain Pipeline     │
             └──────────┬──────────────────┘
                        ▼
            ┌──────────────────────────────┐
            │ Production Model (Latest KD)  │
            └──────────┬───────────────────┘
                       ▼
       ┌────────────────────────────────────────┐
       │ FastAPI + Docker Inference Service     │
       └────────────────────────────────────────┘
```

---

Here is the **correct, complete, final README repo-structure section**, rewritten professionally, fully aligned with the *actual* tree you provided.

Use this **exact block** in your README.
Nothing missing, nothing extra — fully consistent with your repository.

---

# 📁 **4. Repository Structure**

Your project follows a clean MLOps-style layout: modular code, Airflow orchestration, MLflow tracking, inference service, benchmarking, drift monitoring, and evaluation tools.

Below is the **accurate repo structure** based on your latest `tree` output.

---

## **Source Code (`src/`)**

```
src/
│
├── training/                     # Teacher, student, and KD student training scripts
│   ├── train_teacher.py
│   ├── train_student.py
│   └── train_student_kd.py
│
├── distillation/                 # Generate KD pseudo-labels from teacher predictions
│   └── generate_kd_labels.py
│
├── inference/                    # FastAPI inference service + model loader
│   ├── api.py
│   └── model_loader.py
│
├── monitoring/                   # Drift detection + Prometheus instrumentation
│   ├── drift_job.py
│   └── metrics.py
│
├── evaluation/                   # MLflow metric comparison + bar-chart generation
│   └── compare_kd_experiments.py
│
├── benchmarking/                 # Latency + throughput benchmarking scripts
│   └── benchmark.py
│
└── utils/                        # Configuration + logging utilities
    ├── config.py
    └── logger.py
```

---

## **Airflow Orchestration (`airflow_docker/`)**

```
airflow_docker/
│
├── dags/                          # Automated training + automated drift pipelines
│   ├── drift_monitoring_pipeline.py
│   └── yolo_retrain_dag.py
│
├── Dockerfile.worker              # Airflow worker image
├── docker-compose.yaml            # Airflow deployment for Codespaces
│
├── config/
│   └── airflow.cfg                # Airflow configuration
│
└── keys/
    └── yolo-robotics.pem          # SSH key for EC2 remote execution
```

---

## **Models**

```
models/
├── teacher/                       # Trained YOLOv8s teacher weights
├── student/                       # Baseline YOLOv8n student
└── student_kd/                    # Distilled YOLOv8n KD student
```

---

## **Figures (for README)**

```
figures/
├── input_image.jpg
├── detection_teacher.jpg
├── detection_student.jpg
├── detection_student_kd.jpg
└── yolo_kd_comparison.png        # mAP/precision/recall bar chart
```

---

## **Other Important Files**

```
Dockerfile.inference              # FastAPI model-inference image
requirements.txt                  # Full environment for training + Airflow
requirements.inference.txt        # Lightweight inference-only env
benchmark_results.csv             # Latency + model size CSV
mlflow.db                         # MLflow backend store (local mode)
scripts/
│   └── generate_inference_images.py  # Script to generate inference outputs
tests/                            # Unit tests (model loader, API, training)
```

---

# 🖼️ **5. Inference Visualizations**

These are automatically generated using:

```
python scripts/generate_inference_images.py
```

### **Input Image**

<img src="figures/input_image.jpg" width="260" height="350"/>

### **Teacher Detection**

<img src="figures/detection_teacher.jpg" width="260" height="350"/>

### **Student Detection**

<img src="figures/detection_student.jpg" width="260" height="350"/>

### **Distilled Student Detection**

<img src="figures/detection_student_kd.jpg" width="260" height="350"/>

---

# 📊 **6. Performance Comparison**

## ✔ Key Detection Metrics (from YOLO results.csv)

![matric_comparison](figures/yolo_kd_comparison.png)

<table>
<tr>
    <th>metric</th>
    <th>teacher</th>
    <th>student</th>
    <th>student_KD</th>
</tr>

<tr><td>lr/pg0</td><td>0.00002</td><td>0.00002</td><td>0.00002</td></tr>
<tr><td>lr/pg1</td><td>0.00002</td><td>0.00002</td><td>0.00002</td></tr>
<tr><td>lr/pg2</td><td>0.00002</td><td>0.00002</td><td>0.00002</td></tr>

<tr><td>metrics/mAP50-95B</td><td>0.48802</td><td>0.38716</td><td><b>0.61971</b></td></tr>
<tr><td>metrics/mAP50B</td><td>0.64043</td><td>0.52062</td><td><b>0.76087</b></td></tr>
<tr><td>metrics/precisionB</td><td>0.65281</td><td>0.59245</td><td>0.64294</td></tr>
<tr><td>metrics/recallB</td><td>0.60489</td><td>0.50324</td><td><b>0.71211</b></td></tr>

<tr><td>train/box_loss</td><td>1.20095</td><td>1.35058</td><td><b>1.08518</b></td></tr>
<tr><td>train/cls_loss</td><td>1.30011</td><td>2.07919</td><td><b>1.78517</b></td></tr>
<tr><td>train/dfl_loss</td><td>1.18710</td><td>1.28039</td><td><b>1.18243</b></td></tr>

<tr><td>val/box_loss</td><td>0.95001</td><td>1.13281</td><td><b>0.72485</b></td></tr>
<tr><td>val/cls_loss</td><td>0.78308</td><td>1.12302</td><td>0.86491</td></tr>
<tr><td>val/dfl_loss</td><td>0.98932</td><td>1.07025</td><td><b>0.92875</b></td></tr>

</table>

### ✔ Interpretation

* KD student achieves **~46% improvement in mAP50** over the baseline
* KD student surpasses even the teacher in overall mAP on COCO128
* Recall improves significantly → fewer missed detections
* Precision remains similar → quality maintained

---

# ⚡ **7. Model Sizes & Latency**

## **Size Comparison**

<h3>Model Size Comparison</h3>

<table>
<tr>
    <th>model</th>
    <th>size (MB)</th>
    <th>notes</th>
</tr>
<tr>
    <td>teacher (YOLOv8s)</td>
    <td>21.5 MB</td>
    <td>High accuracy but heavy</td>
</tr>
<tr>
    <td>student (YOLOv8n)</td>
    <td>6.2 MB</td>
    <td>Fast but less accurate</td>
</tr>
<tr>
    <td>student_KD (YOLOv8n-KD)</td>
    <td>6.2 MB</td>
    <td><b>Small & fast with teacher-level accuracy</b></td>
</tr>
</table>


## **Average Latency**

<h3>Latency Benchmark (ms)</h3>

<table>
<tr>
    <th>model</th>
    <th>avg latency (ms)</th>
    <th>p95 latency (ms)</th>
    <th>speedup vs teacher</th>
</tr>
<tr>
    <td>teacher</td>
    <td>84</td>
    <td>103</td>
    <td>1× (baseline)</td>
</tr>
<tr>
    <td>student</td>
    <td>34</td>
    <td>52</td>
    <td><b>2.4× faster</b></td>
</tr>
<tr>
    <td>student_KD</td>
    <td>46</td>
    <td>91</td>
    <td><b>1.8× faster</b></td>
</tr>
</table>


## **p95 Worst-Case Latency**

<h3>p95 Latency Comparison</h3>

<table>
<tr>
    <th>model</th>
    <th>p95 latency</th>
    <th>interpretation</th>
</tr>

<tr>
    <td>teacher</td>
    <td>103 ms</td>
    <td>slow heavy model</td>
</tr>

<tr>
    <td>student</td>
    <td>52 ms</td>
    <td>stable & consistent</td>
</tr>

<tr>
    <td>student_KD</td>
    <td>91 ms</td>
    <td>slight KD overhead</td>
</tr>
</table>

---

# 🧪 **8. Full MLOps Pipeline Components**

## ✔ **Training on EC2**

All training scripts log:

* hyperparameters
* losses
* mAP metrics
* weight artifacts
* confusion matrices

into MLflow (running on EC2 + S3).

---

## ✔ **MLflow Model Registry**

Models stored as:

* `yolo-teacher`
* `yolo-student`
* `yolo-student-kd`

Stages:

* Staging
* Production

Promotion automated via Airflow.

---

## ✔ **Airflow Retraining Pipeline**

DAG executes:

1. Check for new data
2. Train teacher
3. Train student
4. Train KD student
5. Register models in MLflow
6. Compare mAP50–95
7. Promote best model → Production
8. Notification

---

## ✔ **Drift Detection Pipeline**

* Runs on EC2
* Computes pixel-level drift
* Saves JSON reports under `reports/drift/`
* Airflow pulls drift results via SCP
* If drift_score > threshold → triggers retraining DAG

This simulates a real production system where environments drift over time.

---

## ✔ **Inference Service (FastAPI + Docker)**

Endpoints:

```
/health
/predict
/metrics  (Prometheus format)
```

Supports shadow deployments:

* primary = production model
* shadow = next candidate model

Allows online A/B evaluation.

---

## ✔ **CI/CD with GitHub Actions**

CI runs:

* Black
* Flake8
* Pytest
* Builds inference Docker image
* Ensures code quality + reproducibility

---

# ⚙️ **9. How to Run the Project**

---

## **A. Local Inference**

```
docker build -t yolo-inference -f Dockerfile.inference .
docker run -p 8000:8000 yolo-inference
```

Open:

```
http://localhost:8000/docs
```

Upload an image → get detections.

---

## **B. Airflow (in Codespace)**

```
cd airflow_docker
docker compose up -d
```

Open UI:

```
http://localhost:8080
```

---

## **C. Training on EC2**

SSH:

```
ssh -i yolo-robotics.pem ubuntu@<EC2-IP>
```

Run training:

```
python -m src.training.train_teacher
```

---

## **D. Drift Job Manual Run**

```
python src/monitoring/drift_job.py
```

---

# 🏁 **10. Conclusion**

This repository demonstrates a **complete, production-grade MLOps system**, built around a robotics-ready object detection pipeline:

* ✔ Lightweight + accurate model via knowledge distillation
* ✔ MLflow model registry for lifecycle management
* ✔ Airflow for orchestration & automated retraining
* ✔ Drift detection pipeline
* ✔ FastAPI inference with metrics
* ✔ CI/CD & containerized deployment
* ✔ Benchmarking + performance analysis

It represents a **real-world MLOps design**, suitable for robotics, IoT, autonomous systems, and any latency-critical ML application.

---
