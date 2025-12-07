import os

from ultralytics import YOLO

MODEL_PATHS = {
    "teacher": "runs/teacher/yolov8s_coco128/weights/best.pt",
    "student": "runs/student/yolov8n_coco128_student/weights/best.pt",
    "student_kd": "runs/student/yolov8n_coco128_student_KD/weights/best.pt",
}

DEFAULT_MODEL = "student_kd"  # best performing model


def load_model():
    name = os.getenv("MODEL_NAME", DEFAULT_MODEL)
    if name not in MODEL_PATHS:
        raise ValueError(
            f"Unknown MODEL_NAME={name}. Choose from {list(MODEL_PATHS.keys())}"
        )

    weights = MODEL_PATHS[name]
    print(f"[model_loader] Loading model '{name}' → {weights}")

    model = YOLO(weights)

    def predictor(image_bytes):
        return model.predict(image_bytes, imgsz=320, verbose=False)

    return predictor
