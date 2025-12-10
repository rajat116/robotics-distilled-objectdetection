"""
Generate inference images for teacher, student, and KD-student models.

Outputs are saved into ./figures/ as:
  - input_image.jpg
  - detection_teacher.jpg
  - detection_student.jpg
  - detection_student_kd.jpg
"""

import os
from ultralytics import YOLO
import shutil

# ------------------------------
# CONFIG
# ------------------------------
INPUT_IMAGE = "bus.jpg"  # you can change this
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

MODELS = {
    "teacher": "models/teacher/best.pt",
    "student": "models/student/best.pt",
    "student_kd": "models/student_kd/best.pt",
}


# ------------------------------
# COPY ORIGINAL IMAGE
# ------------------------------
def save_input_image():
    dst = os.path.join(FIG_DIR, "input_image.jpg")
    shutil.copy(INPUT_IMAGE, dst)
    print(f"✔ Saved original image → {dst}")


# ------------------------------
# RUN INFERENCE & SAVE OUTPUTS
# ------------------------------
def run_inference():
    for name, path in MODELS.items():
        print(f"\n🔵 Running inference for: {name} ({path})")

        model = YOLO(path)

        # Run inference (save=True writes to runs/detect/predict/)
        results = model.predict(source=INPUT_IMAGE, save=True, imgsz=640)

        # YOLO saved output inside: runs/detect/predict/
        pred_dir = results[0].save_dir
        pred_file = os.path.join(pred_dir, os.listdir(pred_dir)[0])  # first image

        # Copy to figures/
        out_path = os.path.join(FIG_DIR, f"detection_{name}.jpg")
        shutil.copy(pred_file, out_path)

        print(f"✔ Saved detection → {out_path}")


# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    save_input_image()
    run_inference()
    print("\n🎉 All inference images generated in ./figures/")
