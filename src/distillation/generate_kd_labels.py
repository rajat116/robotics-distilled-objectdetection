import os
import shutil
from pathlib import Path

# ---------------------------------------------------------
# PATH RESOLUTION HELPERS
# ---------------------------------------------------------


def get_repo_root() -> Path:
    """
    Resolve repository root assuming this file is at:
        <repo>/src/distillation/generate_kd_labels.py
    """
    return Path(__file__).resolve().parents[2]


def resolve_data_root() -> Path:
    """
    Resolve COCO128 dataset root in a robust way.

    Priority:
    1) COCO128_PATH environment variable (if it exists on disk)
    2) <repo_root>/datasets/coco128
    3) /workspaces/datasets/coco128  (common in Codespaces)
    """
    env_path = os.getenv("COCO128_PATH")
    if env_path:
        env_root = Path(env_path).expanduser().resolve()
        if env_root.exists():
            print(f"[KD] Using COCO128_PATH env: {env_root}")
            return env_root
        else:
            print(f"[KD] WARNING: COCO128_PATH points to non-existent path: {env_root}")

    repo_root = get_repo_root()
    candidate1 = repo_root / "datasets" / "coco128"
    if candidate1.exists():
        print(f"[KD] Using dataset under repo: {candidate1}")
        return candidate1

    candidate2 = Path("/workspaces/datasets/coco128")
    if candidate2.exists():
        print(f"[KD] Using dataset at /workspaces/datasets/coco128: {candidate2}")
        return candidate2

    raise FileNotFoundError(
        "[KD] Could not locate COCO128 dataset.\n"
        "Tried:\n"
        f"  - COCO128_PATH env: {env_path}\n"
        f"  - {candidate1}\n"
        f"  - {candidate2}\n"
        "Please set COCO128_PATH or place the dataset in one of these locations."
    )


def resolve_teacher_weights() -> Path:
    """
    Resolve teacher weights path.

    Priority:
    1) TEACHER_WEIGHTS_PATH env var
    2) <repo_root>/runs/teacher/yolov8s_coco128/weights/best.pt
    """
    env_path = os.getenv("TEACHER_WEIGHTS_PATH")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.exists():
            print(f"[KD] Using TEACHER_WEIGHTS_PATH env: {p}")
            return p
        else:
            print(f"[KD] WARNING: TEACHER_WEIGHTS_PATH does not exist: {p}")

    repo_root = get_repo_root()
    default_path = (
        repo_root / "runs" / "teacher" / "yolov8s_coco128" / "weights" / "best.pt"
    )
    if default_path.exists():
        print(f"[KD] Using default teacher checkpoint: {default_path}")
        return default_path

    raise FileNotFoundError(
        "[KD] Could not find teacher weights.\n"
        "Tried:\n"
        f"  - TEACHER_WEIGHTS_PATH env: {env_path}\n"
        f"  - {default_path}\n"
        "Make sure you've trained the teacher and that best.pt exists."
    )


def resolve_distill_run_dir() -> Path:
    """
    Where to store teacher predictions for distillation.
    """
    repo_root = get_repo_root()
    distill_dir = repo_root / "runs" / "distill" / "coco128_train_kd"
    distill_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[KD] Distillation run directory: {distill_dir}")
    return distill_dir


def find_distill_labels_dir(distill_run_dir: Path) -> Path:
    """
    Find the folder that contains YOLO-generated label .txt files.
    Supports:
        <distill_run_dir>/labels
        <distill_run_dir>/predict/labels
    """
    candidates = [
        distill_run_dir / "labels",
        distill_run_dir / "predict" / "labels",
    ]

    for c in candidates:
        if c.exists() and any(c.glob("*.txt")):
            print(f"[KD] Using teacher label directory: {c}")
            return c

    tried_str = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(
        "[KD] Teacher-generated labels not found.\n"
        "Expected .txt files in one of:\n"
        f"{tried_str}\n"
        "Check the teacher.predict() call and where YOLO writes its labels."
    )


# ---------------------------------------------------------
# MAIN KD PIPELINE FUNCTIONS
# ---------------------------------------------------------

from ultralytics import YOLO  # noqa: E402  (import after path helpers)


def backup_ground_truth_labels(gt_labels: Path, backup_labels: Path):
    """
    Backup ground truth labels once, to allow restoring later if needed.
    """
    if not gt_labels.exists():
        raise FileNotFoundError(
            f"[KD] Ground truth labels folder not found: {gt_labels}"
        )

    if backup_labels.exists():
        print(f"[KD] Backup already exists at: {backup_labels}")
        return

    backup_root = backup_labels.parent
    backup_root.mkdir(parents=True, exist_ok=True)

    print(f"[KD] Backing up GT labels from {gt_labels} -> {backup_labels}")
    shutil.copytree(gt_labels, backup_labels)
    print("[KD] Backup complete.")


def run_teacher_predictions(
    teacher_weights: Path,
    train_images: Path,
    distill_run_dir: Path,
):
    """
    Run teacher model on training images and save YOLO-format labels.
    """
    if not teacher_weights.exists():
        raise FileNotFoundError(f"[KD] Teacher weights not found: {teacher_weights}")

    if not train_images.exists():
        raise FileNotFoundError(f"[KD] Train images folder not found: {train_images}")

    print(f"[KD] Loading teacher from: {teacher_weights}")
    teacher = YOLO(str(teacher_weights))

    print(f"[KD] Running teacher predictions on: {train_images}")
    # Note: Ultralytics will create <project>/<name>[/predict]/labels
    teacher.predict(
        source=str(train_images),
        save_txt=True,
        conf=0.3,
        project=str(distill_run_dir.parent),
        name=distill_run_dir.name,
        exist_ok=True,
        verbose=False,
    )

    # We don't assume exact subdir layout; we detect it:
    labels_dir = find_distill_labels_dir(distill_run_dir)
    print(f"[KD] Teacher labels written to: {labels_dir}")
    return labels_dir


def overwrite_labels_with_teacher(
    gt_labels: Path,
    distill_labels: Path,
):
    """
    Overwrite training GT labels with teacher-generated labels (KD).

    NOTE: This is destructive for the train split, but we assume GT
    has been backed up already under labels_gt_backup/.
    """
    print(f"[KD] Overwriting GT labels in: {gt_labels}")
    gt_labels.mkdir(parents=True, exist_ok=True)

    count = 0
    for txt_path in distill_labels.glob("*.txt"):
        target_path = gt_labels / txt_path.name
        shutil.copy2(txt_path, target_path)
        count += 1

    print(f"[KD] Copied {count} label files into {gt_labels}")
    if count == 0:
        print("[KD] WARNING: No .txt files were copied. Check teacher predictions.")


def main():
    print(">>> generate_kd_labels.py STARTED")

    # ---------- Resolve core paths ----------
    data_root = resolve_data_root()
    train_images = data_root / "images" / "train2017"
    gt_labels = data_root / "labels" / "train2017"
    backup_labels_root = data_root / "labels_gt_backup"
    backup_labels = backup_labels_root / "train2017"

    teacher_weights = resolve_teacher_weights()
    distill_run_dir = resolve_distill_run_dir()

    # ---------- Pipeline ----------
    backup_ground_truth_labels(gt_labels, backup_labels)
    distill_labels = run_teacher_predictions(
        teacher_weights=teacher_weights,
        train_images=train_images,
        distill_run_dir=distill_run_dir,
    )
    overwrite_labels_with_teacher(
        gt_labels=gt_labels,
        distill_labels=distill_labels,
    )

    print(
        ">>> generate_kd_labels.py DONE – dataset now uses teacher labels for train split."
    )


if __name__ == "__main__":
    main()
