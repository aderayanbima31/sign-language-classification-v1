# YOLO Live Camera BISINDO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real-time webcam pipeline in the `improvement-use-live-camera` branch that detects a hand with YOLO and classifies the cropped hand with the existing HOG + SVM BISINDO model.

**Architecture:** Convert the existing hand bounding-box annotations into a one-class YOLO dataset, train a lightweight Ultralytics detector, and keep detector and classifier artifacts separate. A testable pipeline layer will select the best detection, crop safely, and smooth classifier labels; the camera UI will only coordinate OpenCV capture and display.

**Tech Stack:** Python 3.9, OpenCV, NumPy, pandas, scikit-learn, scikit-image, joblib, pytest, Ultralytics YOLO.

**Spec:** `docs/superpowers/specs/2026-09-24-live-camera-yolo-design.md`

## Global Constraints

- This branch adds YOLO only for hand localization; HOG + SVM remains the A–E gesture classifier.
- Existing `outputs/phase1` and `outputs/phase1_fullimage` artifacts must not be overwritten.
- The detector has one class named `hand`; it does not classify A–E.
- Dataset conversion must preserve source images and labels and write only under `outputs/live_camera/`.
- The camera must not call the classifier when no detection passes the confidence threshold.
- The default camera index is `0`; confidence threshold, camera index, detector weights, and classifier weights are CLI-configurable.
- The first version selects the highest-confidence hand when multiple hands are detected and documents that limitation.
- YOLO training uses the Ultralytics Python API with a pretrained lightweight checkpoint; the official API supports custom `model.train(data=..., epochs=...)` and webcam/video prediction sources. [Ultralytics Python usage](https://docs.ultralytics.com/usage/python)

## Review Focus

- Source images whose label file is missing: conversion must skip and report them rather than create invalid detector data — covered by Task 1 conversion tests.
- Normalized boxes touching image boundaries: conversion must preserve valid YOLO coordinates and live cropping must remain in frame — covered by Task 1 and Task 3 tests.
- Frames with zero detections: the live pipeline must return no prediction instead of reusing a stale label — covered by Task 3 tests.
- Frames with multiple hands: the highest-confidence detection must be selected deterministically — covered by Task 3 tests.
- Missing model files or unavailable cameras: CLI must fail with actionable errors before entering the loop — covered by Task 4 tests and manual CLI verification.

---

### Task 1: YOLO dependency and detector-dataset conversion

**Files:**
- Modify: `requirements.txt`
- Create: `src/yolo_dataset.py`
- Create: `tests/test_yolo_dataset.py`

**Interfaces:**
- Consumes the external BISINDO dataset root and existing `Sample`/YOLO parsing conventions.
- Produces `convert_to_yolo_dataset(source_root, output_root, classes) -> Path` and `write_dataset_yaml(output_root) -> Path`.

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

from src.yolo_dataset import convert_to_yolo_dataset, write_dataset_yaml


def test_conversion_creates_one_class_yolo_layout(tmp_path):
    source = tmp_path / "source"
    image_dir = source / "images" / "train" / "A"
    label_dir = source / "labels" / "train" / "A"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / "one.jpg").write_bytes(b"jpeg")
    (label_dir / "one.txt").write_text("0 0.5 0.5 0.5 0.5\n")

    output = convert_to_yolo_dataset(source, tmp_path / "detector", ("A",))

    assert (output / "images" / "train" / "one.jpg").is_file()
    assert (output / "labels" / "train" / "one.txt").read_text() == "0 0.5 0.5 0.5 0.5\n"


def test_dataset_yaml_declares_one_hand_class(tmp_path):
    path = write_dataset_yaml(tmp_path / "detector")
    assert "nc: 1" in path.read_text()
    assert "hand" in path.read_text()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/test_yolo_dataset.py -q`

Expected: FAIL because `src.yolo_dataset` does not exist.

- [ ] **Step 3: Implement conversion and dependency declaration**

Add `ultralytics` to `requirements.txt`. Copy only images from configured source classes and their matching label files into `outputs/live_camera/detector_dataset/images/{train,val}` and `labels/{train,val}`. Skip missing label pairs with a returned report, preserve normalized coordinates, and write:

```yaml
path: /absolute/path/to/outputs/live_camera/detector_dataset
train: images/train
val: images/val
nc: 1
names: [hand]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_yolo_dataset.py -q`

Expected: PASS with two tests passing.

- [ ] **Step 5: Convert the real A–E dataset**

Run: `./.venv/bin/python -m src.yolo_dataset --source-root /Users/alamsyahd/Downloads/bisindo --output-root outputs/live_camera/detector_dataset --classes A B C D E`

Expected: a detector dataset and `dataset.yaml` are created under `outputs/live_camera/detector_dataset`; source files remain unchanged.

### Task 2: YOLO detector training command

**Files:**
- Create: `src/train_detector.py`
- Create: `tests/test_train_detector.py`

**Interfaces:**
- Consumes the dataset YAML from Task 1.
- Produces `train_detector(dataset_yaml, output_root, epochs, imgsz) -> Path` returning the best `.pt` weights path.

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

from src.train_detector import detector_output_path


def test_detector_output_path_is_branch_specific(tmp_path):
    path = detector_output_path(tmp_path)
    assert path == tmp_path / "weights" / "best.pt"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_train_detector.py -q`

Expected: FAIL because `src.train_detector` does not exist.

- [ ] **Step 3: Implement the detector training wrapper**

Use the Ultralytics Python API:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.train(
    data=str(dataset_yaml),
    epochs=epochs,
    imgsz=imgsz,
    project=str(output_root),
    name="hand_detector",
    exist_ok=True,
)
```

Expose CLI arguments for `--data`, `--output-root`, `--epochs` (default 20), and `--imgsz` (default 640). Return `hand_detector/weights/best.pt` and fail clearly if it does not exist after training.

- [ ] **Step 4: Run unit tests and CLI help**

Run: `./.venv/bin/python -m pytest tests/test_train_detector.py -q && ./.venv/bin/python -m src.train_detector --help`

Expected: one test passes and help shows data, output, epochs, and image-size arguments.

- [ ] **Step 5: Train the real detector**

Run: `./.venv/bin/python -m src.train_detector --data outputs/live_camera/detector_dataset/dataset.yaml --output-root outputs/live_camera/detector_training --epochs 20 --imgsz 640`

Expected: Ultralytics completes training and saves `outputs/live_camera/detector_training/hand_detector/weights/best.pt`.

### Task 3: Testable live detection and classification pipeline

**Files:**
- Create: `src/live_pipeline.py`
- Create: `tests/test_live_pipeline.py`

**Interfaces:**
- Consumes OpenCV frames, YOLO result boxes, and the existing classifier artifact.
- Produces `select_best_detection(boxes, threshold)`, `crop_detection(frame, box, margin)`, and `smooth_prediction(history, label, window_size)`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from src.live_pipeline import crop_detection, select_best_detection, smooth_prediction


def test_select_best_detection_ignores_low_confidence_and_picks_highest():
    boxes = [(0.4, (1, 1, 10, 10)), (0.9, (2, 2, 20, 20))]
    assert select_best_detection(boxes, threshold=0.5) == (0.9, (2, 2, 20, 20))


def test_no_detection_returns_none():
    assert select_best_detection([], threshold=0.5) is None


def test_crop_detection_is_clipped_to_frame():
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    crop = crop_detection(frame, (-5, -4, 40, 25), margin=0)
    assert crop.shape[:2] == (20, 30)


def test_smoothing_returns_majority_label():
    assert smooth_prediction(["A", "B", "A"], "B", window_size=3) == "A"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/test_live_pipeline.py -q`

Expected: FAIL because `src.live_pipeline` does not exist.

- [ ] **Step 3: Implement pure pipeline helpers and model inference**

Represent boxes as `(confidence, (x1, y1, x2, y2))`, clip coordinates to frame dimensions, return `None` when no confidence passes the threshold, and use a deque for bounded prediction history. Add a `classify_crop(crop, classifier_artifact)` helper that calls the existing preprocessing and HOG feature functions with the cropped model’s saved configuration.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_live_pipeline.py -q`

Expected: PASS with four tests passing.

### Task 4: Webcam application

**Files:**
- Create: `src/live_camera.py`
- Create: `tests/test_live_camera.py`

**Interfaces:**
- Consumes detector weights, classifier weights, camera index, and confidence threshold.
- Produces a CLI command `python -m src.live_camera --detector ... --classifier ...`.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from src.live_camera import validate_runtime_paths


def test_missing_runtime_model_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="detector"):
        validate_runtime_paths(tmp_path / "missing-detector.pt", tmp_path / "classifier.joblib")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_live_camera.py -q`

Expected: FAIL because `src.live_camera` does not exist.

- [ ] **Step 3: Implement the camera loop**

Load `YOLO(detector_path)` and `joblib.load(classifier_path)`, open `cv2.VideoCapture(camera_index)`, run the detector on each frame, select the best hand, crop it, classify it with the cropped HOG-SVM artifact, and draw the box, confidence, and smoothed letter. Show `No hand detected` when appropriate. Exit on `q` or Escape, release the camera, and destroy windows in `finally`.

Use CLI arguments:

```text
--detector PATH
--classifier PATH
--camera-index INTEGER (default 0)
--confidence FLOAT (default 0.50)
--window-size INTEGER (default 5)
```

- [ ] **Step 4: Run tests and help**

Run: `./.venv/bin/python -m pytest tests/test_live_camera.py -q && ./.venv/bin/python -m src.live_camera --help`

Expected: one test passes and help displays all runtime arguments.

- [ ] **Step 5: Run a real camera smoke test**

Run: `./.venv/bin/python -m src.live_camera --detector outputs/live_camera/detector_training/hand_detector/weights/best.pt --classifier outputs/phase1/models/best_model.joblib`

Expected: a camera window opens, a detected hand shows a box and an A–E label, and pressing `q` exits cleanly.

### Task 5: Documentation and final verification

**Files:**
- Modify: `README.md`
- Create: `docs/live_camera.md`
- Create: `tests/test_docs.py`
- Modify: `requirements.txt`

**Interfaces:**
- Documents the exact PyCharm terminal setup, dataset conversion, detector training, and live-camera command.

- [ ] **Step 1: Write the documentation verification test**

```python
from pathlib import Path


def test_live_camera_documentation_contains_run_command():
    text = Path("docs/live_camera.md").read_text()
    assert "src.live_camera" in text
    assert "--detector" in text
    assert "--classifier" in text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_docs.py -q`

Expected: FAIL because the live-camera documentation and test do not yet exist.

- [ ] **Step 3: Write usage and limitations documentation**

Document that YOLO detects the hand and HOG + SVM classifies A–E, that `outputs/phase1` is the cropped classifier, that GPU is optional, that the camera index may need adjustment, and that only one highest-confidence hand is used in this first version. Add the exact commands from Tasks 1, 2, and 4 to `README.md`.

- [ ] **Step 4: Run the complete verification suite**

Run: `./.venv/bin/python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Record final artifacts**

Record detector weights, classifier weights, detector training metrics, camera command, and known limitations in `docs/live_camera.md`. Confirm that `outputs/phase1` and `outputs/phase1_fullimage` are unchanged.
