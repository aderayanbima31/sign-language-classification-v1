# BISINDO Classical ML Gesture Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a configurable Python classical-ML pipeline that classifies BISINDO gesture images A–E with HOG and comparison models, saves metrics and a model, and can later expand to A–Z.

**Architecture:** Small Python modules separate dataset discovery, image preprocessing, feature extraction, training, evaluation, and prediction. A shared configuration controls the external dataset path, class list, image size, random seed, and output directories. scikit-learn pipelines keep scaling and model selection leakage-safe.

**Tech Stack:** Python 3, OpenCV, NumPy, pandas, scikit-image, scikit-learn, matplotlib, seaborn, joblib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-23-bisindo-classical-ml-design.md`

## Global Constraints

- Phase 1 uses only classes `A`, `B`, `C`, `D`, and `E`; Phase 2 changes configuration to all available letters.
- The dataset path must be configurable and must not hard-code `/Users/alamsyahd/Downloads/bisindo`.
- The implementation uses classical computer vision and machine learning only; no CNN, YOLO training, transfer learning, or pretrained neural network.
- Images are cropped from valid YOLO boxes when available, with a documented full-image fallback.
- SVM and KNN preprocessing uses scikit-learn `Pipeline`; the final test set is not used for fitting, scaling, tuning, or model selection.
- All experiment outputs must include the class list, seed, preprocessing, feature configuration, model parameters, split sizes, and metrics.
- The existing `main.py` starter remains usable only as a compatibility entry point; project behavior moves into `src/` modules.
- Git is not initialized in this workspace, so no commit commands are required; each task ends with a test checkpoint and a clean status note.

## Review Focus

- Missing or malformed YOLO labels: the loader must report the sample and use the explicit fallback policy rather than crash — covered by Task 2 label tests.
- Class leakage between source partitions and final test data: the split must be deterministic and stratified — covered by Task 2 split tests.
- Feature dimensions changing between training and prediction: feature extractors must return stable vectors — covered by Task 3 feature tests.
- A model trained with too few classes or unreadable images: training must fail with a clear error before fitting — covered by Task 4 training tests.
- Output directories or model files missing at prediction time: the CLI must return a readable error and non-zero exit code — covered by Task 5 CLI tests.

---

### Task 1: Project scaffolding and configuration

**Files:**
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `requirements.txt`
- Create: `pytest.ini`
- Create: `tests/test_config.py`
- Modify: `main.py`

**Interfaces:**
- Produces `ProjectConfig`, `DEFAULT_CLASSES`, `build_config()`, and stable output-path fields consumed by all later tasks.

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

from src.config import DEFAULT_CLASSES, build_config


def test_default_config_uses_five_phase_one_classes(tmp_path):
    config = build_config(dataset_root=tmp_path)
    assert config.classes == ("A", "B", "C", "D", "E")
    assert config.image_size == (128, 128)
    assert config.random_seed == 42


def test_config_paths_are_under_configured_output_root(tmp_path):
    config = build_config(dataset_root=tmp_path, output_root=tmp_path / "run")
    assert config.metrics_dir == tmp_path / "run" / "metrics"
    assert config.models_dir == tmp_path / "run" / "models"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_config.py -q`

Expected: FAIL because `src.config` and `build_config` do not yet exist.

- [ ] **Step 3: Implement the minimal configuration layer**

Create a frozen dataclass with `dataset_root`, `classes`, `image_size`, `random_seed`, `test_size`, `validation_size`, `output_root`, `metrics_dir`, `figures_dir`, `models_dir`, and `predictions_dir`. Make `build_config()` resolve paths and create no directories until a command needs them. Add the required package dependencies to `requirements.txt` and configure pytest to import from the project root.

Update `main.py` to expose a small help-oriented entry point that imports no training side effects:

```python
from src.config import DEFAULT_CLASSES


if __name__ == "__main__":
    print(f"BISINDO classifier ready for classes: {', '.join(DEFAULT_CLASSES)}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_config.py -q`

Expected: PASS with two tests passing.

- [ ] **Step 5: Record the checkpoint**

Run: `git status --short 2>&1 || true` and record in the progress ledger that Task 1 passed; do not claim a commit because this directory is not a Git repository.

### Task 2: Dataset discovery, YOLO parsing, and deterministic splits

**Files:**
- Create: `src/dataset.py`
- Create: `tests/test_dataset.py`

**Interfaces:**
- Consumes `ProjectConfig` from Task 1.
- Produces `Sample(image_path: Path, label_path: Path | None, class_name: str, split: str)`, `discover_samples(config)`, `parse_yolo_box(line, width, height)`, `clip_box(box, width, height)`, and `make_splits(samples, labels, random_seed=42)`.

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

import pytest

from src.dataset import clip_box, make_splits, parse_yolo_box


def test_parse_yolo_box_converts_normalized_coordinates():
    box = parse_yolo_box("0 0.5 0.5 0.25 0.4", width=640, height=640)
    assert box == (240, 192, 400, 448)


def test_clip_box_keeps_box_inside_image():
    assert clip_box((-4, 10, 700, 650), width=640, height=640) == (0, 10, 640, 640)


def test_split_is_repeatable_and_contains_every_class(tmp_path):
    samples = [object() for _ in range(20)]
    labels = ["A", "B", "C", "D", "E"] * 4
    train_a, val_a, test_a = make_splits(samples, labels, random_seed=42)
    train_b, val_b, test_b = make_splits(samples, labels, random_seed=42)
    assert train_a == train_b and val_a == val_b and test_a == test_b
    assert set(labels[index] for index in test_a) == set("ABCDE")


def test_invalid_label_line_is_rejected():
    with pytest.raises(ValueError, match="YOLO"):
        parse_yolo_box("not-a-yolo-row", width=640, height=640)


def test_split_rejects_a_class_with_only_one_sample():
    with pytest.raises(ValueError, match="at least two"):
        make_splits([object(), object(), object()], ["A", "A", "B"], random_seed=42)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_dataset.py -q`

Expected: FAIL because the dataset module and its functions do not exist.

- [ ] **Step 3: Implement discovery and split logic**

Discover only configured classes below `dataset_root/images/train` and `dataset_root/images/val`. Pair each JPG with the matching label file under `labels`; preserve missing-label samples with `label_path=None` so validation can report them. Parse the first YOLO row, convert normalized center-width-height values to integer pixel coordinates, reject non-finite or non-positive boxes, and clip valid boxes. Use `train_test_split(..., stratify=labels, random_state=...)` twice to make train/validation/test indexes, with clear errors when a class has insufficient samples.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_dataset.py -q`

Expected: PASS with five tests passing.

- [ ] **Step 5: Validate against the downloaded dataset**

Run: `python -c "from src.config import build_config; from src.dataset import discover_samples; c=build_config(dataset_root='/Users/alamsyahd/Downloads/bisindo'); print({s: sum(x.split == s for x in discover_samples(c)) for s in ('train','val')})"`

Expected: the command reports both source partitions and A–E samples without crashing; missing or invalid files are reported through the returned validation information.

### Task 3: Image preprocessing and feature extraction

**Files:**
- Create: `src/preprocessing.py`
- Create: `src/features.py`
- Create: `tests/test_features.py`

**Interfaces:**
- Consumes `Sample` and parsed boxes from Task 2.
- Produces `load_and_preprocess(sample, config, use_crop=True) -> np.ndarray`, `extract_hog(image) -> np.ndarray`, `extract_hu_contour_features(image) -> np.ndarray`, and `extract_features(image, feature_set) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

```python
import cv2
import numpy as np

from src.features import extract_features, extract_hog
from src.preprocessing import preprocess_image


def test_preprocess_returns_fixed_grayscale_shape():
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    result = preprocess_image(image, image_size=(128, 128))
    assert result.shape == (128, 128)
    assert result.dtype == np.uint8


def test_hog_vector_length_is_stable():
    image = np.zeros((128, 128), dtype=np.uint8)
    assert extract_hog(image).shape == extract_hog(image).shape


def test_combined_features_are_longer_than_hog():
    image = np.zeros((128, 128), dtype=np.uint8)
    assert extract_features(image, "hog+shape").size > extract_features(image, "hog").size
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_features.py -q`

Expected: FAIL because preprocessing and feature functions do not exist.

- [ ] **Step 3: Implement preprocessing and features**

Load images with OpenCV, optionally crop the clipped YOLO box with a 10% margin, fall back to the full image when cropping is disabled or the label is absent, resize to `(128, 128)`, and convert BGR to grayscale. Extract HOG with fixed parameters (`orientations=9`, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)`, `block_norm='L2-Hys'`). Add seven Hu Moments in log-scaled form plus contour area, perimeter, aspect ratio, extent, and solidity. Flatten outputs to finite `float32` vectors and raise a clear error for unreadable images.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_features.py -q`

Expected: PASS with three tests passing.

- [ ] **Step 5: Add crop fallback coverage**

Extend the tests with a temporary JPG and a missing label path, then assert `load_and_preprocess(..., use_crop=True)` returns the same fixed shape as the full-image path and does not crash. Run `pytest tests/test_features.py -q` and expect all tests to pass.

### Task 4: Model training and evaluation artifacts

**Files:**
- Create: `src/train.py`
- Create: `src/evaluate.py`
- Create: `tests/test_training.py`

**Interfaces:**
- Consumes `ProjectConfig`, `Sample`, preprocessing, and feature functions from Tasks 1–3.
- Produces `run_experiments(config) -> pandas.DataFrame`, `build_model_pipeline(model_name, feature_set, config)`, `evaluate_predictions(y_true, y_pred, class_names) -> dict`, and saved `.joblib`, `.json`, and `.png` artifacts.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from src.evaluate import evaluate_predictions
from src.train import build_model_pipeline


def test_evaluation_contains_required_metrics():
    result = evaluate_predictions(["A", "B", "A"], ["A", "B", "B"], ["A", "B"])
    assert {"accuracy", "macro_f1", "classification_report", "confusion_matrix"} <= result.keys()
    assert np.asarray(result["confusion_matrix"]).shape == (2, 2)


def test_svm_pipeline_contains_scaler_and_classifier():
    pipeline = build_model_pipeline("svm", "hog")
    assert "scaler" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_training_rejects_fewer_than_two_classes():
    from src.train import validate_training_labels

    with pytest.raises(ValueError, match="two classes"):
        validate_training_labels(["A", "A"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_training.py -q`

Expected: FAIL because training and evaluation modules do not exist.

- [ ] **Step 3: Implement pipelines, experiments, and reports**

Implement E0 resized pixels + KNN, E1 HOG + tuned RBF SVM, E2 HOG + Random Forest, and E3 HOG + shape + tuned RBF SVM. Fit only on the training split, use validation data for model comparison, and evaluate the selected winner once on the final test split. Use macro F1 for SVM `GridSearchCV`, fixed seed 42, and a smaller CV fold count automatically when a class has fewer than five training samples. Save each experiment’s metrics as JSON, aggregate results as CSV, the final model with joblib, and a labeled confusion matrix PNG.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_training.py -q`

Expected: PASS with three tests passing.

- [ ] **Step 5: Run the real Phase 1 training job**

Run: `python -m src.train --dataset-root /Users/alamsyahd/Downloads/bisindo --classes A B C D E --output-root outputs/phase1`

Expected: training completes, creates an aggregate metrics CSV, per-experiment JSON files, a confusion matrix figure, and a saved best model under `outputs/phase1`.

### Task 5: Prediction CLI and project documentation

**Files:**
- Create: `src/predict.py`
- Create: `tests/test_predict.py`
- Create: `README.md`
- Create: `docs/dataset_card.md`
- Create: `docs/experiment_log.md`
- Modify: `src/train.py`

**Interfaces:**
- Consumes the saved joblib model from Task 4.
- Produces `predict_image(model_path, image_path) -> dict` and command-line entry points for training and prediction.

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

import pytest

from src.predict import predict_image


def test_missing_model_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="model"):
        predict_image(tmp_path / "missing.joblib", tmp_path / "image.jpg")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_predict.py -q`

Expected: FAIL because `src.predict` does not exist.

- [ ] **Step 3: Implement prediction and documentation**

Load the saved artifact, preprocess the input using the artifact’s stored configuration, return the predicted class and optional class probabilities, and emit readable errors for missing files. Add commands to the README:

```bash
python -m src.train --dataset-root /path/to/bisindo --classes A B C D E --output-root outputs/phase1
python -m src.predict --model outputs/phase1/models/best_model.joblib --image /path/to/image.jpg
pytest -q
```

Document that the first phase recognizes five static BISINDO letter gestures and is not full sentence translation. The dataset card must record the external source path, 26 available classes, Phase 1 subset, image format, YOLO labels, preprocessing, and known limitations. The experiment log must provide one row per experiment and leave result fields populated by the real run.

- [ ] **Step 4: Run tests and CLI validation**

Run: `pytest -q`

Expected: all unit and smoke tests pass.

Run: `python -m src.predict --help`

Expected: help text shows required model and image arguments.

- [ ] **Step 5: Verify a real prediction**

Run: `python -m src.predict --model outputs/phase1/models/best_model.joblib --image /Users/alamsyahd/Downloads/bisindo/images/val/A/$(find /Users/alamsyahd/Downloads/bisindo/images/val/A -type f -name '*.jpg' | sed 's#.*/##' | head -1)`

Expected: a JSON-like result reports one class from `A`, `B`, `C`, `D`, or `E` and the model artifact loads successfully.

### Task 6: Final verification and Phase 2 readiness

**Files:**
- Modify: `README.md`
- Modify: `docs/experiment_log.md`
- Modify: `src/config.py`
- Create: `tests/test_phase2_config.py`

**Interfaces:**
- Consumes the complete Phase 1 artifacts and test results.
- Produces an explicit, non-destructive A–Z configuration path that does not overwrite Phase 1 outputs.

- [ ] **Step 1: Write the failing test**

```python
from src.config import build_config


def test_all_class_configuration_is_separate_from_phase_one(tmp_path):
    config = build_config(dataset_root=tmp_path, classes=tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), output_root=tmp_path / "phase2")
    assert len(config.classes) == 26
    assert config.output_root.name == "phase2"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_phase2_config.py -q`

Expected: FAIL if `build_config` does not accept an explicit class tuple.

- [ ] **Step 3: Implement the explicit class override and final documentation**

Allow `build_config(classes=...)` to accept a tuple of uppercase labels, keep `DEFAULT_CLASSES` as A–E, and document the A–Z command with a separate `outputs/phase2` directory. Add the actual Phase 1 metrics, strongest confusion pair, and limitations after the real run; do not invent results.

- [ ] **Step 4: Run the complete verification suite**

Run: `pytest -q`

Expected: all tests pass.

Run: `python -m src.train --help && python -m src.predict --help`

Expected: both CLIs print usage successfully.

- [ ] **Step 5: Record final checkpoint**

Record the test command, dataset run command, artifact paths, and any warnings in the project notes. If Git is initialized later, the owner can commit the complete implementation as one or more normal commits.
