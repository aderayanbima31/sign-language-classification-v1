# Real-Time BISINDO Camera Recognition with YOLO and HOG-SVM

## Status

Approved conversational design; awaiting written-spec review before implementation.

## Goal

Extend the existing BISINDO A–E prototype with a real-time webcam pipeline. A YOLO hand detector will locate the hand in each frame, then the existing classical HOG + SVM classifier will classify the cropped hand gesture.

## Scope change

The original project is classical machine learning. This branch intentionally adds one deep-learning component, YOLO, only for hand localization. HOG + SVM remains the letter classifier. Results from this branch must not overwrite `outputs/phase1` or `outputs/phase1_fullimage`.

## Existing assets

The dataset already contains:

```text
images/{train,val}/{A..Z}/*.jpg
labels/{train,val}/{A..Z}/*.txt
```

The label rows use YOLO normalized bounding-box format and currently identify one object class (`0`, hand). The detector dataset will be generated into a separate YOLO layout with one `images/` and one `labels/` directory per split and a single `hand` class.

## Architecture

```text
webcam frame
  → YOLO hand detector
  → highest-confidence hand bounding box
  → safe crop with margin
  → grayscale/resize preprocessing
  → HOG feature extraction
  → cropped Phase 1 HOG-SVM model
  → smoothed A–E prediction and overlay
```

The detector and classifier are separate artifacts. The detector may find no hand; in that case the UI displays `No hand detected` and does not call the classifier. If multiple hands are detected, the highest-confidence detection is selected and the behavior is documented as a Phase 1 limitation.

## Components

- `src/yolo_dataset.py`: convert the downloaded class-folder dataset into a one-class YOLO dataset while preserving image/label pairs.
- `src/train_detector.py`: train a YOLO hand detector from the generated dataset and save its weights under a branch-specific output directory.
- `src/live_camera.py`: open the default webcam, run detector inference, crop the best hand, call the existing HOG-SVM classifier, draw the box and prediction, and exit on `q` or Escape.
- `src/live_pipeline.py`: small testable functions for selecting detections, cropping frames, and smoothing predictions.
- `tests/`: unit tests for dataset conversion, detection selection, crop boundaries, and prediction smoothing. Hardware-dependent webcam capture is not tested in CI; a synthetic frame and fake detector are used instead.

## Training flow

1. Build the detector dataset from the downloaded BISINDO root with a configurable source path.
2. Use a fixed train/validation split and single class name `hand`.
3. Train a lightweight YOLO model for a bounded number of epochs using a branch-specific output directory.
4. Inspect detector validation metrics and sample predictions.
5. Run the live camera using the detector weights and `outputs/phase1/models/best_model.joblib` as the classifier.

The detector is not trained to classify A–E. Letter recognition remains the responsibility of the HOG-SVM artifact.

## Runtime behavior

- Default camera index is `0`, configurable by CLI.
- Detector confidence threshold is configurable, with a documented default.
- The classifier runs only when a hand detection passes the threshold.
- A short majority-vote history reduces flicker; the displayed label is the most common recent prediction.
- `q` and Escape close the window cleanly.
- Camera-open failure produces an actionable error explaining how to choose another camera index.
- Missing detector weights or classifier weights fail before opening the camera.

## Success criteria

1. Detector dataset conversion produces valid one-class YOLO files from the downloaded dataset.
2. Detector training completes and saves weights without modifying Phase 1 artifacts.
3. A synthetic-frame test proves the highest-confidence detection is cropped within frame boundaries.
4. The live-camera command opens the camera when hardware is available and exits cleanly on `q`.
5. A detected hand is passed through the existing cropped HOG-SVM classifier and displays an A–E prediction.
6. No-hand frames do not produce a stale or fabricated classifier result.

## Limitations

- The first version assumes one primary hand region.
- Camera lighting, distance, motion blur, and background can reduce detector and classifier accuracy.
- The classifier recognizes only A–E until a separate A–Z model is trained.
- YOLO adds a deep-learning dependency and GPU is optional; CPU inference may be slower.
