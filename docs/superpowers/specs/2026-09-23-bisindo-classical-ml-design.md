# BISINDO Classical Machine-Learning Gesture Recognition

## Status

Approved design for Phase 1 implementation.

## Goal

Build a reproducible Python prototype that recognizes five static BISINDO hand-gesture classes (`A`, `B`, `C`, `D`, and `E`) from images using classical computer vision and machine learning. After the pipeline is verified, expand the same system to all 26 letter classes.

## User intent and learning goals

The project is for a master's student learning classical machine learning with Python. The implementation should make the full workflow visible and understandable: dataset inspection, preprocessing, feature engineering, model training, evaluation, error analysis, and single-image prediction. Deep-learning methods are explicitly outside scope.

## Dataset assumptions

The current BISINDO dataset is located outside the repository and has this structure:

```text
bisindo/
├── images/
│   ├── train/{A..Z}/*.jpg
│   └── val/{A..Z}/*.jpg
└── labels/
    ├── train/{A..Z}/*.txt
    └── val/{A..Z}/*.txt
```

Images are 640×640 JPG files. Label files contain YOLO-style rows with normalized bounding boxes. The project must not silently depend on the absolute Downloads path; the dataset path must be configurable through a CLI argument or configuration value.

Phase 1 uses only classes `A` through `E`. Phase 2 will use all available letter classes without redesigning the pipeline.

## Scope

### Included

- Static image classification, not continuous video translation.
- Folder-based class discovery with an explicit class allow-list.
- Optional hand cropping from the corresponding YOLO bounding box.
- Grayscale resize preprocessing.
- HOG shape features as the primary representation.
- Optional Hu Moments and contour descriptors.
- KNN, RBF SVM, and Random Forest comparison models.
- Stratified validation and an untouched test evaluation derived from the supplied train/validation data.
- Accuracy, per-class precision/recall, macro F1, confusion matrix, prediction examples, and saved model artifacts.
- A simple prediction command for one image.
- Dataset card and experiment log documentation.

### Excluded

- CNNs, YOLO training, transfer learning, neural networks, or pretrained models.
- Full words, sentences, continuous gesture sequences, or production deployment.
- Mixing BISINDO and SIBI labels.
- Claiming that letter classification is complete sign-language translation.

## Architecture

The repository will be organized into focused modules:

```text
sign-language-classification/
├── data/
│   ├── raw/              # optional local copy or link instructions
│   └── processed/        # generated inspection artifacts only
├── docs/
│   ├── dataset_card.md
│   ├── experiment_log.md
│   └── superpowers/
│       ├── specs/
│       └── plans/
├── notebooks/
├── models/
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── predictions/
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
├── requirements.txt
└── README.md
```

The data module discovers image/label pairs and records missing, unreadable, and invalid samples. The preprocessing module crops a valid bounding box, adds a small safe margin, resizes to a fixed square, and converts to grayscale. The feature module provides HOG and optional shape descriptors. Training code owns model pipelines and cross-validation; evaluation code owns metrics and plots; prediction code loads a saved artifact and returns the predicted class with confidence information when available.

## Data flow

```text
configured dataset path
  → discover A–E image/label pairs
  → validate files and bounding boxes
  → crop hand region with fallback
  → resize and grayscale
  → extract HOG or combined features
  → stratified train/validation/test split
  → fit leakage-safe model pipeline
  → evaluate and save metrics/figures/model
  → predict one new image
```

The supplied `train` and `val` directories will be treated as source partitions. The implementation must create a final untouched test subset from the available data using a fixed random seed, while preserving class balance. No scaler, feature normalizer, or model-selection decision may use the final test subset.

## Experiments

Phase 1 will run these experiments:

| ID | Features | Model | Purpose |
|---|---|---|---|
| E0 | resized grayscale pixels | KNN | simple baseline |
| E1 | HOG | RBF SVM | primary benchmark |
| E2 | HOG | Random Forest | nonlinear comparison |
| E3 | HOG + Hu Moments + contour descriptors | RBF SVM | feature-combination comparison |

SVM and KNN preprocessing must use scikit-learn `Pipeline` so learned scaling is fitted only within training folds. SVM tuning uses macro F1 as the primary score and searches `C ∈ {0.1, 1, 10, 100}` and `gamma ∈ {scale, 0.001, 0.01, 0.1}` with stratified five-fold cross-validation when the available sample count permits it.

## Error handling and data quality

- Missing label files are reported and the image is excluded from cropped-image experiments unless the full-image fallback is explicitly selected.
- Missing or unreadable images are reported without stopping the entire dataset scan.
- Invalid or out-of-range boxes are rejected and recorded; valid boxes are clipped to image boundaries.
- Empty classes or fewer than two samples per class cause a clear configuration error before training.
- Class imbalance is reported rather than hidden.
- Every experiment records its class list, random seed, preprocessing, feature configuration, model parameters, split sizes, and metrics.

## Evaluation and success criteria

The first phase is successful when:

1. The dataset loader discovers and validates A–E samples.
2. At least one baseline trains and predicts without data leakage.
3. All planned models produce a comparable metrics table.
4. The final test report includes accuracy, macro F1, per-class precision/recall, and a confusion matrix.
5. A saved best-model artifact can predict a new JPG image from the command line.
6. Documentation clearly states that the system recognizes five selected static BISINDO letter gestures.
7. The class configuration can be changed to A–Z for Phase 2.

The project must not promise a target accuracy before running the experiments. Results will be interpreted together with confusion pairs, background and crop quality, and the fact that the dataset may not represent unseen participants.

## Testing strategy

- Unit tests for class discovery, label parsing, bounding-box clipping, fallback behavior, preprocessing shape, and feature vector dimensions.
- Deterministic split tests confirming class presence and repeatability.
- Training smoke tests on a small synthetic image fixture to verify pipelines and saved artifacts.
- Evaluation tests confirming required metric keys and confusion-matrix dimensions.
- A final end-to-end run on the downloaded BISINDO dataset after implementation.

## Phase 2 expansion

After Phase 1 is working and documented, change the class configuration from `A-E` to `A-Z`, rerun the same experiment matrix, and update the dataset card and experiment log. Phase 2 is a separate experiment and must not overwrite Phase 1 results.

