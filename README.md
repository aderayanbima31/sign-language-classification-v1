# BISINDO Classical Machine-Learning Gesture Recognition

This project is a learning-oriented classical computer-vision prototype for recognizing five static BISINDO letter gestures: **A, B, C, D, and E**. It uses image preprocessing, HOG features, and scikit-learn models. It does not translate complete words or sentences and does not use deep learning.

## PyCharm setup

Configure PyCharm to use the project interpreter at `.venv/bin/python`. In the PyCharm terminal, install dependencies and run the tests:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

## Dataset

The downloaded Kaggle dataset should contain `images/train`, `images/val`, `labels/train`, and `labels/val`, with one folder per letter. The dataset path is always supplied explicitly; it is not hard-coded in the source.

## Train Phase 1

```bash
python -m src.train \
  --dataset-root /path/to/bisindo \
  --classes A B C D E \
  --output-root outputs/phase1
```

The run compares resized-pixel KNN, HOG SVM, HOG Random Forest, and combined HOG + shape SVM. Results are written to `outputs/phase1/metrics`, the confusion matrix to `outputs/phase1/figures`, and the selected model to `outputs/phase1/models/best_model.joblib`.

## Predict one image

```bash
python -m src.predict \
  --model outputs/phase1/models/best_model.joblib \
  --image /path/to/image.jpg
```

The prediction command uses the saved model’s feature configuration and falls back to the full image because a new image has no dataset label file.

### Recommended terminal prediction model

For images without YOLO label files, train and use the full-image model so training and prediction use the same input format:

```bash
python -m src.train \
  --dataset-root /path/to/bisindo \
  --classes A B C D E \
  --output-root outputs/phase1_fullimage \
  --no-crop

python -m src.predict \
  --model outputs/phase1_fullimage/models/best_model.joblib \
  --image /path/to/image.jpg
```

The cropped model in `outputs/phase1` is useful for dataset-style experiments where a YOLO label is available. The full-image model is the correct model for ordinary terminal images and the future camera interface.

## Phase 2: all 26 letters

Keep Phase 1 results unchanged and use a separate output directory:

```bash
python -m src.train \
  --dataset-root /path/to/bisindo \
  --classes A B C D E F G H I J K L M N O P Q R S T U V W X Y Z \
  --output-root outputs/phase2
```

This remains static letter classification, not full sign-language translation.
