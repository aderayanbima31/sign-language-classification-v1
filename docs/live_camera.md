# Live BISINDO Camera Recognition

This branch combines two models:

```text
YOLO hand detector → hand crop → HOG + SVM letter classifier
```

YOLO detects the hand location. The existing cropped HOG-SVM model predicts one of `A`, `B`, `C`, `D`, or `E`.

## One-time setup

Use the PyCharm project interpreter `.venv/bin/python` and install dependencies:

```bash
python -m pip install -r requirements.txt
```

Convert the downloaded annotations into a one-class detector dataset:

```bash
python -m src.yolo_dataset \
  --source-root /Users/alamsyahd/Downloads/bisindo \
  --output-root outputs/live_camera/detector_dataset_v2 \
  --classes A B C D E
```

Train the lightweight detector. The CPU-friendly defaults are 3 epochs, 320px images, and 50% of the training data:

```bash
python -m src.train_detector \
  --data outputs/live_camera/detector_dataset_v2/dataset.yaml \
  --output-root outputs/live_camera/detector_training_v3
```

The detector weights are saved at:

```text
outputs/live_camera/detector_training_v3/hand_detector/weights/best.pt
```

## Run the camera

```bash
python -m src.live_camera \
  --detector outputs/live_camera/detector_training_v3/hand_detector/weights/best.pt \
  --classifier outputs/phase1/models/best_model.joblib
```

The window shows the hand bounding box, detector confidence, and smoothed A–E prediction. Press `q` or Escape to exit.

If the default camera does not open, try another index:

```bash
python -m src.live_camera \
  --detector outputs/live_camera/detector_training_v3/hand_detector/weights/best.pt \
  --classifier outputs/phase1/models/best_model.joblib \
  --camera-index 1
```

On macOS, allow PyCharm or the terminal to access the camera in **System Settings → Privacy & Security → Camera**.

## Limitations

- The detector selects the highest-confidence hand when multiple hands are visible.
- The classifier currently recognizes A–E only.
- The first detector smoke training uses CPU-friendly settings; more epochs and the full dataset can improve robustness but take longer on CPU.
