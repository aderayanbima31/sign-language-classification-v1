# Dataset Card — BISINDO Static Gesture Images

## Identity and scope

The project uses a Kaggle BISINDO image dataset organized by letter folders. The labels are treated as static gesture classes under one sign system, BISINDO. Phase 1 intentionally uses only A–E for learning and debugging; all 26 letters are reserved for Phase 2.

## Structure observed

- Images: JPEG, observed size 640×640.
- Source partitions: `images/train/{A..Z}` and `images/val/{A..Z}`.
- Object labels: matching YOLO-style text files under `labels/train` and `labels/val`.
- Phase 1 observation: 2,182 A–E images were discovered, with class counts between 429 and 441.

## Processing

Valid YOLO boxes are clipped and expanded with a small margin before preprocessing. Missing, malformed, or unreadable labels use the documented full-image fallback. Images are converted to grayscale and resized to 128×128. HOG is calculated on a 64×64 working image to keep the classical baseline practical; the pixel baseline uses a 32×32 working image.

## Limitations

The dataset’s participant identities and collection conditions are not fully documented in this repository. The final test is a stratified image-level split, not a guaranteed participant-held-out test. The high Phase 1 score should therefore not be interpreted as general BISINDO translation accuracy. The system recognizes selected static letter gestures only.
