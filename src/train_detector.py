"""Train a lightweight one-class YOLO hand detector."""

from __future__ import annotations

import argparse
from pathlib import Path


def detector_output_path(output_root: str | Path) -> Path:
    return Path(output_root) / "weights" / "best.pt"


def train_detector(
    dataset_yaml: str | Path,
    output_root: str | Path,
    *,
    epochs: int = 3,
    imgsz: int = 320,
    fraction: float = 0.5,
) -> Path:
    from ultralytics import YOLO

    output_root = Path(output_root).expanduser().resolve()
    model = YOLO("yolo26n.pt")
    model.train(
        data=str(Path(dataset_yaml).expanduser().resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=32,
        workers=0,
        fraction=fraction,
        project=str(output_root),
        name="hand_detector",
        exist_ok=True,
    )
    best_path = output_root / "hand_detector" / "weights" / "best.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"YOLO training finished without weights at {best_path}")
    return best_path


def _parse_args():
    parser = argparse.ArgumentParser(description="Train the one-class YOLO hand detector")
    parser.add_argument("--data", required=True, help="path to detector dataset.yaml")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--fraction", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(train_detector(args.data, args.output_root, epochs=args.epochs, imgsz=args.imgsz, fraction=args.fraction))
