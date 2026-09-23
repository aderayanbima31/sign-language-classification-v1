"""Image loading, optional YOLO cropping, and normalization."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import ProjectConfig
from .dataset import Sample, parse_yolo_box


def preprocess_image(image: np.ndarray, image_size: tuple[int, int] = (128, 128)) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("cannot preprocess an empty image")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8, copy=False)


def _read_first_label(label_path: Path, width: int, height: int) -> tuple[int, int, int, int]:
    try:
        line = next(line for line in label_path.read_text().splitlines() if line.strip())
    except (OSError, StopIteration) as exc:
        raise ValueError(f"could not read a YOLO label from {label_path}") from exc
    return parse_yolo_box(line, width, height)


def _crop_with_margin(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    margin_x = int((x2 - x1) * 0.10)
    margin_y = int((y2 - y1) * 0.10)
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(width, x2 + margin_x)
    y2 = min(height, y2 + margin_y)
    return image[y1:y2, x1:x2]


def load_and_preprocess(sample: Sample, config: ProjectConfig, use_crop: bool = True) -> np.ndarray:
    image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read image: {sample.image_path}")
    if use_crop and sample.label_path is not None:
        try:
            box = _read_first_label(sample.label_path, image.shape[1], image.shape[0])
            cropped = _crop_with_margin(image, box)
            if cropped.size:
                image = cropped
        except ValueError:
            pass
    return preprocess_image(image, config.image_size)
