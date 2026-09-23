"""Pure helpers shared by live-camera detection and unit tests."""

from __future__ import annotations

from collections import Counter, deque
from typing import Iterable, Sequence

import cv2
import numpy as np

from .features import extract_features
from .preprocessing import preprocess_image


def select_best_detection(
    boxes: Iterable[tuple[float, tuple[int, int, int, int]]], threshold: float
) -> tuple[float, tuple[int, int, int, int]] | None:
    candidates = [item for item in boxes if item[0] >= threshold]
    return max(candidates, key=lambda item: item[0]) if candidates else None


def clip_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def crop_detection(frame: np.ndarray, box: tuple[int, int, int, int], margin: float = 0.10) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    margin_x = int(max(0, x2 - x1) * margin)
    margin_y = int(max(0, y2 - y1) * margin)
    clipped = clip_box((x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y), width, height)
    x1, y1, x2, y2 = clipped
    return frame[y1:y2, x1:x2]


def smooth_prediction(history: Sequence[str], label: str, window_size: int = 5) -> str:
    recent = list(history)[-(window_size - 1) :] + [label]
    counts = Counter(recent)
    return max(counts, key=lambda value: (counts[value], -recent[::-1].index(value)))


def classify_crop(crop: np.ndarray, classifier_artifact: dict) -> str:
    config = classifier_artifact["config"]
    image = preprocess_image(crop, config.image_size)
    vector = extract_features(image, classifier_artifact["feature_set"])
    return str(classifier_artifact["model"].predict([vector])[0])
