"""Fixed-length classical visual feature extractors."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog


def extract_hog(image: np.ndarray) -> np.ndarray:
    # A 64x64 working image keeps the classical baseline fast while the
    # preprocessing contract remains a 128x128 image for inspection.
    working_image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    vector = hog(
        working_image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return np.asarray(vector, dtype=np.float32)


def extract_hu_contour_features(image: np.ndarray) -> np.ndarray:
    _, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea) if contours else None

    if contour is None or cv2.contourArea(contour) <= 0:
        shape = np.zeros(5, dtype=np.float32)
        moments = np.zeros(7, dtype=np.float32)
    else:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        x, y, width, height = cv2.boundingRect(contour)
        rect_area = float(max(1, width * height))
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        shape = np.array(
            [
                area / (image.shape[0] * image.shape[1]),
                perimeter / (2 * (image.shape[0] + image.shape[1])),
                width / max(1, height),
                area / rect_area,
                area / max(1.0, hull_area),
            ],
            dtype=np.float32,
        )
        moments = cv2.HuMoments(cv2.moments(contour)).flatten().astype(np.float32)
    moments = (-np.sign(moments) * np.log10(np.abs(moments) + 1e-12)).astype(np.float32)
    return np.concatenate([moments, shape]).astype(np.float32)


def extract_features(image: np.ndarray, feature_set: str) -> np.ndarray:
    if feature_set == "pixels":
        vector = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1) / 255.0
    elif feature_set == "hog":
        vector = extract_hog(image)
    elif feature_set == "hog+shape":
        vector = np.concatenate([extract_hog(image), extract_hu_contour_features(image)])
    else:
        raise ValueError(f"unknown feature set: {feature_set}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("feature vector contains non-finite values")
    return np.asarray(vector, dtype=np.float32)
