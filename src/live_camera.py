"""Real-time YOLO hand detection plus HOG-SVM BISINDO classification."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2
import joblib
from ultralytics import YOLO

from .live_pipeline import classify_crop, crop_detection, select_best_detection, smooth_prediction


def validate_runtime_paths(detector_path: str | Path, classifier_path: str | Path) -> None:
    if not Path(detector_path).is_file():
        raise FileNotFoundError(f"detector weights not found: {detector_path}")
    if not Path(classifier_path).is_file():
        raise FileNotFoundError(f"classifier artifact not found: {classifier_path}")


def _result_boxes(result) -> list[tuple[float, tuple[int, int, int, int]]]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = []
    for coordinates, confidence in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
        x1, y1, x2, y2 = (int(value) for value in coordinates)
        boxes.append((float(confidence), (x1, y1, x2, y2)))
    return boxes


def run_camera(
    detector_path: str | Path,
    classifier_path: str | Path,
    *,
    camera_index: int = 0,
    confidence: float = 0.50,
    window_size: int = 5,
) -> None:
    validate_runtime_paths(detector_path, classifier_path)
    detector = YOLO(str(detector_path))
    classifier_artifact = joblib.load(classifier_path)
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"could not open camera index {camera_index}; try --camera-index 1 or check PyCharm camera permissions"
        )

    history = deque(maxlen=window_size)
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("could not read a frame from the camera")
            result = detector(frame, verbose=False)[0]
            selected = select_best_detection(_result_boxes(result), confidence)
            if selected is None:
                cv2.putText(frame, "No hand detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                score, box = selected
                x1, y1, x2, y2 = box
                crop = crop_detection(frame, box)
                label = classify_crop(crop, classifier_artifact)
                history.append(label)
                displayed_label = smooth_prediction(history, label, window_size)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{displayed_label} ({score:.2f})",
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            cv2.putText(frame, "Press q or Esc to quit", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("BISINDO Live Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def _parse_args():
    parser = argparse.ArgumentParser(description="Detect a hand and classify BISINDO A-E from a webcam")
    parser.add_argument("--detector", required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--window-size", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_camera(
        args.detector,
        args.classifier,
        camera_index=args.camera_index,
        confidence=args.confidence,
        window_size=args.window_size,
    )
