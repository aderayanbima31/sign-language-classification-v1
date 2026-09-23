import numpy as np

from src.live_pipeline import crop_detection, select_best_detection, smooth_prediction


def test_select_best_detection_ignores_low_confidence_and_picks_highest():
    boxes = [(0.4, (1, 1, 10, 10)), (0.9, (2, 2, 20, 20))]
    assert select_best_detection(boxes, threshold=0.5) == (0.9, (2, 2, 20, 20))


def test_no_detection_returns_none():
    assert select_best_detection([], threshold=0.5) is None


def test_crop_detection_is_clipped_to_frame():
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    crop = crop_detection(frame, (-5, -4, 40, 25), margin=0)
    assert crop.shape[:2] == (20, 30)


def test_smoothing_returns_majority_label():
    assert smooth_prediction(["A", "B"], "A", window_size=3) == "A"
