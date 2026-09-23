import pytest

from src.dataset import clip_box, make_splits, parse_yolo_box


def test_parse_yolo_box_converts_normalized_coordinates():
    box = parse_yolo_box("0 0.5 0.5 0.25 0.4", width=640, height=640)
    assert box == (240, 192, 400, 448)


def test_clip_box_keeps_box_inside_image():
    assert clip_box((-4, 10, 700, 650), width=640, height=640) == (0, 10, 640, 640)


def test_split_is_repeatable_and_contains_every_class():
    samples = [object() for _ in range(50)]
    labels = ["A", "B", "C", "D", "E"] * 10
    train_a, val_a, test_a = make_splits(samples, labels, random_seed=42)
    train_b, val_b, test_b = make_splits(samples, labels, random_seed=42)
    assert train_a == train_b and val_a == val_b and test_a == test_b
    assert set(labels[index] for index in test_a) == set("ABCDE")


def test_invalid_label_line_is_rejected():
    with pytest.raises(ValueError, match="YOLO"):
        parse_yolo_box("not-a-yolo-row", width=640, height=640)


def test_split_rejects_a_class_with_only_one_sample():
    with pytest.raises(ValueError, match="at least two"):
        make_splits([object(), object(), object()], ["A", "A", "B"], random_seed=42)
