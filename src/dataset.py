"""Dataset discovery and safe split helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.model_selection import train_test_split

from .config import ProjectConfig


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_path: Path | None
    class_name: str
    split: str


def parse_yolo_box(line: str, width: int, height: int) -> tuple[int, int, int, int]:
    """Convert the first normalized YOLO row to clipped pixel coordinates."""

    parts = line.strip().split()
    if len(parts) < 5:
        raise ValueError("invalid YOLO label: expected class, center x/y, width, height")
    try:
        _, center_x, center_y, box_width, box_height = map(float, parts[:5])
    except ValueError as exc:
        raise ValueError("invalid YOLO label values") from exc
    values = (center_x, center_y, box_width, box_height)
    if not np.all(np.isfinite(values)) or box_width <= 0 or box_height <= 0:
        raise ValueError("invalid YOLO label values")

    x1 = int(round((center_x - box_width / 2) * width))
    y1 = int(round((center_y - box_height / 2) * height))
    x2 = int(round((center_x + box_width / 2) * width))
    y2 = int(round((center_y + box_height / 2) * height))
    clipped = clip_box((x1, y1, x2, y2), width, height)
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError("invalid YOLO label: empty box")
    return clipped


def clip_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def _matching_label(dataset_root: Path, split: str, class_name: str, image_path: Path) -> Path | None:
    label_path = dataset_root / "labels" / split / class_name / f"{image_path.stem}.txt"
    return label_path if label_path.is_file() else None


def discover_samples(config: ProjectConfig) -> list[Sample]:
    """Discover configured classes from both source partitions."""

    samples: list[Sample] = []
    for split in ("train", "val"):
        for class_name in config.classes:
            image_dir = config.dataset_root / "images" / split / class_name
            if not image_dir.is_dir():
                continue
            for image_path in sorted(image_dir.glob("*.jpg")):
                samples.append(
                    Sample(
                        image_path=image_path,
                        label_path=_matching_label(config.dataset_root, split, class_name, image_path),
                        class_name=class_name,
                        split=split,
                    )
                )
    if not samples:
        raise FileNotFoundError(f"no JPG samples found for classes {config.classes} under {config.dataset_root}")
    return samples


def make_splits(
    samples: Sequence[object],
    labels: Sequence[str],
    *,
    random_seed: int = 42,
    test_size: float = 0.15,
    validation_size: float = 0.15,
) -> tuple[list[int], list[int], list[int]]:
    """Return deterministic train, validation, and test indexes."""

    if len(samples) != len(labels):
        raise ValueError("samples and labels must have the same length")
    counts = {label: labels.count(label) for label in set(labels)}
    if any(count < 2 for count in counts.values()):
        raise ValueError("each class must have at least two samples")

    indexes = np.arange(len(samples))
    train_val, test = train_test_split(
        indexes,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels,
    )
    train_val_labels = [labels[index] for index in train_val]
    validation_fraction = validation_size / (1 - test_size)
    train, validation = train_test_split(
        train_val,
        test_size=validation_fraction,
        random_state=random_seed,
        stratify=train_val_labels,
    )
    return sorted(train.tolist()), sorted(validation.tolist()), sorted(test.tolist())
