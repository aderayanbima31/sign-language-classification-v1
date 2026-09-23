"""Shared configuration for the BISINDO classical-ML pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_CLASSES = ("A", "B", "C", "D", "E")


@dataclass(frozen=True)
class ProjectConfig:
    dataset_root: Path
    classes: tuple[str, ...] = DEFAULT_CLASSES
    image_size: tuple[int, int] = (128, 128)
    random_seed: int = 42
    test_size: float = 0.15
    validation_size: float = 0.15
    use_crop: bool = True
    output_root: Path = Path("outputs/phase1")
    metrics_dir: Path = Path("outputs/phase1/metrics")
    figures_dir: Path = Path("outputs/phase1/figures")
    models_dir: Path = Path("outputs/phase1/models")
    predictions_dir: Path = Path("outputs/phase1/predictions")


def build_config(
    dataset_root: str | Path,
    *,
    classes: Iterable[str] = DEFAULT_CLASSES,
    image_size: tuple[int, int] = (128, 128),
    random_seed: int = 42,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    use_crop: bool = True,
    output_root: str | Path = Path("outputs/phase1"),
) -> ProjectConfig:
    """Build a normalized configuration without creating output directories."""

    root = Path(dataset_root).expanduser().resolve()
    output = Path(output_root).expanduser()
    class_tuple = tuple(label.upper() for label in classes)
    if not class_tuple or any(len(label) != 1 or not label.isalpha() for label in class_tuple):
        raise ValueError("classes must contain one-letter alphabetic labels")
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("test_size and validation_size must be between 0 and 1")
    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be less than 1")

    return ProjectConfig(
        dataset_root=root,
        classes=class_tuple,
        image_size=image_size,
        random_seed=random_seed,
        test_size=test_size,
        validation_size=validation_size,
        use_crop=use_crop,
        output_root=output,
        metrics_dir=output / "metrics",
        figures_dir=output / "figures",
        models_dir=output / "models",
        predictions_dir=output / "predictions",
    )
