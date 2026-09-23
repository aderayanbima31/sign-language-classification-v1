"""Convert the BISINDO class-folder data into a one-class YOLO dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable


def write_dataset_yaml(output_root: str | Path) -> Path:
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    yaml_path = output_root / "dataset.yaml"
    yaml_path.write_text(
        f"path: {output_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: [hand]\n"
    )
    return yaml_path


def convert_to_yolo_dataset(
    source_root: str | Path, output_root: str | Path, classes: Iterable[str]
) -> Path:
    source_root = Path(source_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    copied = 0
    skipped = []
    for split in ("train", "val"):
        for class_name in classes:
            image_dir = source_root / "images" / split / class_name
            for image_path in sorted(image_dir.glob("*.jpg")):
                label_path = source_root / "labels" / split / class_name / f"{image_path.stem}.txt"
                if not label_path.is_file():
                    skipped.append(str(image_path))
                    continue
                destination_image_dir = output_root / "images" / split
                destination_label_dir = output_root / "labels" / split
                destination_image_dir.mkdir(parents=True, exist_ok=True)
                destination_label_dir.mkdir(parents=True, exist_ok=True)
                destination_stem = f"{class_name}_{image_path.stem}"
                shutil.copy2(image_path, destination_image_dir / f"{destination_stem}{image_path.suffix.lower()}")
                normalized_rows = []
                for line in label_path.read_text().splitlines():
                    fields = line.split()
                    if len(fields) >= 5:
                        normalized_rows.append("0 " + " ".join(fields[1:5]))
                if not normalized_rows:
                    skipped.append(str(image_path))
                    continue
                (destination_label_dir / f"{destination_stem}.txt").write_text("\n".join(normalized_rows) + "\n")
                copied += 1
    write_dataset_yaml(output_root)
    (output_root / "conversion_report.txt").write_text(
        f"copied_images={copied}\nskipped_missing_labels={len(skipped)}\n"
        + "\n".join(skipped)
    )
    if copied == 0:
        raise FileNotFoundError(f"no labeled JPG images found under {source_root}")
    return output_root


def _parse_args():
    parser = argparse.ArgumentParser(description="Convert BISINDO annotations for one-class YOLO training")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--classes", nargs="+", default=list("ABCDE"))
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = convert_to_yolo_dataset(args.source_root, args.output_root, args.classes)
    print(f"YOLO dataset written to {result}")
