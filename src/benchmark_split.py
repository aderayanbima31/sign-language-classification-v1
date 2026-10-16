"""Reproducible dataset manifest generation for the quantum benchmark."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

from sklearn.model_selection import train_test_split


DEFAULT_CLASSES = ("A", "B", "C", "D", "E")
DEFAULT_PARTITIONS = ("train", "val")
DEFAULT_SEED = 42
DEFAULT_TEST_FRACTION = 0.15
DEFAULT_VALIDATION_FRACTION = 0.15


def discover_sample_ids(
    dataset_root: str | Path,
    *,
    classes: Iterable[str] = DEFAULT_CLASSES,
    partitions: Iterable[str] = DEFAULT_PARTITIONS,
    extensions: Iterable[str] = (".jpg",),
) -> list[dict[str, str]]:
    """Discover image identifiers in the protocol's canonical order.

    Discovery is intentionally limited to paths and parent-directory labels. It
    does not read images, labels, or fit any preprocessing.
    """

    root = Path(dataset_root).expanduser().resolve()
    class_names = tuple(classes)
    partition_names = tuple(partitions)
    normalized_extensions = {extension.lower() for extension in extensions}
    if not class_names:
        raise ValueError("classes must not be empty")
    if not normalized_extensions:
        raise ValueError("extensions must not be empty")

    samples: list[dict[str, str]] = []
    for partition in partition_names:
        for class_name in class_names:
            image_dir = root / "images" / partition / class_name
            if not image_dir.is_dir():
                continue
            for image_path in image_dir.rglob("*"):
                if image_path.is_file() and image_path.suffix.lower() in normalized_extensions:
                    samples.append(
                        {
                            "id": image_path.relative_to(root).as_posix(),
                            "label": class_name,
                        }
                    )

    samples.sort(key=lambda sample: sample["id"])
    if not samples:
        raise FileNotFoundError(
            f"no benchmark images found under {root / 'images'} for classes {class_names}"
        )
    missing_classes = sorted(set(class_names) - {sample["label"] for sample in samples})
    if missing_classes:
        raise ValueError(f"dataset is missing configured classes: {missing_classes}")
    if len({sample["id"] for sample in samples}) != len(samples):
        raise ValueError("dataset contains duplicate sample identifiers")
    return samples


def _split_indices(
    samples: Sequence[dict[str, str]],
    *,
    seed: int,
    test_fraction: float,
    validation_fraction: float,
) -> tuple[list[int], list[int], list[int]]:
    """Apply the protocol's 85/15 then 70/15 stratified split."""

    if not 0 < test_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between 0 and 1")
    if test_fraction + validation_fraction >= 1:
        raise ValueError("test and validation fractions must sum to less than 1")

    labels = [sample["label"] for sample in samples]
    class_names = tuple(dict.fromkeys(labels))
    class_count = len(class_names)
    counts = {label: labels.count(label) for label in class_names}
    if class_count < 2:
        raise ValueError("at least two classes are required for stratification")

    test_size = _split_size(len(samples), test_fraction)
    train_validation_size = len(samples) - test_size
    validation_within_remainder = validation_fraction / (1 - test_fraction)
    validation_size = _split_size(train_validation_size, validation_within_remainder)
    train_size = train_validation_size - validation_size

    # The first stage must leave at least two examples per class: the second
    # stage needs one for validation and one for the final training split.
    # These bounds are intentionally checked before sklearn so small datasets
    # fail with an actionable protocol error instead of an opaque splitter
    # exception.
    if test_size < class_count or train_validation_size < 2 * class_count:
        raise ValueError(
            "dataset is too small for the two-stage 70/15/15 split: "
            f"need at least {class_count} test and {2 * class_count} "
            "train/validation samples"
        )
    if validation_size < class_count or train_size < class_count:
        raise ValueError(
            "dataset is too small for the two-stage 70/15/15 split: "
            f"the second stage needs at least {class_count} validation and "
            f"{class_count} training samples"
        )
    if any(count < 3 for count in counts.values()):
        raise ValueError(
            "each class must have at least three samples for every class to "
            "appear in train, validation, and test"
        )
    if any(
        _split_size(len(samples), test_fraction * count / len(samples)) > count - 2
        for count in counts.values()
    ):
        raise ValueError(
            "class counts cannot retain two samples after the first "
            "stratified split; add more samples to the smallest class"
        )

    indices = list(range(len(samples)))
    train_validation, test = train_test_split(
        indices,
        test_size=test_fraction,
        random_state=seed,
        stratify=labels,
    )
    train_validation_labels = [labels[index] for index in train_validation]
    train, validation = train_test_split(
        train_validation,
        test_size=validation_within_remainder,
        random_state=seed,
        stratify=train_validation_labels,
    )
    split_indices = (
        sorted(train),
        sorted(validation),
        sorted(test),
    )
    split_names = ("train", "validation", "test")
    for split_name, split in zip(split_names, split_indices):
        split_labels = {labels[index] for index in split}
        if split_labels != set(class_names):
            raise ValueError(
                "two-stage stratified split failed class coverage: "
                f"{split_name} has {sorted(split_labels)}; expected {list(class_names)}"
            )
    return split_indices


def _split_size(sample_count: int, fraction: float) -> int:
    """Match sklearn's float ``test_size`` rule (ceil of the requested size)."""

    return math.ceil(sample_count * fraction)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def manifest_sha256(manifest: dict[str, object]) -> str:
    """Return the hash of a manifest excluding its self-referential hash field."""

    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def create_manifest(
    dataset_root: str | Path,
    *,
    classes: Iterable[str] = DEFAULT_CLASSES,
    seed: int = DEFAULT_SEED,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    partitions: Iterable[str] = DEFAULT_PARTITIONS,
    extensions: Iterable[str] = (".jpg",),
) -> dict[str, object]:
    """Create the shared benchmark split manifest without preprocessing."""

    root = Path(dataset_root).expanduser().resolve()
    class_names = list(classes)
    partition_names = list(partitions)
    samples = discover_sample_ids(
        root,
        classes=class_names,
        partitions=partition_names,
        extensions=extensions,
    )
    train_indices, validation_indices, test_indices = _split_indices(
        samples,
        seed=seed,
        test_fraction=test_fraction,
        validation_fraction=validation_fraction,
    )

    inventory = {
        "classes": class_names,
        "partitions": partition_names,
        "samples": samples,
    }
    inventory_hash = hashlib.sha256(_canonical_json(inventory)).hexdigest()

    splits = {
        "train": [samples[index] for index in train_indices],
        "validation": [samples[index] for index in validation_indices],
        "test": [samples[index] for index in test_indices],
    }
    manifest: dict[str, object] = {
        "schema_version": "q2.1",
        "dataset_identity": {
            "classes": class_names,
            "source_partitions": partition_names,
            "sample_count": len(samples),
            "inventory_sha256": inventory_hash,
            # Persist the canonical inventory so downstream validation can
            # prove that split IDs and labels are the original dataset's
            # records, rather than merely unique records with a valid hash.
            "samples": samples,
        },
        "seed": seed,
        "split_policy": {
            "strategy": "two_stage_stratified_split",
            "train_fraction": 1 - test_fraction - validation_fraction,
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
            "first_stage_train_validation_fraction": 1 - test_fraction,
            "second_stage_validation_fraction_of_remainder": validation_fraction / (1 - test_fraction),
        },
        "classes": class_names,
        "counts": {name: len(records) for name, records in splits.items()},
        "splits": splits,
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    return manifest


def write_manifest(manifest: dict[str, object], output_path: str | Path) -> Path:
    """Write a validated manifest and return its path."""

    expected_hash = manifest_sha256(manifest)
    if manifest.get("manifest_sha256") != expected_hash:
        raise ValueError("manifest_sha256 does not match manifest contents")

    _validate_manifest(manifest)
    requested_destination = Path(output_path).expanduser()
    destination = requested_destination.resolve(strict=False)
    protected_parts = {"phase1", "phase1_fullimage"}
    if any(
        part in protected_parts
        for path in (requested_destination, destination)
        for part in path.parts
    ):
        raise ValueError("benchmark manifests cannot be written under protected Phase 1 outputs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return destination


def _validate_manifest(manifest: dict[str, object]) -> None:
    """Validate the structural and split invariants consumed by later tickets."""

    classes = manifest.get("classes")
    splits = manifest.get("splits")
    counts = manifest.get("counts")
    identity = manifest.get("dataset_identity")
    if not isinstance(classes, list) or not classes or len(set(classes)) != len(classes):
        raise ValueError("manifest classes must be a non-empty list of unique names")
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
        raise ValueError("manifest splits must contain train, validation, and test")
    if not isinstance(counts, dict) or set(counts) != set(splits):
        raise ValueError("manifest counts must match its split names")
    if not isinstance(identity, dict) or identity.get("classes") != classes:
        raise ValueError("manifest dataset identity does not match classes")
    sample_count = identity.get("sample_count")
    if not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError("manifest dataset identity must contain a positive sample_count")
    inventory_samples = identity.get("samples")
    if not isinstance(inventory_samples, list) or len(inventory_samples) != sample_count:
        raise ValueError("manifest dataset identity must contain the complete sample inventory")
    if any(
        not isinstance(record, dict)
        or not isinstance(record.get("id"), str)
        or record.get("label") not in set(classes)
        for record in inventory_samples
    ):
        raise ValueError("manifest dataset identity contains an invalid sample inventory")
    if len({record["id"] for record in inventory_samples}) != sample_count:
        raise ValueError("manifest dataset identity contains duplicate sample identifiers")
    inventory = {
        "classes": classes,
        "partitions": identity.get("source_partitions"),
        "samples": inventory_samples,
    }
    expected_inventory_hash = hashlib.sha256(_canonical_json(inventory)).hexdigest()
    if identity.get("inventory_sha256") != expected_inventory_hash:
        raise ValueError("manifest dataset identity inventory hash does not match samples")

    all_ids: list[str] = []
    expected_classes = set(classes)
    for split_name, records in splits.items():
        if not isinstance(records, list) or counts[split_name] != len(records):
            raise ValueError(f"manifest count does not match {split_name} records")
        split_ids: set[str] = set()
        split_labels: set[str] = set()
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("id"), str):
                raise ValueError(f"manifest {split_name} contains an invalid sample record")
            label = record.get("label")
            if label not in expected_classes:
                raise ValueError(f"manifest {split_name} contains an unknown class label")
            sample_id = record["id"]
            if sample_id in split_ids:
                raise ValueError("manifest contains duplicate sample identifiers")
            split_ids.add(sample_id)
            split_labels.add(label)
        if split_labels != expected_classes:
            raise ValueError(f"manifest {split_name} must contain every configured class")
        all_ids.extend(split_ids)

    if len(all_ids) != sample_count or len(set(all_ids)) != sample_count:
        raise ValueError("manifest splits must be disjoint and cover the dataset exactly once")

    inventory_by_id = {record["id"]: record for record in inventory_samples}
    split_records = [
        record
        for split_name in ("train", "validation", "test")
        for record in splits[split_name]
    ]
    if {record["id"] for record in split_records} != set(inventory_by_id):
        raise ValueError("manifest splits must contain exactly the inventory sample identifiers")
    if any(inventory_by_id[record["id"]] != record for record in split_records):
        raise ValueError("manifest split records must match the inventory IDs and labels")
