"""Run the manifest-bound classical baseline for the quantum benchmark.

This module deliberately keeps the comparison runner separate from the Phase 1
training command.  The Q2 manifest is the only source of split membership;
preprocessing and model selection are fitted from the training records only.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Iterable

import joblib
from sklearn.model_selection import GridSearchCV

from .benchmark_split import _validate_manifest, manifest_sha256
from .config import ProjectConfig
from .dataset import Sample
from .evaluate import evaluate_predictions
from .features import extract_features
from .preprocessing import load_and_preprocess
from .train import build_model_pipeline


RUN_ID = "phase1_e1_reproduced"
PROTECTED_OUTPUTS = {"phase1", "phase1_fullimage"}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read benchmark manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("benchmark manifest must contain a JSON object")
    return payload


def load_manifest(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load and validate a Q2 manifest, returning it and its locked hash."""

    manifest_path = Path(path).expanduser().resolve()
    manifest = _read_json(manifest_path)
    expected_hash = manifest_sha256(manifest)
    if manifest.get("manifest_sha256") != expected_hash:
        raise ValueError("manifest_sha256 does not match manifest contents")
    _validate_manifest(manifest)
    return manifest, expected_hash


def _safe_output_dir(path: str | Path) -> Path:
    requested = Path(path).expanduser()
    resolved = requested.resolve(strict=False)
    if any(part in PROTECTED_OUTPUTS for candidate in (requested, resolved) for part in candidate.parts):
        raise ValueError("quantum comparison artifacts cannot be written under protected Phase 1 outputs")
    repository_output_root = (Path(__file__).resolve().parents[1] / "outputs" / "quantum_comparison").resolve(strict=False)
    try:
        resolved.relative_to(repository_output_root)
    except ValueError as exc:
        raise ValueError("classical comparison output must be under outputs/quantum_comparison") from exc
    return resolved


def _sample_from_record(dataset_root: Path, record: dict[str, str]) -> Sample:
    sample_id = record["id"]
    image_path = (dataset_root / sample_id).resolve()
    if not image_path.is_file() or dataset_root not in image_path.parents:
        raise FileNotFoundError(f"manifest image does not exist under dataset root: {sample_id}")
    parts = Path(sample_id).parts
    if len(parts) < 4 or parts[0] != "images":
        raise ValueError(f"manifest sample ID is not a supported image path: {sample_id}")
    source_partition, class_name = parts[1], parts[2]
    label_path = dataset_root / "labels" / source_partition / class_name / f"{image_path.stem}.txt"
    return Sample(
        image_path=image_path,
        label_path=label_path if label_path.is_file() else None,
        class_name=record["label"],
        split=source_partition,
    )


def _records(manifest: dict[str, Any], split_name: str) -> list[dict[str, str]]:
    records = manifest["splits"][split_name]
    return [dict(record) for record in records]


def _feature_matrix(
    dataset_root: Path,
    records: Iterable[dict[str, str]],
    config: ProjectConfig,
) -> list[Any]:
    matrix = []
    for record in records:
        sample = _sample_from_record(dataset_root, record)
        image = load_and_preprocess(sample, config, use_crop=True)
        matrix.append(extract_features(image, "hog"))
    return matrix


def _fit_baseline(x_train, y_train, seed: int):
    pipeline = build_model_pipeline("svm", "hog", ProjectConfig(dataset_root=Path("."), random_seed=seed))
    counts = {label: y_train.count(label) for label in set(y_train)}
    cv_folds = min(5, min(counts.values()))
    if cv_folds < 2:
        raise ValueError("classical comparison requires at least two training samples per class")
    search = GridSearchCV(
        pipeline,
        {
            "model__C": [0.1, 1, 10, 100],
            "model__gamma": ["scale", 0.001, 0.01, 0.1],
        },
        scoring="f1_macro",
        cv=cv_folds,
        n_jobs=1,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_, dict(search.best_params_), cv_folds


def _fit_selected(x_train, y_train, best_params: dict[str, Any], seed: int):
    """Fit the train-only selected hyperparameters without re-searching."""

    pipeline = build_model_pipeline("svm", "hog", ProjectConfig(dataset_root=Path("."), random_seed=seed))
    pipeline.set_params(**best_params)
    pipeline.fit(x_train, y_train)
    return pipeline


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value)!r}")


def _write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "sample_id", "true_label", "predicted_label"])
        writer.writeheader()
        writer.writerows(rows)


def run_classical_comparison(
    dataset_root: str | Path,
    manifest_path: str | Path,
    *,
    output_root: str | Path = "outputs/quantum_comparison/classical",
) -> dict[str, Any]:
    """Train, select, refit, and evaluate the manifest-bound HOG/RBF-SVM."""

    dataset = Path(dataset_root).expanduser().resolve()
    output = _safe_output_dir(output_root)
    manifest, manifest_hash = load_manifest(manifest_path)
    classes = tuple(manifest["classes"])
    seed = int(manifest["seed"])
    config = ProjectConfig(dataset_root=dataset, classes=classes, random_seed=seed, use_crop=True)

    train_records = _records(manifest, "train")
    validation_records = _records(manifest, "validation")
    test_records = _records(manifest, "test")
    train_features = _feature_matrix(dataset, train_records, config)
    validation_features = _feature_matrix(dataset, validation_records, config)
    test_features = _feature_matrix(dataset, test_records, config)
    train_labels = [record["label"] for record in train_records]
    validation_labels = [record["label"] for record in validation_records]
    test_labels = [record["label"] for record in test_records]

    fit_started = time.perf_counter()
    selected_model, best_params, cv_folds = _fit_baseline(train_features, train_labels, seed)
    fit_runtime = time.perf_counter() - fit_started
    predict_started = time.perf_counter()
    validation_predictions = selected_model.predict(validation_features).tolist()
    validation_predict_runtime = time.perf_counter() - predict_started
    validation_metrics = evaluate_predictions(validation_labels, validation_predictions, classes)

    train_validation_features = train_features + validation_features
    train_validation_labels = train_labels + validation_labels
    final_fit_started = time.perf_counter()
    final_model = _fit_selected(train_validation_features, train_validation_labels, best_params, seed)
    final_fit_runtime = time.perf_counter() - final_fit_started
    test_predict_started = time.perf_counter()
    test_predictions = final_model.predict(test_features).tolist()
    test_predict_runtime = time.perf_counter() - test_predict_started
    test_metrics = evaluate_predictions(test_labels, test_predictions, classes)

    output.mkdir(parents=True, exist_ok=True)
    predictions = []
    for split_name, records, values in (
        ("validation", validation_records, validation_predictions),
        ("test", test_records, test_predictions),
    ):
        predictions.extend(
            {"split": split_name, "sample_id": record["id"], "true_label": record["label"], "predicted_label": value}
            for record, value in zip(records, values)
        )
    _write_predictions(output / "predictions.csv", predictions)
    _write_json(output / "confusion_matrix.json", {"classes": list(classes), "validation": validation_metrics["confusion_matrix"], "test": test_metrics["confusion_matrix"]})
    _write_json(output / "metrics.json", {
        "run_id": RUN_ID,
        "model_family": "classical_rbf_svm",
        "split_manifest_sha256": manifest_hash,
        "classes": list(classes),
        "feature_set": "hog",
        "feature_dimension": int(len(train_features[0])),
        "train_size": len(train_records),
        "validation_size": len(validation_records),
        "test_size": len(test_records),
        "validation": validation_metrics,
        "test": test_metrics,
        "best_params": best_params,
        "cross_validation_folds": cv_folds,
        "fit_runtime_seconds": fit_runtime + final_fit_runtime,
        "predict_runtime_seconds": validation_predict_runtime + test_predict_runtime,
    })
    config_payload = {
        "run_id": RUN_ID,
        "model_family": "classical_rbf_svm",
        "dataset_root": str(dataset),
        "split_manifest_path": str(Path(manifest_path).expanduser().resolve()),
        "split_manifest_sha256": manifest_hash,
        "classes": list(classes),
        "preprocessing_mode": "yolo_crop_with_margin",
        "feature_set": "hog",
        "feature_dimension": int(len(train_features[0])),
        "qubit_count": None,
        "backend": None,
        "shots": None,
        "noise_model": None,
        "circuit_depth": None,
        "gate_count": None,
        "random_seed": seed,
        "train_size": len(train_records),
        "validation_size": len(validation_records),
        "test_size": len(test_records),
        "hyperparameter_selection": "training_only_macro_f1_grid_search",
        "final_test_fit": "train_plus_validation",
        "best_params": best_params,
    }
    _write_json(output / "config.json", config_payload)
    joblib.dump({"model": final_model, "config": config_payload}, output / "model.joblib")
    _write_json(output / "artifacts.json", {
        "run_id": RUN_ID,
        "manifest_sha256": manifest_hash,
        "files": ["config.json", "metrics.json", "predictions.csv", "confusion_matrix.json", "model.joblib", "artifacts.json"],
    })
    return {"output_root": output, "config": config_payload, "metrics": {"validation": validation_metrics, "test": test_metrics}}


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the manifest-bound classical quantum-comparison baseline")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", default="outputs/quantum_comparison/classical")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_classical_comparison(args.dataset_root, args.manifest, output_root=args.output_root)
