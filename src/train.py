"""Train and compare classical models for BISINDO gesture images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import DEFAULT_CLASSES, ProjectConfig, build_config
from .dataset import discover_samples, make_splits
from .evaluate import evaluate_predictions
from .features import extract_features
from .preprocessing import load_and_preprocess


def validate_training_labels(labels: Sequence[str]) -> None:
    if len(set(labels)) < 2:
        raise ValueError("training requires at least two classes")


def build_model_pipeline(model_name: str, feature_set: str, config: ProjectConfig | None = None) -> Pipeline:
    del feature_set
    seed = config.random_seed if config else 42
    if model_name == "knn":
        model = KNeighborsClassifier(n_neighbors=5)
        return Pipeline([("scaler", StandardScaler()), ("model", model)])
    if model_name == "svm":
        model = SVC(kernel="rbf", probability=False, random_state=seed)
        return Pipeline([("scaler", StandardScaler()), ("model", model)])
    if model_name == "random_forest":
        model = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        return Pipeline([("model", model)])
    raise ValueError(f"unknown model: {model_name}")


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value)!r}")


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def _build_feature_matrix(samples, feature_set: str, config: ProjectConfig):
    rows = []
    labels = []
    for sample in samples:
        image = load_and_preprocess(sample, config, use_crop=config.use_crop)
        rows.append(extract_features(image, feature_set))
        labels.append(sample.class_name)
    return rows, labels


def _tune_if_needed(pipeline, model_name: str, x_train, y_train, config: ProjectConfig):
    if model_name != "svm":
        pipeline.fit(x_train, y_train)
        return pipeline, {}
    counts = pd.Series(y_train).value_counts()
    cv_folds = min(5, int(counts.min()))
    if cv_folds < 2:
        raise ValueError("training requires at least two samples per class for cross-validation")
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
    return search.best_estimator_, search.best_params_


def run_experiments(config: ProjectConfig) -> pd.DataFrame:
    samples = discover_samples(config)
    labels = [sample.class_name for sample in samples]
    validate_training_labels(labels)
    train_idx, val_idx, test_idx = make_splits(
        samples,
        labels,
        random_seed=config.random_seed,
        test_size=config.test_size,
        validation_size=config.validation_size,
    )
    config.metrics_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    config.models_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        ("E0", "pixels", "knn"),
        ("E1", "hog", "svm"),
        ("E2", "hog", "random_forest"),
        ("E3", "hog+shape", "svm"),
    ]
    feature_cache = {}
    for _, feature_set, _ in experiments:
        if feature_set not in feature_cache:
            feature_cache[feature_set] = _build_feature_matrix(samples, feature_set, config)

    records = []
    fitted = {}
    for experiment_id, feature_set, model_name in experiments:
        matrix, matrix_labels = feature_cache[feature_set]
        x_train = [matrix[index] for index in train_idx]
        y_train = [matrix_labels[index] for index in train_idx]
        x_val = [matrix[index] for index in val_idx]
        y_val = [matrix_labels[index] for index in val_idx]
        pipeline, best_params = _tune_if_needed(
            build_model_pipeline(model_name, feature_set, config), model_name, x_train, y_train, config
        )
        y_pred = pipeline.predict(x_val)
        metrics = evaluate_predictions(y_val, y_pred, config.classes)
        payload = {
            "experiment_id": experiment_id,
            "feature_set": feature_set,
            "model": model_name,
            "classes": config.classes,
            "random_seed": config.random_seed,
            "image_size": config.image_size,
            "train_size": len(train_idx),
            "validation_size": len(val_idx),
            "test_size": len(test_idx),
            "best_params": best_params,
            "metrics": metrics,
        }
        _save_json(config.metrics_dir / f"{experiment_id}.json", payload)
        records.append({"experiment_id": experiment_id, "feature_set": feature_set, "model": model_name, **metrics})
        fitted[experiment_id] = (pipeline, feature_set)

    results = pd.DataFrame(
        [
            {
                "experiment_id": record["experiment_id"],
                "feature_set": record["feature_set"],
                "model": record["model"],
                "accuracy": record["accuracy"],
                "macro_f1": record["macro_f1"],
            }
            for record in records
        ]
    ).sort_values("macro_f1", ascending=False)
    results.to_csv(config.metrics_dir / "experiment_results.csv", index=False)

    best_id = str(results.iloc[0]["experiment_id"])
    best_pipeline, best_feature_set = fitted[best_id]
    matrix, matrix_labels = feature_cache[best_feature_set]
    train_val_idx = train_idx + val_idx
    best_pipeline.fit([matrix[index] for index in train_val_idx], [matrix_labels[index] for index in train_val_idx])
    y_test = [matrix_labels[index] for index in test_idx]
    test_predictions = best_pipeline.predict([matrix[index] for index in test_idx])
    test_metrics = evaluate_predictions(y_test, test_predictions, config.classes)
    _save_json(
        config.metrics_dir / "final_test.json",
        {"best_experiment": best_id, "feature_set": best_feature_set, "metrics": test_metrics},
    )
    sns.heatmap(
        test_metrics["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config.classes,
        yticklabels=config.classes,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"BISINDO final test confusion matrix ({best_id})")
    plt.tight_layout()
    plt.savefig(config.figures_dir / "final_confusion_matrix.png", dpi=150)
    plt.close()
    joblib.dump(
        {"model": best_pipeline, "feature_set": best_feature_set, "config": config},
        config.models_dir / "best_model.joblib",
    )
    return results


def _parse_args():
    parser = argparse.ArgumentParser(description="Train BISINDO classical-ML experiments")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    parser.add_argument("--output-root", default="outputs/phase1")
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="train on full images so prediction can use images without label files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    frame = run_experiments(
        build_config(
            dataset_root=args.dataset_root,
            classes=args.classes,
            output_root=args.output_root,
            use_crop=not args.no_crop,
        )
    )
    print(frame.to_string(index=False))
