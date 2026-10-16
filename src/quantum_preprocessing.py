"""Leakage-safe PCA and angle preprocessing for quantum model inputs.

Q4 consumes the sample records from the Q2 split manifest and a shared feature
mapping (for example, HOG vectors keyed by manifest sample ID).  PCA and the
per-component angle ranges are fitted on the manifest's training IDs only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.decomposition import PCA


ANGLE_MIN = -math.pi
ANGLE_MAX = math.pi
SUPPORTED_QUBITS = (4, 6, 8)
SCHEMA_VERSION = "q4.1"


def _records_by_id(feature_records: Mapping[str, object] | Sequence[Mapping[str, object]]) -> dict[str, np.ndarray]:
    """Normalize either an ID-to-vector mapping or shared feature records."""

    if isinstance(feature_records, Mapping):
        records = feature_records.items()
        normalized = {str(sample_id): np.asarray(vector, dtype=np.float64) for sample_id, vector in records}
    else:
        normalized = {}
        for record in feature_records:
            if not isinstance(record, Mapping) or not isinstance(record.get("id"), str):
                raise ValueError("feature records must contain string id fields")
            if "features" not in record:
                raise ValueError("feature records must contain features fields")
            sample_id = record["id"]
            if sample_id in normalized:
                raise ValueError(f"duplicate feature record: {sample_id}")
            normalized[sample_id] = np.asarray(record["features"], dtype=np.float64)
    if not normalized:
        raise ValueError("feature records must not be empty")
    dimensions = {vector.ndim for vector in normalized.values()}
    if dimensions != {1}:
        raise ValueError("feature vectors must be one-dimensional")
    widths = {vector.shape[0] for vector in normalized.values()}
    if len(widths) != 1 or not widths or next(iter(widths)) < 1:
        raise ValueError("feature vectors must have one consistent non-zero dimension")
    if not all(np.all(np.isfinite(vector)) for vector in normalized.values()):
        raise ValueError("feature vectors must contain only finite values")
    return normalized


def _manifest_ids(manifest: Mapping[str, object]) -> dict[str, list[str]]:
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {"train", "validation", "test"}:
        raise ValueError("manifest splits must contain train, validation, and test")
    result: dict[str, list[str]] = {}
    all_ids: list[str] = []
    for split_name in ("train", "validation", "test"):
        records = splits[split_name]
        if not isinstance(records, list) or not records:
            raise ValueError(f"manifest {split_name} must contain records")
        ids: list[str] = []
        for record in records:
            if not isinstance(record, Mapping) or not isinstance(record.get("id"), str):
                raise ValueError(f"manifest {split_name} contains an invalid record")
            ids.append(record["id"])
        if len(set(ids)) != len(ids):
            raise ValueError(f"manifest {split_name} contains duplicate IDs")
        result[split_name] = ids
        all_ids.extend(ids)
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("manifest splits must be disjoint")
    return result


class QuantumFeaturePreprocessor:
    """Fit train-only PCA and deterministic per-qubit angle scaling."""

    def __init__(self, qubit_count: int, *, angle_range: tuple[float, float] = (ANGLE_MIN, ANGLE_MAX)) -> None:
        if qubit_count not in SUPPORTED_QUBITS:
            raise ValueError(f"qubit_count must be one of {SUPPORTED_QUBITS}")
        if len(angle_range) != 2 or not np.all(np.isfinite(angle_range)) or angle_range[0] >= angle_range[1]:
            raise ValueError("angle_range must contain two finite, increasing values")
        self.qubit_count = qubit_count
        self.angle_range = (float(angle_range[0]), float(angle_range[1]))
        self._pca: PCA | None = None
        self._train_min: np.ndarray | None = None
        self._train_max: np.ndarray | None = None
        self._input_dimension: int | None = None
        self._train_ids: tuple[str, ...] = ()

    @property
    def fitted(self) -> bool:
        return self._pca is not None

    def fit(self, train_features: np.ndarray, *, train_ids: Sequence[str] = ()) -> "QuantumFeaturePreprocessor":
        features = self._validate_matrix(train_features)
        if features.shape[1] < self.qubit_count:
            raise ValueError("input feature dimension must be at least the qubit count")
        if features.shape[0] < self.qubit_count:
            raise ValueError(
                f"at least {self.qubit_count} training samples are required "
                f"for {self.qubit_count}-qubit PCA"
            )
        self._pca = PCA(n_components=self.qubit_count, whiten=False, svd_solver="full")
        reduced = self._pca.fit_transform(features)
        self._train_min = reduced.min(axis=0)
        self._train_max = reduced.max(axis=0)
        self._input_dimension = int(features.shape[1])
        self._train_ids = tuple(train_ids)
        if len(set(self._train_ids)) != len(self._train_ids):
            raise ValueError("train_ids must be unique")
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        self._require_fitted()
        matrix = self._validate_matrix(features)
        if matrix.shape[1] != self._input_dimension:
            raise ValueError(f"expected feature dimension {self._input_dimension}, got {matrix.shape[1]}")
        assert self._pca is not None and self._train_min is not None and self._train_max is not None
        reduced = self._pca.transform(matrix)
        low, high = self.angle_range
        clipped = np.clip(reduced, self._train_min, self._train_max)
        span = self._train_max - self._train_min
        normalized = np.divide(clipped - self._train_min, span, out=np.full_like(clipped, 0.5), where=span > 0)
        return (low + normalized * (high - low)).astype(np.float64)

    def fit_from_manifest(
        self,
        manifest: Mapping[str, object],
        feature_records: Mapping[str, object] | Sequence[Mapping[str, object]],
    ) -> dict[str, np.ndarray]:
        """Fit on manifest train IDs and return angle matrices for all splits."""

        split_ids = _manifest_ids(manifest)
        features_by_id = _records_by_id(feature_records)
        expected_ids = {sample_id for ids in split_ids.values() for sample_id in ids}
        if set(features_by_id) != expected_ids:
            missing = sorted(expected_ids - set(features_by_id))
            extra = sorted(set(features_by_id) - expected_ids)
            raise ValueError(f"feature records do not match manifest IDs; missing={missing}, extra={extra}")
        train_matrix = np.vstack([features_by_id[sample_id] for sample_id in split_ids["train"]])
        self.fit(train_matrix, train_ids=split_ids["train"])
        return {
            split_name: self.transform(np.vstack([features_by_id[sample_id] for sample_id in ids]))
            for split_name, ids in split_ids.items()
        }

    def metadata(self) -> dict[str, object]:
        self._require_fitted()
        assert self._pca is not None and self._train_min is not None and self._train_max is not None
        return {
            "schema_version": SCHEMA_VERSION,
            "qubit_count": self.qubit_count,
            "input_dimension": self._input_dimension,
            "pca": {
                "n_components": self.qubit_count,
                "whiten": False,
                "svd_solver": "full",
                "components": self._pca.components_.tolist(),
                "mean": self._pca.mean_.tolist(),
                "explained_variance": self._pca.explained_variance_.tolist(),
            },
            "angle_scaling": {
                "method": "training_fitted_minmax",
                "range_radians": list(self.angle_range),
                "clip_transformed_values": True,
                "train_min": self._train_min.tolist(),
                "train_max": self._train_max.tolist(),
            },
            "fit_scope": "manifest.splits.train_only",
            "train_ids": list(self._train_ids),
        }

    def save_metadata(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.metadata(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> "QuantumFeaturePreprocessor":
        _validate_metadata(metadata)
        scaling = metadata["angle_scaling"]
        assert isinstance(scaling, Mapping)
        instance = cls(int(metadata["qubit_count"]), angle_range=tuple(scaling["range_radians"]))  # type: ignore[arg-type]
        pca_data = metadata["pca"]
        assert isinstance(pca_data, Mapping)
        pca = PCA(n_components=instance.qubit_count, whiten=False, svd_solver="full")
        pca.components_ = np.asarray(pca_data["components"], dtype=np.float64)
        pca.mean_ = np.asarray(pca_data["mean"], dtype=np.float64)
        pca.explained_variance_ = np.asarray(pca_data["explained_variance"], dtype=np.float64)
        pca.n_features_in_ = int(metadata["input_dimension"])
        pca.n_samples_ = len(metadata["train_ids"])
        instance._pca = pca
        instance._input_dimension = pca.n_features_in_
        instance._train_min = np.asarray(scaling["train_min"], dtype=np.float64)
        instance._train_max = np.asarray(scaling["train_max"], dtype=np.float64)
        instance._train_ids = tuple(metadata["train_ids"])
        return instance

    @classmethod
    def load_metadata(cls, input_path: str | Path) -> "QuantumFeaturePreprocessor":
        return cls.from_metadata(json.loads(Path(input_path).read_text(encoding="utf-8")))

    @staticmethod
    def _validate_matrix(features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("features must be a non-empty two-dimensional matrix")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("features must contain only finite values")
        return matrix

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("QuantumFeaturePreprocessor must be fitted before use")


def _validate_metadata(metadata: Mapping[str, object]) -> None:
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported quantum preprocessing metadata schema")
    qubit_count = metadata.get("qubit_count")
    input_dimension = metadata.get("input_dimension")
    train_ids = metadata.get("train_ids")
    pca = metadata.get("pca")
    scaling = metadata.get("angle_scaling")
    if qubit_count not in SUPPORTED_QUBITS or not isinstance(input_dimension, int) or input_dimension < qubit_count:
        raise ValueError("invalid quantum preprocessing dimensions")
    if not isinstance(train_ids, list) or not train_ids or len(set(train_ids)) != len(train_ids):
        raise ValueError("metadata train_ids must be a non-empty unique list")
    if not isinstance(pca, Mapping) or not isinstance(scaling, Mapping):
        raise ValueError("metadata must contain pca and angle_scaling objects")
    low_high = scaling.get("range_radians")
    components = np.asarray(pca.get("components"), dtype=np.float64)
    mean = np.asarray(pca.get("mean"), dtype=np.float64)
    variance = np.asarray(pca.get("explained_variance"), dtype=np.float64)
    train_min = np.asarray(scaling.get("train_min"), dtype=np.float64)
    train_max = np.asarray(scaling.get("train_max"), dtype=np.float64)
    if (
        not isinstance(low_high, list)
        or len(low_high) != 2
        or low_high[0] >= low_high[1]
        or components.shape != (qubit_count, input_dimension)
        or mean.shape != (input_dimension,)
        or variance.shape != (qubit_count,)
        or train_min.shape != (qubit_count,)
        or train_max.shape != (qubit_count,)
        or np.any(train_min > train_max)
        or not all(np.all(np.isfinite(array)) for array in (components, mean, variance, train_min, train_max))
    ):
        raise ValueError("invalid quantum preprocessing metadata arrays")
