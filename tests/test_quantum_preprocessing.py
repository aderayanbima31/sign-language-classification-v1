import json

import numpy as np
import pytest

from src.quantum_preprocessing import QuantumFeaturePreprocessor


def _manifest():
    return {
        "splits": {
            "train": [{"id": f"train-{index}", "label": "A"} for index in range(8)],
            "validation": [{"id": "validation-0", "label": "A"}, {"id": "validation-1", "label": "B"}],
            "test": [{"id": "test-0", "label": "A"}, {"id": "test-1", "label": "B"}],
        }
    }


def _features(manifest):
    records = {}
    for index, record in enumerate(
        record for records in manifest["splits"].values() for record in records
    ):
        records[record["id"]] = np.arange(10, dtype=float) + index * 0.25
    return records


def test_manifest_fit_is_deterministic_and_supports_configured_qubits():
    manifest = _manifest()
    features = _features(manifest)
    first = QuantumFeaturePreprocessor(4).fit_from_manifest(manifest, features)
    second = QuantumFeaturePreprocessor(4).fit_from_manifest(manifest, features)
    assert first.keys() == {"train", "validation", "test"}
    assert first["train"].shape == (8, 4)
    assert first["validation"].shape == (2, 4)
    assert np.array_equal(first["test"], second["test"])
    for qubits in (4, 6, 8):
        outputs = QuantumFeaturePreprocessor(qubits).fit_from_manifest(manifest, features)
        assert outputs["train"].shape == (8, qubits)


def test_pca_requires_at_least_one_training_sample_per_qubit():
    manifest = _manifest()
    features = _features(manifest)
    small_train = {sample_id: features[sample_id] for sample_id in list(features)[:6]}
    with pytest.raises(ValueError, match="at least 8 training samples are required"):
        QuantumFeaturePreprocessor(8).fit(np.vstack(list(small_train.values())))


def test_pca_and_scaling_fit_only_on_train_and_angles_are_clipped():
    manifest = _manifest()
    features = _features(manifest)
    processor = QuantumFeaturePreprocessor(4)
    outputs = processor.fit_from_manifest(manifest, features)
    train = outputs["train"]
    assert np.all(train >= -np.pi) and np.all(train <= np.pi)
    assert np.all(outputs["validation"] >= -np.pi) and np.all(outputs["validation"] <= np.pi)
    assert np.all(outputs["test"] >= -np.pi) and np.all(outputs["test"] <= np.pi)
    metadata = processor.metadata()
    assert metadata["fit_scope"] == "manifest.splits.train_only"
    assert metadata["train_ids"] == [record["id"] for record in manifest["splits"]["train"]]

    altered = dict(features)
    altered["validation-0"] = np.full(10, 1_000_000.0)
    altered_outputs = QuantumFeaturePreprocessor(4).fit_from_manifest(manifest, altered)
    assert np.array_equal(train, altered_outputs["train"])
    assert np.all(altered_outputs["validation"] <= np.pi)


def test_metadata_round_trip_and_validation(tmp_path):
    manifest = _manifest()
    processor = QuantumFeaturePreprocessor(6).fit_from_manifest(manifest, _features(manifest))
    path = processor.save_metadata(tmp_path / "preprocessor.json")
    restored = QuantumFeaturePreprocessor.load_metadata(path)
    matrix = np.vstack(list(_features(manifest).values()))
    assert np.array_equal(processor.transform(matrix), restored.transform(matrix))

    metadata = json.loads(path.read_text())
    metadata["pca"]["components"] = metadata["pca"]["components"][:-1]
    with pytest.raises(ValueError, match="invalid quantum preprocessing metadata"):
        QuantumFeaturePreprocessor.from_metadata(metadata)


def test_feature_records_must_match_manifest():
    manifest = _manifest()
    features = _features(manifest)
    features.pop("test-1")
    with pytest.raises(ValueError, match="do not match manifest IDs"):
        QuantumFeaturePreprocessor(4).fit_from_manifest(manifest, features)
