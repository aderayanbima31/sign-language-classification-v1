import csv
import json
from pathlib import Path

import pytest

import src.classical_comparison as classical_comparison
from src.classical_comparison import _safe_output_dir, load_manifest
from src.benchmark_split import create_manifest, manifest_sha256


def _manifest(tmp_path: Path):
    root = tmp_path / "dataset"
    for partition in ("train", "val"):
        for label in "AB":
            folder = root / "images" / partition / label
            folder.mkdir(parents=True)
            for index in range(12):
                (folder / f"{index}.jpg").write_bytes(b"fixture")
    manifest = create_manifest(root, classes=("A", "B"))
    path = tmp_path / "split_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_load_manifest_requires_locked_hash_and_valid_split(tmp_path):
    path = _manifest(tmp_path)
    loaded, digest = load_manifest(path)
    assert digest == manifest_sha256(loaded)
    assert loaded["counts"] == {"train": 17, "validation": 4, "test": 3}


def test_load_manifest_rejects_tampering(tmp_path):
    path = _manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["classes"] = ["A", "C"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_sha256"):
        load_manifest(path)


def test_output_is_restricted_to_quantum_comparison_and_protects_phase1(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    allowed = repository_root / "outputs" / "quantum_comparison" / "classical"
    assert _safe_output_dir(allowed) == allowed.resolve()
    with pytest.raises(ValueError, match="quantum_comparison"):
        _safe_output_dir(tmp_path / "outputs" / "other")
    with pytest.raises(ValueError, match="protected"):
        _safe_output_dir(repository_root / "outputs" / "phase1")


def test_output_boundary_rejects_sibling_and_prefix_paths(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    quantum_root = repository_root / "outputs" / "quantum_comparison"
    with pytest.raises(ValueError, match="quantum_comparison"):
        _safe_output_dir(repository_root / "outputs" / "quantum_comparison-sibling")
    with pytest.raises(ValueError, match="quantum_comparison"):
        _safe_output_dir(tmp_path / "quantum_comparison" / "nested")
    assert _safe_output_dir(quantum_root / "nested" / "run") == (quantum_root / "nested" / "run").resolve()


def test_output_symlink_cannot_escape_or_enter_phase1(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    quantum_root = repository_root / "outputs" / "quantum_comparison"
    escape_link = quantum_root / "pytest-escape-link"
    phase1_link = quantum_root / "pytest-phase1-link"
    escape_link.unlink(missing_ok=True)
    phase1_link.unlink(missing_ok=True)
    escape_link.symlink_to(tmp_path, target_is_directory=True)
    phase1_link.symlink_to(repository_root / "outputs" / "phase1", target_is_directory=True)
    try:
        with pytest.raises(ValueError, match="quantum_comparison"):
            _safe_output_dir(escape_link / "run")
        with pytest.raises(ValueError, match="protected"):
            _safe_output_dir(phase1_link / "run")
    finally:
        escape_link.unlink(missing_ok=True)
        phase1_link.unlink(missing_ok=True)


def test_runner_uses_manifest_records_and_writes_complete_artifacts(tmp_path, monkeypatch):
    manifest_path = _manifest(tmp_path)
    repository_root = Path(__file__).resolve().parents[1]
    output_root = repository_root / "outputs" / "quantum_comparison" / f"pytest-{tmp_path.name}"
    feature_calls = []
    fit_calls = []
    selected_calls = []

    def fake_feature_matrix(dataset_root, records, config):
        records = list(records)
        feature_calls.append([record["id"] for record in records])
        return [[float(index), 1.0] for index, _ in enumerate(records)]

    class FakeModel:
        def predict(self, features):
            return ["A" for _ in features]

    def fake_fit_baseline(features, labels, seed):
        fit_calls.append((len(features), list(labels), seed))
        return FakeModel(), {"model__C": 1, "model__gamma": "scale"}, 2

    def fake_fit_selected(features, labels, best_params, seed):
        selected_calls.append((len(features), list(labels), dict(best_params), seed))
        return FakeModel()

    def fake_evaluate(true, predicted, classes):
        return {"accuracy": 1.0, "macro_f1": 1.0, "confusion_matrix": [[len(true)]]}

    def fake_dump(payload, path):
        Path(path).write_bytes(b"fake model")

    monkeypatch.setattr(classical_comparison, "_feature_matrix", fake_feature_matrix)
    monkeypatch.setattr(classical_comparison, "_fit_baseline", fake_fit_baseline)
    monkeypatch.setattr(classical_comparison, "_fit_selected", fake_fit_selected)
    monkeypatch.setattr(classical_comparison, "evaluate_predictions", fake_evaluate)
    monkeypatch.setattr(classical_comparison.joblib, "dump", fake_dump)
    try:
        result = classical_comparison.run_classical_comparison(tmp_path / "dataset", manifest_path, output_root=output_root)
        manifest, _ = load_manifest(manifest_path)
        assert feature_calls == [[record["id"] for record in manifest["splits"][name]] for name in ("train", "validation", "test")]
        assert fit_calls == [(17, [record["label"] for record in manifest["splits"]["train"]], manifest["seed"])]
        assert selected_calls == [(21, [record["label"] for record in manifest["splits"]["train"] + manifest["splits"]["validation"]], {"model__C": 1, "model__gamma": "scale"}, manifest["seed"])]
        expected_files = {"config.json", "metrics.json", "predictions.csv", "confusion_matrix.json", "model.joblib", "artifacts.json"}
        assert {path.name for path in output_root.iterdir()} == expected_files
        assert result["output_root"] == output_root.resolve()
        with (output_root / "predictions.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert [row["sample_id"] for row in rows] == [
            record["id"] for name in ("validation", "test") for record in manifest["splits"][name]
        ]
        assert all(set(row) == {"split", "sample_id", "true_label", "predicted_label"} for row in rows)
    finally:
        for child in output_root.iterdir() if output_root.exists() else ():
            child.unlink()
        output_root.rmdir() if output_root.exists() else None
