import json
from pathlib import Path

import pytest

from src.benchmark_split import create_manifest, manifest_sha256, write_manifest


def _make_fixture(root: Path, samples_per_class: int = 10) -> Path:
    for partition in ("train", "val"):
        for label in "ABCDE":
            image_dir = root / "images" / partition / label
            image_dir.mkdir(parents=True, exist_ok=True)
            for index in range(samples_per_class // 2):
                (image_dir / f"{partition}_{label}_{index}.jpg").write_bytes(b"fixture")
    return root


def test_manifest_uses_two_stage_stratified_split_and_records_contract(tmp_path):
    manifest = create_manifest(_make_fixture(tmp_path))

    assert manifest["classes"] == list("ABCDE")
    assert manifest["seed"] == 42
    assert manifest["counts"] == {"train": 34, "validation": 8, "test": 8}
    assert manifest["split_policy"]["strategy"] == "two_stage_stratified_split"
    assert sum(manifest["counts"].values()) == 50
    assert manifest_sha256(manifest) == manifest["manifest_sha256"]

    all_records = [record for records in manifest["splits"].values() for record in records]
    assert len({record["id"] for record in all_records}) == 50
    assert {record["label"] for record in manifest["splits"]["test"]} == set("ABCDE")
    assert all(record["id"].endswith(".jpg") for record in all_records)


def test_manifest_splits_are_class_complete_disjoint_and_cover_inventory(tmp_path):
    manifest = create_manifest(_make_fixture(tmp_path))
    split_records = manifest["splits"]
    split_ids = [
        {record["id"] for record in records}
        for records in split_records.values()
    ]

    assert all(
        {record["label"] for record in records} == set("ABCDE")
        for records in split_records.values()
    )
    assert split_ids[0].isdisjoint(split_ids[1])
    assert split_ids[0].isdisjoint(split_ids[2])
    assert split_ids[1].isdisjoint(split_ids[2])
    expected_ids = {
        f"images/{partition}/{label}/{partition}_{label}_{index}.jpg"
        for partition in ("train", "val")
        for label in "ABCDE"
        for index in range(5)
    }
    assert set.union(*split_ids) == expected_ids


def test_manifest_is_repeatable_and_independent_of_discovery_order(tmp_path):
    root = _make_fixture(tmp_path)
    first = create_manifest(root)
    second = create_manifest(root)

    assert first == second
    for records in first["splits"].values():
        assert records == sorted(records, key=lambda record: record["id"])


def test_manifest_rejects_insufficient_class_samples(tmp_path):
    root = _make_fixture(tmp_path)
    for image_path in (root / "images").glob("*/E/*.jpg"):
        if image_path.name != "train_E_0.jpg":
            image_path.unlink()
    with pytest.raises(ValueError, match="at least three|too small"):
        create_manifest(root)


def test_manifest_rejects_dataset_that_cannot_allocate_each_split(tmp_path):
    with pytest.raises(ValueError, match="two-stage 70/15/15 split"):
        create_manifest(_make_fixture(tmp_path, samples_per_class=2))


def test_manifest_write_round_trip_and_protected_output(tmp_path):
    manifest = create_manifest(_make_fixture(tmp_path / "dataset"))
    destination = tmp_path / "outputs" / "split_manifest.json"
    assert write_manifest(manifest, destination) == destination
    assert json.loads(destination.read_text(encoding="utf-8")) == manifest

    with pytest.raises(ValueError, match="protected"):
        write_manifest(manifest, tmp_path / "outputs" / "phase1" / "split_manifest.json")


def test_manifest_write_rejects_semantically_invalid_manifest_with_recomputed_hash(tmp_path):
    manifest = create_manifest(_make_fixture(tmp_path / "dataset"))
    manifest["splits"]["test"] = manifest["splits"]["test"][:-1]
    manifest["counts"]["test"] -= 1
    manifest["manifest_sha256"] = manifest_sha256(manifest)

    with pytest.raises(ValueError, match="every configured class|disjoint|cover"):
        write_manifest(manifest, tmp_path / "outputs" / "invalid.json")


def test_manifest_write_rejects_same_size_id_replacement_with_recomputed_hash(tmp_path):
    manifest = create_manifest(_make_fixture(tmp_path / "dataset"))
    replacement = dict(manifest["splits"]["test"][0])
    replacement["id"] = "images/test/A/replaced.jpg"
    manifest["splits"]["test"][0] = replacement
    manifest["manifest_sha256"] = manifest_sha256(manifest)

    with pytest.raises(ValueError, match="exactly the inventory|match the inventory"):
        write_manifest(manifest, tmp_path / "outputs" / "invalid-id.json")


def test_manifest_write_rejects_label_tampering_with_recomputed_hash(tmp_path):
    manifest = create_manifest(_make_fixture(tmp_path / "dataset"))
    tampered = dict(manifest["splits"]["test"][0])
    tampered["label"] = next(label for label in "ABCDE" if label != tampered["label"])
    manifest["splits"]["test"][0] = tampered
    manifest["manifest_sha256"] = manifest_sha256(manifest)

    with pytest.raises(ValueError, match="match the inventory"):
        write_manifest(manifest, tmp_path / "outputs" / "invalid-label.json")


def test_manifest_write_rejects_symlinked_protected_output(tmp_path):
    manifest = create_manifest(_make_fixture(tmp_path / "dataset"))
    protected = tmp_path / "outputs" / "phase1"
    protected.mkdir(parents=True)
    link = tmp_path / "outputs" / "quantum_comparison"
    link.symlink_to(protected, target_is_directory=True)

    with pytest.raises(ValueError, match="protected"):
        write_manifest(manifest, link / "split_manifest.json")


def test_manifest_does_not_require_image_decoding(tmp_path):
    root = _make_fixture(tmp_path)
    (root / "images" / "train" / "A" / "train_A_0.jpg").write_bytes(b"not-an-image")

    manifest = create_manifest(root)
    assert manifest["dataset_identity"]["sample_count"] == 50
