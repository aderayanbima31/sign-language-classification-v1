import json
from pathlib import Path


CONFIG_PATH = Path("configs/quantum_comparison.json")
PROTOCOL_PATH = Path("docs/quantum_comparison_protocol.md")


def load_protocol_config():
    return json.loads(CONFIG_PATH.read_text())


def test_machine_readable_protocol_defines_primary_benchmark_and_classes():
    config = load_protocol_config()

    assert config["dataset"]["classes"] == ["A", "B", "C", "D", "E"]
    assert config["dataset"]["primary_benchmark"] == "yolo_cropped"
    assert config["dataset"]["label_policy"] == {
        "invalid_or_missing": "fallback_to_full_image",
        "class_id": "ignored_parent_directory_is_authoritative",
        "first_non_blank_row": "authoritative",
        "multiple_rows": "ignored_after_first_non_blank_row",
        "blank_file": "invalid_then_fallback_to_full_image",
        "unreadable_label": "invalid_then_fallback_to_full_image",
        "unreadable_image": "fail_run",
        "minimum_fields": 5,
        "numeric_fields": ["center_x", "center_y", "box_width", "box_height"],
        "positive_dimensions": True,
        "empty_clipped_box": "invalid_then_fallback_to_full_image",
    }
    assert config["preprocessing"]["mode"] == "yolo_crop_with_margin"
    assert config["preprocessing"]["image_size"] == [128, 128]
    assert config["preprocessing"]["missing_or_invalid_label"] == "fallback_to_full_image"
    assert config["preprocessing"]["crop_margin_rounding"] == "integer_truncation"


def test_protocol_locks_seed_split_and_primary_metric():
    config = load_protocol_config()

    assert config["split"]["random_seed"] == 42
    assert config["split"]["train_fraction"] == 0.7
    assert config["split"]["validation_fraction"] == 0.15
    assert config["split"]["test_fraction"] == 0.15
    fractions = [
        config["split"]["train_fraction"],
        config["split"]["validation_fraction"],
        config["split"]["test_fraction"],
    ]
    assert sum(fractions) == 1.0
    assert config["split"]["strategy"] == "two_stage_stratified_split"
    assert "85% train_validation" in config["split"]["first_stage"]
    assert "0.15 / 0.85" in config["split"]["second_stage"]
    assert config["evaluation"]["primary_metric"] == "macro_f1"
    assert config["split"]["manifest_required"] is True


def test_protocol_defines_qubit_matrix_simulators_and_outputs():
    config = load_protocol_config()

    assert config["quantum_inputs"]["qubit_counts"] == [4, 6, 8]
    assert config["simulator"]["ideal"]["enabled_by_default"] is True
    assert config["simulator"]["ideal"]["shots"] is None
    assert config["simulator"]["noisy"]["enabled_by_default"] is False
    assert config["simulator"]["noisy"]["shots"] == 1024
    assert config["outputs"]["root"] == "outputs/quantum_comparison"


def test_protocol_protects_existing_phase_one_outputs():
    config = load_protocol_config()

    assert config["protected_paths"] == ["outputs/phase1", "outputs/phase1_fullimage"]
    assert all(path not in config["outputs"]["root"] for path in config["protected_paths"])
    protected = {Path(path).parts for path in config["protected_paths"]}
    output_paths = {
        Path(value).parts
        for value in config["outputs"].values()
        if isinstance(value, str)
    }
    assert protected.isdisjoint(output_paths)
    assert config["outputs"]["root"] not in config["protected_paths"]


def test_protocol_documents_required_reporting_fields_and_scope():
    config = load_protocol_config()
    protocol = PROTOCOL_PATH.read_text()

    required_fields = set(config["required_reporting_fields"])
    assert {"macro_f1", "accuracy", "qubit_count", "circuit_depth", "confusion_matrix"} <= required_fields
    assert "A`, `B`, `C`, `D`, and `E" in protocol
    assert "Phase 1 outputs are protected" in protocol
    assert "must not claim quantum advantage" in protocol


def test_protocol_and_config_agree_on_label_fallback_and_validation():
    config = load_protocol_config()
    protocol = " ".join(PROTOCOL_PATH.read_text().split())
    label_policy = config["dataset"]["label_policy"]

    assert config["preprocessing"]["missing_or_invalid_label"] == label_policy["invalid_or_missing"]
    for phrase in (
        "a missing or invalid YOLO label falls back to the full image",
        "first non-blank row",
        "Multiple rows are allowed but ignored",
        "An unreadable image is not a label fallback case",
    ):
        assert phrase in protocol
    assert "parent directory is the authoritative class label" in protocol


def test_protocol_and_config_agree_on_quantum_angle_contract():
    config = load_protocol_config()
    protocol = " ".join(PROTOCOL_PATH.read_text().split())
    angle_scaling = config["quantum_inputs"]["angle_scaling"]

    assert angle_scaling["range_radians"] == [-3.141592653589793, 3.141592653589793]
    assert angle_scaling["fit_scope"] == "training_split_only"
    assert angle_scaling["clip_transformed_values"] is True
    assert "with transformed validation/test values clipped to that range" in protocol
    assert "integer truncation exactly as Phase 1" in protocol


def test_protocol_defines_output_isolation_and_required_artifact_contract():
    config = load_protocol_config()
    protocol = " ".join(PROTOCOL_PATH.read_text().split())

    output_root = config["outputs"]["root"]
    assert output_root == "outputs/quantum_comparison"
    assert all(
        path.startswith(output_root + "/")
        for key, path in config["outputs"].items()
        if key != "root" and isinstance(path, str)
    )
    assert set(config["outputs"]["required_per_run_files"]) == {
        "config.json",
        "metrics.json",
        "predictions.csv",
        "confusion_matrix.json",
    }
    assert "must not be overwritten, deleted, or used as the destination" in protocol
