from src.config import build_config


def test_all_class_configuration_is_separate_from_phase_one(tmp_path):
    config = build_config(
        dataset_root=tmp_path,
        classes=tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        output_root=tmp_path / "phase2",
    )
    assert len(config.classes) == 26
    assert config.output_root.name == "phase2"
