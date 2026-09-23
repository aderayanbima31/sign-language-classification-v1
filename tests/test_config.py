from src.config import build_config


def test_default_config_uses_five_phase_one_classes(tmp_path):
    config = build_config(dataset_root=tmp_path)
    assert config.classes == ("A", "B", "C", "D", "E")
    assert config.image_size == (128, 128)
    assert config.random_seed == 42


def test_config_paths_are_under_configured_output_root(tmp_path):
    config = build_config(dataset_root=tmp_path, output_root=tmp_path / "run")
    assert config.metrics_dir == tmp_path / "run" / "metrics"
    assert config.models_dir == tmp_path / "run" / "models"


def test_config_can_select_full_image_preprocessing(tmp_path):
    config = build_config(dataset_root=tmp_path, use_crop=False)
    assert config.use_crop is False
