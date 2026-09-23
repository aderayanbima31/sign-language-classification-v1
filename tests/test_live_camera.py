import pytest

from src.live_camera import validate_runtime_paths


def test_missing_runtime_model_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="detector"):
        validate_runtime_paths(tmp_path / "missing-detector.pt", tmp_path / "classifier.joblib")
