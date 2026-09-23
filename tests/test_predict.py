import pytest

from src.predict import predict_image


def test_missing_model_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="model"):
        predict_image(tmp_path / "missing.joblib", tmp_path / "image.jpg")
