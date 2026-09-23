import numpy as np
import pytest

from src.evaluate import evaluate_predictions
from src.train import build_model_pipeline, validate_training_labels


def test_evaluation_contains_required_metrics():
    result = evaluate_predictions(["A", "B", "A"], ["A", "B", "B"], ["A", "B"])
    assert {"accuracy", "macro_f1", "classification_report", "confusion_matrix"} <= result.keys()
    assert np.asarray(result["confusion_matrix"]).shape == (2, 2)


def test_svm_pipeline_contains_scaler_and_classifier():
    pipeline = build_model_pipeline("svm", "hog")
    assert "scaler" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_training_rejects_fewer_than_two_classes():
    with pytest.raises(ValueError, match="two classes"):
        validate_training_labels(["A", "A"])
