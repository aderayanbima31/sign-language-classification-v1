"""Metrics and evaluation artifact helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def evaluate_predictions(
    y_true: Sequence[str], y_pred: Sequence[str], class_names: Sequence[str]
) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(class_names), average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, labels=list(class_names), target_names=list(class_names), output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(class_names)).tolist(),
    }
