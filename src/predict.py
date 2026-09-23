"""Predict one BISINDO gesture image with a saved model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from .dataset import Sample
from .features import extract_features
from .preprocessing import load_and_preprocess


def predict_image(model_path: str | Path, image_path: str | Path) -> dict:
    model_path = Path(model_path)
    image_path = Path(image_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"model file not found: {model_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"image file not found: {image_path}")
    artifact = joblib.load(model_path)
    config = artifact["config"]
    sample = Sample(image_path=image_path, label_path=None, class_name="unknown", split="predict")
    image = load_and_preprocess(sample, config, use_crop=config.use_crop)
    vector = extract_features(image, artifact["feature_set"])
    model = artifact["model"]
    predicted = str(model.predict([vector])[0])
    result = {"image": str(image_path), "prediction": predicted}
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([vector])[0]
        result["probabilities"] = {str(label): float(score) for label, score in zip(model.classes_, probabilities)}
    return result


def _parse_args():
    parser = argparse.ArgumentParser(description="Predict a BISINDO gesture image")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(json.dumps(predict_image(args.model, args.image), indent=2))
