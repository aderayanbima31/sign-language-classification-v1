import cv2
import numpy as np

from src.config import build_config
from src.dataset import Sample
from src.features import extract_features, extract_hog
from src.preprocessing import load_and_preprocess, preprocess_image


def test_preprocess_returns_fixed_grayscale_shape():
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    result = preprocess_image(image, image_size=(128, 128))
    assert result.shape == (128, 128)
    assert result.dtype == np.uint8


def test_hog_vector_length_is_stable():
    image = np.zeros((128, 128), dtype=np.uint8)
    assert extract_hog(image).shape == extract_hog(image).shape


def test_combined_features_are_longer_than_hog():
    image = np.zeros((128, 128), dtype=np.uint8)
    assert extract_features(image, "hog+shape").size > extract_features(image, "hog").size


def test_missing_label_falls_back_to_full_image(tmp_path):
    image_path = tmp_path / "sample.jpg"
    cv2.imwrite(str(image_path), np.zeros((40, 60, 3), dtype=np.uint8))
    sample = Sample(image_path=image_path, label_path=None, class_name="A", split="train")
    result = load_and_preprocess(sample, build_config(dataset_root=tmp_path), use_crop=True)
    assert result.shape == (128, 128)
