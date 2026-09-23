from src.yolo_dataset import convert_to_yolo_dataset, write_dataset_yaml


def test_conversion_creates_one_class_yolo_layout(tmp_path):
    source = tmp_path / "source"
    image_dir = source / "images" / "train" / "A"
    label_dir = source / "labels" / "train" / "A"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / "one.jpg").write_bytes(b"jpeg")
    (label_dir / "one.txt").write_text("0 0.5 0.5 0.5 0.5\n")

    output = convert_to_yolo_dataset(source, tmp_path / "detector", ("A",))

    assert (output / "images" / "train" / "A_one.jpg").is_file()
    assert (output / "labels" / "train" / "A_one.txt").read_text() == "0 0.5 0.5 0.5 0.5\n"


def test_conversion_rewrites_letter_class_to_hand_class(tmp_path):
    source = tmp_path / "source"
    image_dir = source / "images" / "train" / "B"
    label_dir = source / "labels" / "train" / "B"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / "one.jpg").write_bytes(b"jpeg")
    (label_dir / "one.txt").write_text("4 0.5 0.5 0.5 0.5\n")

    output = convert_to_yolo_dataset(source, tmp_path / "detector", ("B",))

    assert (output / "labels" / "train" / "B_one.txt").read_text() == "0 0.5 0.5 0.5 0.5\n"


def test_dataset_yaml_declares_one_hand_class(tmp_path):
    path = write_dataset_yaml(tmp_path / "detector")
    assert "nc: 1" in path.read_text()
    assert "hand" in path.read_text()
