from src.train_detector import detector_output_path


def test_detector_output_path_is_branch_specific(tmp_path):
    path = detector_output_path(tmp_path)
    assert path == tmp_path / "weights" / "best.pt"
