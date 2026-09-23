from pathlib import Path


def test_live_camera_documentation_contains_run_command():
    text = Path("docs/live_camera.md").read_text()
    assert "src.live_camera" in text
    assert "--detector" in text
    assert "--classifier" in text
