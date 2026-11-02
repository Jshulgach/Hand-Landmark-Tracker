"""External example paths and CLI availability without ML dependencies."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from mavis_track.research import research_data_dir, research_example_args

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = (
    "Joint_Kinematics_from_EMG/create_joint_angle_dataset.py",
    "Joint_Kinematics_from_EMG/visualize_predictions.py",
    "Joint_Kinematics_from_EMG/predict_joint_angles_full_dataset.py",
    "Smoothing_Motion_with_RNN/1_feature_extraction.py",
    "Smoothing_Motion_with_RNN/2_train_gru.py",
    "Smoothing_Motion_with_RNN/3_predict.py",
    "Smoothing_Motion_with_RNN/4_evaluate.py",
)


def test_default_external_path_is_independent_of_working_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("MAVIS_DATA_DIR", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    expected = tmp_path / "home/Documents/Hand-Landmark-Tracker-Research/data"
    monkeypatch.chdir(tmp_path)
    assert research_data_dir() == expected.resolve()
    assert research_data_dir(collection="smoothing") == (expected / "smoothing").resolve()
    assert not expected.exists()  # Resolving a path does not create user data.


def test_explicit_directory_overrides_environment_and_collection(monkeypatch, tmp_path):
    monkeypatch.setenv("MAVIS_DATA_DIR", str(tmp_path / "shared"))
    assert research_data_dir(collection="smoothing") == (tmp_path / "shared/smoothing").resolve()
    explicit = tmp_path / "specific"
    assert research_data_dir(explicit, collection="smoothing") == explicit.resolve()


def test_video_argument_and_legacy_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("MAVIS_DATA_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("HANDTRACK_VIDEO_PATH", str(tmp_path / "source.mp4"))
    args = research_example_args("test", [], video=True)
    assert args.video == (tmp_path / "source.mp4").resolve()
    args = research_example_args("test", ["--data-dir", str(tmp_path), "--video", str(tmp_path / "other.mp4")], video=True)
    assert args.data_dir == tmp_path and args.video == tmp_path / "other.mp4"


@pytest.mark.parametrize("relative", EXAMPLES)
def test_research_help_works_from_another_directory_without_loading_data(relative, tmp_path):
    environment = dict(os.environ, MAVIS_DATA_DIR=str(tmp_path / "missing-data"))
    result = subprocess.run([sys.executable, "-B", str(ROOT / "examples" / relative), "--help"],
                            cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "--data-dir" in result.stdout
    assert not (tmp_path / "missing-data").exists()
