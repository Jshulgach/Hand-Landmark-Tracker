from types import SimpleNamespace

import numpy as np
import pytest

from handtrack.applications import _calibration_runtime


@pytest.mark.parametrize(
    ("backend", "manager_module"),
    [
        ("webcam", "handtrack.cameras.webcam.multi_webcam"),
        ("optitrack", "handtrack.cameras.optitrack.multi_mjpeg"),
    ],
)
def test_load_backend_routes_config_and_manager(monkeypatch, backend, manager_module):
    config_values = {
        name: index for index, name in enumerate(_calibration_runtime._CONFIG_NAMES)
    }
    config = SimpleNamespace(**config_values)
    manager = object()

    def fake_import(module_name):
        if module_name == f"handtrack.cameras.{backend}.config":
            return config
        if module_name == manager_module:
            return SimpleNamespace(CameraManager=manager)
        raise AssertionError(f"Unexpected module import: {module_name}")

    monkeypatch.setattr(_calibration_runtime.importlib, "import_module", fake_import)

    assert _calibration_runtime._load_backend(backend) is manager
    assert _calibration_runtime.ARUCO_DICT == config_values["ARUCO_DICT"]


def test_load_backend_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported calibration backend"):
        _calibration_runtime._load_backend("unknown")


def test_failed_quality_candidate_preserves_previous_calibration(
    monkeypatch, tmp_path, capsys
):
    published = tmp_path / "latest.npz"
    published.write_bytes(b"known-good-calibration")
    monkeypatch.setattr(_calibration_runtime, "CALIBRATION_DIR", str(tmp_path))
    monkeypatch.setattr(_calibration_runtime, "CALIBRATION_FILE", str(published))
    monkeypatch.setattr(_calibration_runtime, "MAX_HELDOUT_REPROJECTION_ERROR", 1.0)
    result = SimpleNamespace(
        rms=0.2,
        heldout_errors=np.array([2.0]),
        per_view_errors=np.array([0.2]),
        inlier_indices=np.array([0]),
        outlier_indices=np.array([], dtype=np.int32),
        heldout_indices=np.array([1]),
        std_intrinsics=np.zeros(18),
        model_flags=0,
        candidate_flags=np.array([0]),
        candidate_heldout_rms=np.array([2.0]),
    )

    with pytest.raises(ValueError, match="previous calibration preserved"):
        _calibration_runtime.save_calibration(
            1, (1280, 1024), 30, [np.eye(3)], [np.zeros((1, 5))],
            [np.eye(3)], [np.zeros((3, 1))], intrinsic_results=[result])
    assert published.read_bytes() == b"known-good-calibration"
    with np.load(tmp_path / "latest.npz.candidate.npz", allow_pickle=False) as data:
        assert not data["quality_passed"].item()
