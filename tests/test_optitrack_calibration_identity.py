from types import SimpleNamespace

import numpy as np
import pytest

from handtrack.cameras.optitrack import mocap_tracker


def _write_calibration(path, uid_low=(11, 22)):
    data = {
        "schema_version": 2,
        "num_cameras": 2,
        "img_size": np.array([1280, 1024]),
        "camera_uid_high": np.array([1, 1]),
        "camera_uid_low": np.array(uid_low),
        "camera_serials": np.array(["F13-A", "F13-B"]),
    }
    for camera in range(2):
        data[f"camera_matrix_{camera}"] = np.eye(3)
        data[f"dist_coeffs_{camera}"] = np.zeros((1, 5))
        data[f"R_{camera}"] = np.eye(3)
        data[f"T_{camera}"] = np.zeros((3, 1))
    np.savez(path, **data)


def _tracker():
    tracker = mocap_tracker.MultiCameraTracker()
    tracker.num_cameras = 2
    tracker.img_width = 1280
    tracker.img_height = 1024
    tracker.cam_mgr = SimpleNamespace(
        get_camera_diagnostics=lambda: [
            {"uid": (1, 11), "serial": "F13-A"},
            {"uid": (1, 22), "serial": "F13-B"},
        ]
    )
    return tracker


def test_schema_v2_calibration_accepts_exact_camera_identity(monkeypatch, tmp_path):
    calibration = tmp_path / "calibration.npz"
    _write_calibration(calibration)
    monkeypatch.setattr(mocap_tracker, "CALIBRATION_FILE", str(calibration))
    tracker = _tracker()

    tracker.load_calibration()

    assert len(tracker.projection_matrices) == 2


def test_schema_v2_calibration_rejects_reordered_camera_uid(monkeypatch, tmp_path):
    calibration = tmp_path / "calibration.npz"
    _write_calibration(calibration, uid_low=(22, 11))
    monkeypatch.setattr(mocap_tracker, "CALIBRATION_FILE", str(calibration))
    tracker = _tracker()

    tracker.load_calibration()
    assert len(tracker.projection_matrices) == 2


def test_schema_v2_calibration_no_longer_requires_quality_flag(monkeypatch, tmp_path):
    calibration = tmp_path / "calibration.npz"
    _write_calibration(calibration)
    monkeypatch.setattr(mocap_tracker, "CALIBRATION_FILE", str(calibration))
    tracker = _tracker()

    tracker.load_calibration()
