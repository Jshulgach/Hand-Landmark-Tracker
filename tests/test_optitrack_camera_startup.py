from pathlib import Path

import pytest

from unity_hand_tracking.optitrack_cam_py import config, multi_mjpeg


class _FakeCamera:
    def __init__(self, index):
        self.index = index
        self.released = False

    def release(self):
        self.released = True


class _FakeWorker:
    def __init__(self, camera, index, exposure, mjpeg_mode):
        self.camera = camera
        self.index = index
        self.started = False

    def start(self):
        self.started = True


class _DelayedCameraSdk:
    def __init__(self, camera_count=3, never_ready=()):
        self._camera_count = camera_count
        self.never_ready = set(never_ready)
        self.calls = {index: 0 for index in range(camera_count)}
        self.cameras = {
            index: _FakeCamera(index) for index in range(camera_count)
        }

    def initialize_sdk(self):
        return None

    def camera_count(self):
        return self._camera_count

    def get_camera_by_index(self, index):
        self.calls[index] += 1
        if index in self.never_ready or self.calls[index] < 2:
            return None
        return self.cameras[index]


def test_optitrack_uses_core_backend_calibration_file():
    calibration = Path(config.CALIBRATION_FILE)

    assert calibration.parts[-5:] == (
        "handtrack",
        "cameras",
        "optitrack",
        "calibration_data",
        "multi_camera_calib_latest.npz",
    )
    assert calibration.is_file()


def test_camera_acquisition_retries_every_missing_index(monkeypatch):
    sdk = _DelayedCameraSdk()
    monkeypatch.setattr(multi_mjpeg, "optitrack_cam", sdk)
    monkeypatch.setattr(multi_mjpeg, "CameraWorker", _FakeWorker)
    monkeypatch.setattr(multi_mjpeg.time, "sleep", lambda _seconds: None)

    manager = multi_mjpeg.CameraManager(discovery_wait=0)
    workers = manager.start_all(acquisition_rounds=2, retry_delay=0)

    assert [worker.index for worker in workers] == [0, 1, 2]
    assert sdk.calls == {0: 2, 1: 2, 2: 2}


def test_camera_start_fails_before_workers_start_when_a_handle_is_missing(
    monkeypatch,
):
    sdk = _DelayedCameraSdk(never_ready={1})
    monkeypatch.setattr(multi_mjpeg, "optitrack_cam", sdk)
    monkeypatch.setattr(multi_mjpeg, "CameraWorker", _FakeWorker)
    monkeypatch.setattr(multi_mjpeg.time, "sleep", lambda _seconds: None)

    manager = multi_mjpeg.CameraManager(discovery_wait=0)

    with pytest.raises(RuntimeError, match=r"indices \[1\] after 2 attempts"):
        manager.start_all(acquisition_rounds=2, retry_delay=0)

    assert manager.workers == []
    assert sdk.cameras[0].released
    assert sdk.cameras[2].released
