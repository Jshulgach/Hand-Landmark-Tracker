"""A terminal SDK shutdown must not happen during tracking stop/model switches."""
import threading
from types import SimpleNamespace

import pytest

from mavis_track._optitrack_runtime import SDKRuntime
from unity_hand_tracking.optitrack_cam_py import multi_mjpeg


class TerminalSDK:
    def __init__(self):
        self.init_calls = self.shutdown_calls = 0
        self.dead = False

    def initialize_sdk(self):
        self.init_calls += 1
        # The existing binding accesses the singleton; it cannot revive it.

    def camera_count(self):
        return 0 if self.dead else 4

    def shutdown_sdk(self):
        self.shutdown_calls += 1
        self.dead = True


def test_sdk_is_initialized_once_and_exclusive_until_the_old_owner_releases():
    sdk = TerminalSDK()
    runtime = SDKRuntime(sdk)
    first, second = object(), object()
    assert runtime.acquire(first)
    with pytest.raises(RuntimeError, match="another capture session"):
        runtime.acquire(second)
    runtime.release(first)
    assert not runtime.acquire(second)
    runtime.release(first)  # Stale cleanup must not release the new owner.
    assert runtime.owner is second
    runtime.release(second)
    assert sdk.camera_count() == 4 and sdk.init_calls == 1 and sdk.shutdown_calls == 0
    runtime.close()
    runtime.close()
    assert sdk.shutdown_calls == 1
    with pytest.raises(RuntimeError, match="restart the application"):
        runtime.acquire(first)


def test_final_shutdown_stops_owner_before_shutting_the_sdk():
    sdk = TerminalSDK()
    runtime = SDKRuntime(sdk)
    events = []
    def stop():
        assert sdk.shutdown_calls == 0
        events.append("handles released")
        runtime.release(owner)
    owner = SimpleNamespace(stop_all=stop)
    runtime.acquire(owner)
    runtime.close()
    assert events == ["handles released"] and sdk.shutdown_calls == 1


def test_stop_timeout_retains_ownership_and_allows_retry_after_worker_finishes(monkeypatch):
    sdk = TerminalSDK()
    monkeypatch.setattr(multi_mjpeg, "optitrack_cam", sdk)
    first = multi_mjpeg.CameraManager(discovery_wait=0)
    worker = SimpleNamespace(name="blocked", alive=True, stop=lambda: None,
                             join=lambda **kwargs: None)
    worker.is_alive = lambda: worker.alive
    first.workers.append(worker)
    try:
        with pytest.raises(RuntimeError, match="did not stop"):
            first.stop_all()
        with pytest.raises(RuntimeError, match="another capture session"):
            multi_mjpeg.CameraManager(discovery_wait=0)
        assert sdk.init_calls == 1 and sdk.shutdown_calls == 0
        worker.alive = False
        first.stop_all()
        second = multi_mjpeg.CameraManager(discovery_wait=0)
        assert second.num_cameras == 4
        first.stop_all()  # Old manager cannot release the new lease.
        with pytest.raises(RuntimeError, match="another capture session"):
            multi_mjpeg.CameraManager(discovery_wait=0)
        second.stop_all()
    finally:
        worker.alive = False
        first.stop_all()
        first._runtime.close()
    assert sdk.shutdown_calls == 1


def test_final_shutdown_does_not_deadlock_with_capture_stop():
    sdk = TerminalSDK()
    runtime = SDKRuntime(sdk)
    stop_lock = threading.RLock()
    entered, proceed = threading.Event(), threading.Event()
    def stop():
        with stop_lock:
            runtime.release(owner)
    owner = SimpleNamespace(stop_all=stop)
    runtime.acquire(owner)
    def stopping():
        with stop_lock:
            entered.set()
            assert proceed.wait(3)
            runtime.release(owner)
    capture_stop = threading.Thread(target=stopping, daemon=True)
    app_close = threading.Thread(target=runtime.close, daemon=True)
    capture_stop.start()
    assert entered.wait(2)
    app_close.start()
    proceed.set()
    capture_stop.join(3)
    app_close.join(3)
    assert not capture_stop.is_alive() and not app_close.is_alive()
    assert sdk.shutdown_calls == 1



def test_discovery_failure_releases_owner_without_terminating_sdk(monkeypatch):
    class Sdk(TerminalSDK):
        calls = 0
        def camera_count(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("discovery failed")
            return super().camera_count()
    sdk = Sdk()
    monkeypatch.setattr(multi_mjpeg, "optitrack_cam", sdk)
    with pytest.raises(RuntimeError, match="discovery failed"):
        multi_mjpeg.CameraManager(discovery_wait=0)
    manager = multi_mjpeg.CameraManager(discovery_wait=0)
    try:
        assert manager.num_cameras == 4 and sdk.init_calls == 1 and sdk.shutdown_calls == 0
    finally:
        manager.stop_all()
        manager._runtime.close()
