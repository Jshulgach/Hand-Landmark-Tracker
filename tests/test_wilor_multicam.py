"""Shared model ownership, camera association and responsive multi-view preview."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from mavis_track import WiLoRMultiViewTracker
from mavis_track import wilor
from mavis_track._capture import FrameBuffer
from mavis_track._illumination import IlluminationState
from handtrack.applications import _wilor_multicam as sessions
from unity_hand_tracking.optitrack_cam_py import mocap_handtrack_gui as gui


def wait_until(predicate, app=None, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if app is not None:
            app.processEvents()
        if predicate():
            return
        time.sleep(.003)
    raise AssertionError("Timed out waiting for multi-camera state")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def models(monkeypatch, tmp_path):
    for name in wilor.missing_assets(tmp_path):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    state = SimpleNamespace(loads=0, closes=0, batches=[], load_gate=None,
                            infer_gate=None, entered=threading.Event(), fail=False)
    class Runtime:
        def __init__(self, root, device, fast):
            state.loads += 1
            self.faces = np.array([[0, 1, 2]], np.int32)
            self.device = "cpu"
            self.cfg = SimpleNamespace(EXTRA=SimpleNamespace(FOCAL_LENGTH=5000),
                                       MODEL=SimpleNamespace(IMAGE_SIZE=256))
            if state.load_gate is not None:
                state.entered.set()
                assert state.load_gate.wait(5)
        def infer_many(self, images, confidence, max_hands, *, batch_size):
            state.batches.append((len(images), batch_size, confidence))
            if state.infer_gate is not None:
                state.entered.set()
                assert state.infer_gate.wait(5)
            if state.fail:
                raise RuntimeError("CUDA inference failed")
            results = []
            for image in images:
                h, w = image.shape[:2]
                joints = np.zeros((21, 3), np.float32)
                joints[:, 0] = np.linspace(0, .1, 21)
                joints[:, 1] = np.linspace(0, .05, 21)
                results.append([dict(
                    joints=joints, vertices=np.zeros((778, 3), np.float32),
                    camera=np.array([1., 0, 0]), center=np.array([w / 2, h / 2]),
                    box_size=40., bbox=np.array([0, 0, w, h]), score=.9, right=1)])
            return results
        def close(self):
            state.closes += 1
    monkeypatch.setattr(wilor, "_OfficialWiLoR", Runtime)
    return tmp_path, state


@pytest.fixture
def cameras(monkeypatch):
    state = SimpleNamespace(managers=[], capture_stop=threading.Event())
    class Buffer(FrameBuffer):
        def __init__(self):
            self.lock = threading.Lock()
            self.ready = threading.Event()
            self.latest_frame = None
            self.frame_timestamp = None
            self.width, self.height = 96, 72
            self.connected = True
    class Manager:
        num_cameras = 4
        camera_indices = [0, 1, 2, 3]
        def __init__(self):
            self.workers = [Buffer() for _ in range(4)]
            self.stopped = False
            self.exposures = []
            self.stop_event = threading.Event()
            self.counter = 0
            state.managers.append(self)
        def start_all(self, exposure=None):
            def capture():
                while not self.stop_event.is_set():
                    self.counter += 1
                    for index, worker in enumerate(self.workers):
                        if worker.connected:
                            worker.publish_frame(np.full(
                                (72, 96, 3), (self.counter + index) % 255, np.uint8))
                    self.stop_event.wait(.01)
            self.capture = threading.Thread(target=capture, daemon=True)
            self.capture.start()
        def wait_ready(self):
            for worker in self.workers:
                assert worker.ready.wait(2)
        def get_resolution(self, index):
            return (96, 72)
        def set_exposure(self, index, value):
            self.exposures.append((index, value))
        def get_illumination(self, index):
            return IlluminationState(intensity=15, minimum=0, maximum=15)
        def set_illumination(self, index, **kwargs):
            return self.get_illumination(index)
        def stop_all(self):
            self.stop_event.set()
            self.capture.join(2)
            assert not self.capture.is_alive()
            self.stopped = True
    monkeypatch.setattr(sessions, "camera_manager", Manager)
    monkeypatch.setattr(gui.StereoHandTrackerGUI, "start_udp_broadcast", lambda self: None)
    monkeypatch.setattr(gui.StereoHandTrackerGUI, "stop_udp_broadcast", lambda self: None)
    return state


def test_one_runtime_for_four_views_with_separate_ids_and_missing_camera(models):
    root, state = models
    frame = np.zeros((72, 96, 3), np.uint8)
    with WiLoRMultiViewTracker(root=root, batch_size=2) as tracker:
        first = tracker.process_views([frame, frame, None, frame], timestamp=1)
        assert state.loads == 1 and state.batches == [(3, 2, .3)]
        assert len(first) == 4 and first[2] is None
        assert [first[index].source_id for index in (0, 1, 3)] == [
            "frames:camera:0", "frames:camera:1", "frames:camera:3"]
        assert all(first[index].hands[0].track_id == 0 for index in (0, 1, 3))
        second = tracker.process_views([None, frame, frame, frame], timestamp=1.1)
        third = tracker.process_views([frame, frame, frame, frame], timestamp=1.2)
        assert second[1].hands[0].track_id == first[1].hands[0].track_id
        assert third[0].hands[0].track_id != first[0].hands[0].track_id
        assert third[1].hands[0].track_id == first[1].hands[0].track_id
        assert not np.shares_memory(third[0].hands[0].joints, third[1].hands[0].joints)
    tracker.close()
    assert state.closes == 1
    with pytest.raises(RuntimeError, match="closed"):
        tracker.process_views([frame])


def test_view_validation_is_atomic_and_preserves_resolution(models):
    root, state = models
    with WiLoRMultiViewTracker(root=root) as tracker:
        with pytest.raises(ValueError, match="frame"):
            tracker.process_views([np.zeros((72, 96, 3), np.uint8), np.zeros((4, 4))], timestamp=1)
        assert state.batches == []
        results = tracker.process_views([np.zeros((72, 96, 3), np.uint8),
                                         np.zeros((144, 192, 3), np.uint8)], timestamp=1)
        assert [result.image_size for result in results] == [(96, 72), (192, 144)]
        assert all(result.frame_id == 0 for result in results)


def test_four_live_tiles_and_controls_work_during_blocked_model_load(models, cameras, app):
    root, model_state = models
    gate = threading.Event()
    model_state.load_gate = gate
    window = gui.StereoHandTrackerGUI(model="wilor", wilor_root=root)
    timer = QTimer()
    ticks = []
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(5)
    try:
        window.show()
        window.start_tracking()
        wait_until(model_state.entered.is_set, app)
        wait_until(lambda: len(window.video_labels) == 4 and
                   all(label.pixmap() is not None for label in window.video_labels), app)
        manager = cameras.managers[0]
        count = manager.counter
        wait_until(lambda: len(ticks) >= 8 and manager.counter > count + 2, app)
        assert not window.show_3d_checkbox.isEnabled()
        assert not window.udp_checkbox.isEnabled()
        window.exposure_spinboxes[2].setValue(25)
        assert (2, 25) in manager.exposures
        assert not window.model_combo.isEnabled()
        window.close()
        assert window._wilor_session is not None and not manager.stopped
        gate.set()
        wait_until(lambda: window._wilor_session is None, app)
        assert manager.stopped and model_state.closes == 1
        assert not window.isVisible()
    finally:
        gate.set()
        timer.stop()
        window.close()
        wait_until(lambda: window._wilor_session is None, app)


def test_matched_packet_stays_paired_and_camera_loss_clears_tile(models, cameras, app):
    root, model_state = models
    gate = threading.Event()
    model_state.infer_gate = gate
    window = gui.StereoHandTrackerGUI(model="wilor", wilor_root=root)
    try:
        window.show()
        window.start_tracking()
        wait_until(model_state.entered.is_set, app)
        gate.set()
        wait_until(lambda: bool(window._wilor_session.snapshot()[0]), app)
        wait_until(lambda: "4" in window.camera_info_label.text(), app)
        packet, _, _ = window._wilor_session.snapshot()
        assert len(packet.frames) == len(packet.results) == 4
        assert all(result.hands for result in packet.results)
        assert all(not frame.flags.writeable for frame in packet.frames)
        assert model_state.loads == 1 and model_state.batches[0][:2] == (4, 2)
        assert not window.tracker.calibration_available
        manager = cameras.managers[0]
        manager.workers[1].connected = False
        with manager.workers[1].lock:
            manager.workers[1].frame_timestamp = None
        wait_until(lambda: "fresh" in window.video_labels[1].text(), app)
        assert window.video_labels[1].pixmap() is None
        window.wilor_view_combo.setCurrentIndex(1)
        assert window.wilor_view_combo.currentData() == "live"
        window.stop_tracking()
        wait_until(lambda: window._wilor_session is None, app)
        assert manager.stopped and model_state.closes == 1
    finally:
        gate.set()
        window.close()
        wait_until(lambda: window._wilor_session is None, app)


def test_inference_failure_keeps_fresh_camera_preview_until_stop(models, cameras):
    root, model_state = models
    model_state.fail = True
    session = sessions.WiLoRCameraSession(root=root)
    session.start()
    try:
        wait_until(lambda: session.error is not None)
        packet, state, error = session.snapshot()
        assert packet is None and "CUDA inference failed" in error
        assert session.is_alive() and session.ready.is_set()
        assert all(frame is not None for frame, _ in session.raw_snapshots())
        assert not cameras.managers[0].stopped
    finally:
        session.request_stop()
        session.join(5)
    assert not session.is_alive() and cameras.managers[0].stopped


def test_cli_routes_wilor_options_to_optitrack_multicam(monkeypatch):
    from handtrack.applications import _cli
    called = []
    monkeypatch.setattr(_cli, "_run_module_entrypoint",
                        lambda module, argv=None: called.append((module, argv)) or 0)
    assert _cli.main(["gui", "--backend", "optitrack", "--model", "wilor",
                      "--wilor-device", "cuda", "--wilor-fast",
                      "--wilor-batch-size", "1"]) == 0
    module, args = called[0]
    assert module.endswith("optitrack_cam_py.mocap_handtrack_gui")
    assert args[args.index("--model") + 1] == "wilor"
    assert args[args.index("--wilor-device") + 1] == "cuda"
    assert args[args.index("--wilor-batch-size") + 1] == "1"
    assert "--wilor-fast" in args



def test_unchanged_capture_is_not_reprocessed_and_startup_stop_is_cooperative(models, monkeypatch):
    root, model_state = models
    frame = np.zeros((72, 96, 3), np.uint8)
    frame.flags.writeable = False
    stamp = time.perf_counter()
    stopped = []
    manager = SimpleNamespace(
        num_cameras=2, camera_indices=[0, 1],
        workers=[SimpleNamespace(get_snapshot=lambda: (frame, stamp)) for _ in range(2)],
        start_all=lambda **kwargs: None, wait_ready=lambda: None,
        get_resolution=lambda index: (96, 72), stop_all=lambda: stopped.append(1))
    monkeypatch.setattr(sessions, "camera_manager", lambda: manager)
    session = sessions.WiLoRCameraSession(root=root)
    session.start()
    try:
        wait_until(lambda: session.snapshot()[0] is not None)
        time.sleep(.04)
        assert len(model_state.batches) == 1
    finally:
        session.request_stop()
        session.join(5)
    assert stopped == [1] and model_state.closes == 1

    # A stop during SDK discovery must not start capture or load another model.
    gate = threading.Event()
    entered = threading.Event()
    def discover():
        entered.set()
        assert gate.wait(5)
        return manager
    monkeypatch.setattr(sessions, "camera_manager", discover)
    second = sessions.WiLoRCameraSession(root=root)
    second.start()
    try:
        assert entered.wait(2)
        second.request_stop()
        assert second.is_alive() and not second.ready.is_set()
    finally:
        gate.set()
        second.join(5)
    assert not second.is_alive() and model_state.loads == 1
    assert stopped == [1, 1]



def test_missing_views_still_validate_clock_and_color(models):
    root, state = models
    with WiLoRMultiViewTracker(root=root) as tracker:
        for stamp in (float("nan"), float("inf"), -1):
            with pytest.raises(ValueError, match="timestamps"):
                tracker.process_views([None, None], timestamp=stamp)
        with pytest.raises(ValueError, match="color"):
            tracker.process_views([None, None], color="gray")
        assert tracker.process_views([None, None], timestamp=1) == (None, None)
        assert state.batches == []
