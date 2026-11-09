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
from mavis_track.wilor import WiLoRHandResult, WiLoRFrameResult
from mavis_track._capture import FrameBuffer
from mavis_track._illumination import IlluminationState
from handtrack.applications import _wilor_multicam as sessions
from unity_hand_tracking.optitrack_cam_py import mocap_handtrack_gui as gui


def test_calibrated_wilor_gui_renders_mesh_and_skeleton(cameras, app, monkeypatch, tmp_path):
    import cv2
    rng=np.random.default_rng(641)
    joints=rng.normal(0,.03,(21,3));joints[0]=0
    joints+=np.array([0.,0.,1.2])
    vertices=rng.normal(0,.03,(778,3))+joints[0]
    matrix=np.array([[100.,0,48],[0,100.,36],[0,0,1]])
    saved=dict(schema_version=2,num_cameras=4,image_sizes=np.array([[96,72]]*4),
               camera_ids=np.arange(4),quality_passed=True,coordinate_units='meters')
    predictions=[]
    for c in range(4):
        translation=np.array([[-.15*c],[0.],[0.]])
        for prefix,value in [('camera_matrix',matrix),('dist_coeffs',np.zeros(5)),
                             ('R',np.eye(3)),('T',translation)]:
            saved[f'{prefix}_{c}']=value
        pixels,_=cv2.projectPoints(joints,np.zeros(3),translation,matrix,np.zeros(5))
        mesh_pixels,_=cv2.projectPoints(vertices,np.zeros(3),translation,matrix,np.zeros(5))
        hand=WiLoRHandResult(c,'Right',.9,pixels.reshape(21,2),mesh_pixels.reshape(778,2),
            joints-joints[0],vertices-joints[0],np.array([999.,999.,999.]),np.array([0.,0.,96.,72.]),{})
        predictions.append(WiLoRFrameResult(0,1.,str(c),(96,72),(hand,),5000.,'caller'))
    path=tmp_path/'calibration.npz'
    np.savez(path,**saved)
    monkeypatch.setattr(sessions,'calibration_file',lambda:path)
    class Model:
        faces=np.array([[0,1,2],[3,4,5]],np.int32)
        device='cpu'
        def __init__(self,**kwargs):pass
        def process_views(self,frames,**kwargs):
            return tuple(predictions[c] if frame is not None else None for c,frame in enumerate(frames))
        def close(self):pass
    monkeypatch.setattr(sessions,'WiLoRMultiViewTracker',Model)
    window=gui.StereoHandTrackerGUI(model='wilor')
    rendered=[]
    original_render=window.update_3d_view
    def render(raw,filtered,meshes=None,faces=None):
        rendered.append((filtered,meshes,faces))
        return original_render(raw,filtered,meshes=meshes,faces=faces)
    window.update_3d_view=render
    try:
        window.show()
        window.start_tracking()
        wait_until(lambda: window._wilor_session.snapshot()[0] is not None and
                   bool(window._wilor_session.snapshot()[0].calibrated_hands),app)
        wait_until(window.show_3d_checkbox.isEnabled,app)
        assert 'Calibration ready' in window.calibration_label.text()
        assert window._wilor_session.calibration_identity_verified
        # Exercise every projection, including the fifth palm-plane view.
        window.show_3d_top=window.show_3d_front=window.show_3d_side=True
        window.show_3d_iso=window.show_3d_palm=True
        window.show_3d_checkbox.setChecked(True)
        wait_until(lambda: bool(rendered),app)
        assert window.vis_3d_label.pixmap() is not None
        assert 'calibrated WiLoR' in window.tracked_hand_label.text()
        np.testing.assert_allclose(rendered[-1][0][0],joints,atol=1e-8)
        np.testing.assert_allclose(rendered[-1][1][0],vertices,atol=1e-8)
        np.testing.assert_array_equal(rendered[-1][2],Model.faces)
        for worker in cameras.managers[0].workers[1:]:
            worker.connected=False
        wait_until(lambda: 'waiting for consistent' in window.tracked_hand_label.text(),app)
        pixmap=window.vis_3d_label.pixmap()
        assert pixmap is None or pixmap.isNull()
        for worker in cameras.managers[0].workers:
            worker.connected=True
        wait_until(lambda: 'calibrated WiLoR' in window.tracked_hand_label.text(),app)
        # Removing the second viewpoint must clear the old mesh, not replace
        # it with the model's arbitrary monocular camera translation.
        window._wilor_session.set_enabled_cameras({0})
        wait_until(lambda: 'waiting for consistent' in window.tracked_hand_label.text(),app)
        pixmap=window.vis_3d_label.pixmap()
        assert pixmap is None or pixmap.isNull()
    finally:
        window.close()
        wait_until(lambda: window._wilor_session is None,app)


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
                            infer_gate=None, entered=threading.Event(), fail=False, empty=False)
    class Runtime:
        def __init__(self, root, device, fast, progress=None):
            state.loads += 1
            if progress:
                progress("Loading reconstruction checkpoint")
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
            if state.empty:
                return [[] for image in images]
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
        assert "Loading reconstruction checkpoint" in window.status_label.text()
        assert "stage" in window.fps_label.text()
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


def test_inference_toggle_keeps_cameras_live_and_defers_model_load(models, cameras, app):
    root, state = models
    window = gui.StereoHandTrackerGUI(model="wilor", wilor_root=root)
    try:
        window.inference_checkbox.setChecked(False)
        window.start_tracking()
        wait_until(lambda: window._wilor_session.ready.is_set(), app)
        wait_until(lambda: len(window.video_labels) == 4 and
                   all(label.pixmap() is not None for label in window.video_labels), app)
        manager = cameras.managers[0]
        assert state.loads == 0
        count = manager.counter
        wait_until(lambda: manager.counter > count + 3, app)
        assert state.loads == 0 and not manager.stopped
        window.inference_checkbox.setChecked(True)
        wait_until(lambda: window._wilor_session.snapshot()[0] is not None, app)
        window.inference_checkbox.setChecked(False)
        assert window._wilor_session.snapshot()[0] is None
        batches = window._wilor_session.completed_batches
        count = manager.counter
        wait_until(lambda: manager.counter > count + 4, app)
        assert window._wilor_session.completed_batches == batches
        assert not manager.stopped and not manager.inference_enabled
        window.inference_checkbox.setChecked(True)
        wait_until(lambda: window._wilor_session.completed_batches > batches, app)
        assert state.loads == 1
    finally:
        window.close()
        wait_until(lambda: window._wilor_session is None, app)


@pytest.mark.parametrize('network', [False, True])
def test_recording_without_inference_flushes_before_gui_closes(models, cameras, app, monkeypatch, tmp_path, network):
    from types import MethodType
    from mavis_track import _camera_recording
    from unity_hand_tracking.optitrack_cam_py.multi_mjpeg import CameraManager
    gate, entered = threading.Event(), threading.Event()
    class Writer:
        def __init__(self, *args): pass
        def isOpened(self): return True
        def write(self, image):
            entered.set()
            assert gate.wait(3)
        def release(self): pass
    monkeypatch.setattr(_camera_recording.cv2, 'VideoWriter', Writer)
    monkeypatch.setattr(gui.QFileDialog, 'getExistingDirectory', lambda *args: str(tmp_path))
    root, model_state = models
    from mavis_track._nexus_remote import NexusRemote
    sent = []
    def send(self, action, name):
        sent.append((action, name))
        return dict(action=action, trial_name=name, status='sent_unconfirmed')
    monkeypatch.setattr(NexusRemote, 'send', send)
    window = gui.StereoHandTrackerGUI(model='wilor', wilor_root=root)
    window.inference_checkbox.setChecked(False)
    window.trial_name_edit.setText('Reach_A')
    window.nexus_network_checkbox.setChecked(network)
    try:
        window.show()
        window.start_tracking()
        wait_until(lambda: window._wilor_session.ready.is_set(), app)
        manager = cameras.managers[0]
        manager._stop_lock = threading.RLock()
        manager._shutdown = False
        manager.recorder = None
        manager.start_recording = MethodType(CameraManager.start_recording, manager)
        manager.stop_recording = MethodType(CameraManager.stop_recording, manager)
        manager.get_sync_status = lambda: dict(available=False, reason='No input edge telemetry')
        for index, worker in enumerate(manager.workers):
            worker.capture_settings = dict(camera_index=index)
            original = worker.publish_frame
            def publish(frame, original=original, index=index):
                original(frame)
                recorder = manager.recorder
                if recorder is not None:
                    recorder.frame(index, frame, dict(host_received_monotonic=time.perf_counter(),
                                                     inference_enabled=False))
            worker.publish_frame = publish
        window._toggle_recording()
        wait_until(entered.is_set, app)
        recorder = manager.recorder
        assert model_state.loads == 0
        window.close()
        assert window.isVisible() and not manager.stopped
        count = manager.counter
        wait_until(lambda: manager.counter > count + 3, app)
        assert window._record_stop_thread is not None
        gate.set()
        wait_until(lambda: window._wilor_session is None and not window.isVisible(), app)
        assert recorder.status()['written'] > 0 and manager.recorder is None and manager.stopped
        import json
        manifest = json.loads((recorder.directory / 'manifest.json').read_text())
        assert manifest['complete'] and not manifest['input_edge_recording_supported']
        assert manifest['trial_name'] == 'Reach_A' and recorder.directory.name.startswith('Reach_A_')
        assert window.record_directory_edit.text() == str(tmp_path)
        assert sent == ([('start', 'Reach_A'), ('stop', 'Reach_A')] if network else [])
        if network:
            logged = [json.loads(line) for line in (recorder.directory/'control_events.jsonl').read_text().splitlines()]
            assert [row['action'] for row in logged] == ['start', 'stop']
            assert all(row['status'] == 'sent_unconfirmed' for row in logged)
    finally:
        gate.set()
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



def test_mediapipe_wilor_switches_reopen_four_cameras_without_terminal_sdk_shutdown(
        models, app, monkeypatch, tmp_path):
    from mavis_track._optitrack_runtime import capture_runtime
    from unity_hand_tracking.optitrack_cam_py import multi_mjpeg, mocap_tracker

    root, model_state = models

    class Frame:
        def __init__(self, camera):
            self.camera = camera
            self.released = False
            camera.frames += 1
        def rasterize(self, width, height):
            return np.full((height, width, 4), 50 + self.camera.index * 30, np.uint8)
        def release(self):
            assert not self.released
            self.released = True
            self.camera.frames -= 1

    class Camera:
        def __init__(self, sdk, index):
            self.sdk, self.index = sdk, index
            self.frames = 0
            self.running = self.released = False
            self.exposure = 15
            self.intensity = 15
        def width(self): return 96
        def height(self): return 72
        def set_video_type(self, value): pass
        def set_exposure(self, value): self.exposure = value
        def get_exposure(self): return self.exposure
        def set_aec(self, value): pass
        def set_agc(self, value): pass
        def set_text_overlay(self, value): pass
        def get_intensity(self): return self.intensity
        def min_intensity(self): return 0
        def max_intensity(self): return 15
        def set_intensity(self, value): self.intensity = value; return value
        def is_high_power_supported(self): return False
        def is_high_power_available(self): return False
        def get_high_power(self): return False
        def set_high_power(self, value): raise ValueError("unsupported")
        def start(self): self.running = True
        def stop(self): self.running = False
        def get_latest_frame(self): return Frame(self) if self.running else None
        def release(self):
            assert not self.released and not self.running and self.frames == 0
            self.released = True

    class TerminalSdk:
        # Mirrors the existing binding: initialize accesses a singleton;
        # shutdown is terminal and subsequent initialize cannot revive it.
        def __init__(self):
            self.initializations = self.shutdowns = 0
            self.dead = False
            self.leases = []
        def initialize_sdk(self): self.initializations += 1
        def camera_count(self): return 0 if self.dead else 4
        def get_camera_by_index(self, index):
            assert not self.dead
            camera = Camera(self, index)
            self.leases.append(camera)
            return camera
        def shutdown_sdk(self):
            assert all(camera.released for camera in self.leases)
            self.shutdowns += 1
            self.dead = True

    class Detector:
        def __init__(self, **kwargs): pass
        def process(self, image): return None
        def close(self): pass

    sdk = TerminalSdk()
    monkeypatch.setattr(multi_mjpeg, "optitrack_cam", sdk)
    manager = lambda: multi_mjpeg.CameraManager(discovery_wait=0)
    monkeypatch.setattr(mocap_tracker, "CameraManager", manager)
    monkeypatch.setattr(sessions, "camera_manager", manager)
    monkeypatch.setattr(mocap_tracker, "CALIBRATION_FILE", str(tmp_path / "no-calibration.npz"))
    monkeypatch.setattr(mocap_tracker.MultiCameraTracker, "_load_mediapipe_hands_class",
                        staticmethod(lambda: Detector))
    monkeypatch.setattr(gui, "_preload_mediapipe_runtime", lambda: (True, ""))
    monkeypatch.setattr(gui, "LSLBroadcaster", lambda **kwargs: None)
    monkeypatch.setattr(gui.StereoHandTrackerGUI, "start_udp_broadcast", lambda self: None)
    monkeypatch.setattr(gui.StereoHandTrackerGUI, "stop_udp_broadcast", lambda self: None)
    window = gui.StereoHandTrackerGUI(model="mediapipe", wilor_root=root)
    window._mediapipe_ready = True
    try:
        window.show()
        for mode in ("mediapipe", "wilor", "mediapipe", "wilor", "wilor"):
            window.model_combo.setCurrentIndex(window.model_combo.findData(mode))
            window.start_tracking()
            if mode == "wilor":
                wait_until(lambda: window._wilor_session.snapshot()[0] is not None, app)
            wait_until(lambda: len(window.video_labels) == 4 and
                       all(label.pixmap() is not None for label in window.video_labels), app)
            assert window.num_cameras == 4
            assert sdk.initializations == 1 and sdk.shutdowns == 0
            window.exposure_spinboxes[0].setValue(20 if mode == "wilor" else 21)
            window.stop_tracking()
            wait_until(lambda: window._wilor_session is None, app)
            assert all(camera.released for camera in sdk.leases)
            assert sdk.shutdowns == 0
            assert window.model_combo.isEnabled() and window.start_btn.isEnabled()
        assert len(sdk.leases) == 20
        assert model_state.loads == model_state.closes == 3
    finally:
        window.close()
        wait_until(lambda: window._wilor_session is None, app)
        capture_runtime(sdk).close()
    assert sdk.shutdowns == 1


def test_first_inference_is_not_reported_as_model_loading(models, cameras, app):
    root, model_state = models
    gate = threading.Event()
    model_state.infer_gate = gate
    window = gui.StereoHandTrackerGUI(model="wilor", wilor_root=root)
    try:
        window.show()
        window.start_tracking()
        wait_until(model_state.entered.is_set, app)
        wait_until(lambda: "First inference batch" in window.status_label.text(), app)
        assert window._wilor_session.device == "cpu"
        assert window._wilor_session.completed_batches == 0
        assert "CPU" in window.fps_label.text()
        assert "Loading" not in window.fps_label.text()
        assert "no inference result yet" in window.timing_label.text()
        assert all(label.pixmap() is not None for label in window.video_labels)
        gate.set()
        wait_until(lambda: "Batch #" in window.timing_label.text(), app)
        assert window._wilor_session.completed_batches >= 1
        assert "CPU" in window.fps_label.text()
    finally:
        gate.set()
        window.close()
        wait_until(lambda: window._wilor_session is None, app)


def test_completed_zero_detection_batches_and_raw_display_are_explicit(models, cameras, app):
    root, model_state = models
    model_state.empty = True
    window = gui.StereoHandTrackerGUI(model="wilor", wilor_root=root)
    try:
        window.show()
        window.start_tracking()
        wait_until(lambda: "No hands detected" in window.status_label.text(), app)
        assert window.hands_label.text() == "Hands/view: 0"
        assert "Batch #" in window.timing_label.text()
        assert "Loading" not in window.fps_label.text()
        window.wilor_view_combo.setCurrentIndex(1)
        wait_until(lambda: "overlays off" in window.status_label.text(), app)
    finally:
        window.close()
        wait_until(lambda: window._wilor_session is None, app)


def test_model_load_failure_prints_exception_chain_and_interpreter(cameras, monkeypatch, capsys):
    import sys
    def fail(**options):
        options["progress"]("Loading reconstruction checkpoint")
        try:
            raise ImportError("missing_example_dependency")
        except ImportError as exc:
            raise RuntimeError("WiLoR initialization failed") from exc
    monkeypatch.setattr(sessions, "WiLoRMultiViewTracker", fail)
    session = sessions.WiLoRCameraSession()
    session.start()
    try:
        wait_until(lambda: session.error is not None)
        assert session.ready.is_set() and session.is_alive()
        assert session.completed_batches == 0 and session.device is None
        assert all(frame is not None for frame, _ in session.raw_snapshots())
    finally:
        session.request_stop()
        session.join(5)
    output = capsys.readouterr()
    assert "Loading reconstruction checkpoint" in output.out
    assert sys.executable in output.err
    assert "missing_example_dependency" in output.err
    assert "WiLoR initialization failed" in output.err
    assert "Traceback" in output.err


def test_api_forwards_optional_startup_progress(models):
    root, _ = models
    stages = []
    with WiLoRMultiViewTracker(root=root, progress=stages.append):
        assert stages == ["Loading reconstruction checkpoint"]
    with pytest.raises(TypeError, match="progress"):
        WiLoRMultiViewTracker(root=root, progress="not a callback")
