"""Video acquisition must survive inference stalls and expose recording loss."""
import json
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from mavis_track import _camera_recording as recording
from unity_hand_tracking.optitrack_cam_py.multi_mjpeg import CameraWorker


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_multicamera_roundtrip_preserves_frames_and_separate_timing(tmp_path):
    directory = tmp_path / "session"
    writer = recording.CameraRecording(directory, nominal_fps=100, camera_ids=[0, 1])
    for frame_id in range(4):
        for camera in range(2):
            image = np.full((48, 64, 3), 30 + frame_id * 20 + camera, np.uint8)
            writer.frame(camera, image, dict(host_received_monotonic=10 + frame_id * .013,
                         sdk_frame_id=frame_id, hardware_timestamp_ticks=frame_id * 13,
                         hardware_timestamp_frequency=1000, inference_enabled=False))
    writer.sync(dict(host_sampled_monotonic=10, input_level=None, input_edges=None,
                     input_edge_recording_supported=False, input_source=2))
    writer.close()
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["complete"] and manifest["frames_written"] == 8
    assert not manifest["input_edge_recording_supported"]
    metadata = rows(directory / "frames.jsonl")
    assert len(metadata) == 8 and not any(row["inference_enabled"] for row in metadata)
    for camera in range(2):
        selected = [row for row in metadata if row["camera_index"] == camera]
        assert [row["video_frame_index"] for row in selected] == list(range(4))
        assert [row["sdk_frame_id"] for row in selected] == list(range(4))
        video = cv2.VideoCapture(str(directory / manifest['video_files'][str(camera)]))
        try:
            for frame_id in range(4):
                ok, image = video.read()
                assert ok and image.shape == (48, 64, 3)
                assert abs(image.mean() - (30 + frame_id * 20 + camera)) < 3
            assert not video.read()[0]
        finally:
            video.release()
    assert rows(directory / "sync.jsonl")[0]["input_edges"] is None


def test_slow_disk_never_blocks_capture_and_reports_overload(monkeypatch, tmp_path):
    entered, release = threading.Event(), threading.Event()
    class Writer:
        def __init__(self, *args): pass
        def isOpened(self): return True
        def write(self, image):
            entered.set()
            assert release.wait(3)
        def release(self): pass
    monkeypatch.setattr(recording.cv2, "VideoWriter", Writer)
    writer = recording.CameraRecording(tmp_path / "session", queue_size=1)
    image = np.zeros((48, 64, 3), np.uint8)
    try:
        writer.frame(0, image, {})
        assert entered.wait(2)
        writer.frame(0, image, {})
        writer.frame(0, image, {})
        assert writer.status()["dropped"] == 1
        assert writer.status()["pending"] == 1
    finally:
        release.set()
        writer.close()
    assert writer.status()["written"] == 2


def test_writer_failure_is_visible_and_manifest_incomplete(monkeypatch, tmp_path):
    monkeypatch.setattr(recording.cv2, "VideoWriter", lambda *args:
                        SimpleNamespace(isOpened=lambda: False, release=lambda: None))
    writer = recording.CameraRecording(tmp_path / "session")
    writer.frame(0, np.zeros((48, 64, 3), np.uint8), {})
    with pytest.raises(RuntimeError, match="Cannot open"):
        writer.close()
    manifest = json.loads((writer.directory / "manifest.json").read_text())
    assert not manifest["complete"] and manifest["error"]


@pytest.mark.parametrize("codec", ["MJPG", "FFV1"])
def test_recording_and_frozen_calibration(tmp_path, codec):
    if codec == "FFV1":
        probe = cv2.VideoWriter(str(tmp_path/"codec-probe.avi"),
                                cv2.VideoWriter_fourcc(*codec), 30., (64, 48))
        available = probe.isOpened()
        probe.release()
        if not available:
            pytest.skip("Installed OpenCV build cannot encode optional FFV1 video")
    calibration = tmp_path/'original.npz'
    np.savez(calibration, schema_version=2, quality_passed=True)
    original = calibration.read_bytes()
    writer = recording.CameraRecording(tmp_path/'session', trial_name='Reach', codec=codec,
                                       calibration_path=calibration)
    rng = np.random.default_rng(52)
    frame = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
    writer.frame(0, frame, dict(host_received_monotonic=1.))
    writer.control_event('test_trigger', status='sent_unconfirmed')
    writer.close()
    calibration.write_bytes(b'changed later')
    manifest = json.loads((writer.directory/'manifest.json').read_text())
    assert (writer.directory/'calibration.npz').read_bytes() == original
    assert manifest['calibration']['sha256'] and manifest['video_codec'] == codec
    video = cv2.VideoCapture(str(writer.directory/manifest['video_files']['0']))
    try:
        ok, saved = video.read()
        assert ok
        assert saved.shape == frame.shape
        if codec == 'FFV1':
            np.testing.assert_array_equal(saved, frame)
    finally:
        video.release()
    assert rows(writer.directory/'control_events.jsonl')[0]['status'] == 'sent_unconfirmed'


def test_capture_reads_every_sdk_frame_when_recording_and_preserves_metadata(monkeypatch):
    captured = []
    class Frame:
        def __init__(self, frame_id): self.frame_id = frame_id
        def timing(self): return {"sdk_frame_id": self.frame_id}
        def rasterize(self, width, height):
            return np.full((height, width, 4), self.frame_id, np.uint8)
        def release(self): pass
    class Camera:
        def width(self): return 64
        def height(self): return 48
        def set_video_type(self, *args): pass
        def set_exposure(self, *args): pass
        def set_aec(self, *args): pass
        def set_agc(self, *args): pass
        def set_text_overlay(self, *args): pass
        def start(self): pass
        def get_exposure(self): return 10
        def get_latest_frame(self): raise AssertionError("Recording discarded buffered frames")
        def get_frame(self):
            index = len(captured)
            if index == 3:
                worker.running = False
                return None
            return Frame(index)
    manager = SimpleNamespace(inference_enabled=False, recorder=SimpleNamespace(
        frame=lambda index, image, metadata: captured.append((index, image, metadata))))
    worker = CameraWorker(Camera(), 0, manager=manager)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    worker._capture()
    assert [metadata["sdk_frame_id"] for _, _, metadata in captured] == [0, 1, 2]
    assert all(not metadata["inference_enabled"] for _, _, metadata in captured)
    assert all(not image.flags.writeable for _, image, _ in captured)


def test_inflight_inference_is_discarded_when_disabled_and_can_resume():
    from unity_hand_tracking.optitrack_cam_py._processing_thread import ProcessingThread
    entered, release = threading.Event(), threading.Event()
    class Tracker:
        calls = 0
        def process_frame(self):
            self.calls += 1
            entered.set()
            assert release.wait(3)
            return self.calls
    tracker = Tracker()
    worker = ProcessingThread(tracker)
    worker.start()
    try:
        assert entered.wait(2)
        worker.set_inference_enabled(False)
        release.set()
        deadline = time.monotonic() + 2
        while not worker.get_processing_fps() and time.monotonic() < deadline:
            time.sleep(.005)
        assert worker.get_latest() is None
        assert tracker.calls == 1 and worker.is_alive()
        worker.set_inference_enabled(True)
        deadline = time.monotonic() + 2
        while tracker.calls < 2 and time.monotonic() < deadline:
            time.sleep(.005)
        assert tracker.calls >= 2
    finally:
        release.set()
        worker.stop()


def test_inference_error_keeps_worker_available_for_retry():
    from unity_hand_tracking.optitrack_cam_py._processing_thread import ProcessingThread
    class Tracker:
        fail = True
        def process_frame(self):
            if self.fail:
                raise RuntimeError('Model unavailable')
            return 'recovered'
    tracker = Tracker()
    worker = ProcessingThread(tracker)
    worker.start()
    try:
        deadline = time.monotonic() + 2
        while worker.error is None and time.monotonic() < deadline:
            time.sleep(.005)
        assert worker.error == 'Model unavailable' and worker.is_alive()
        tracker.fail = False
        worker.set_inference_enabled(True)
        deadline = time.monotonic() + 2
        packet = None
        while packet is None and time.monotonic() < deadline:
            packet = worker.get_latest()
            time.sleep(.005)
        assert packet == 'recovered' and worker.error is None
    finally:
        worker.stop()


@pytest.mark.parametrize('extended', [False, True])
def test_sync_samples_never_claim_optihub_electrical_edges(monkeypatch, extended):
    from unity_hand_tracking.optitrack_cam_py import multi_mjpeg
    manager = multi_mjpeg.CameraManager.__new__(multi_mjpeg.CameraManager)
    manager._sync_stop = threading.Event()
    manager._sync_lock = threading.RLock()
    manager._expected_sync_hz = None
    manager.workers = []
    saved = []
    def save(status):
        saved.append(status)
        manager._sync_stop.set()
    manager.recorder = SimpleNamespace(sync=save)
    sdk = SimpleNamespace()
    if extended:
        sdk.get_sync_status = lambda: dict(available=True, input_source=2, input_divider=1,
                                           device_name='OptiHub 2')
    monkeypatch.setattr(multi_mjpeg, 'optitrack_cam', sdk)
    manager._monitor_sync()
    status = saved[0]
    assert status['input_edges'] is None and status['input_level'] is None
    assert not status['input_edge_recording_supported']
    assert status['available'] is extended and status['host_sampled_monotonic'] > 0
    if not extended:
        assert 'Rebuild' in status['reason']
