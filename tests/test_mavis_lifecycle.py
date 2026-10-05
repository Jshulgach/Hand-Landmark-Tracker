import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from mavis_track._capture import FrameBuffer, snapshots
from handtrack.tracker._mediapipe_modes import MediaPipeModeTracker, TrackingMode


def test_failed_mode_switch_preserves_working_solution(monkeypatch):
    closed = []
    def create(self):
        if self.mode == TrackingMode.FACE:
            raise RuntimeError('model unavailable')
        self._solution = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(MediaPipeModeTracker, '_create_solution', create)
    tracker = MediaPipeModeTracker('hands')
    with pytest.raises(RuntimeError, match='unavailable'):
        tracker.set_mode('face')
    assert tracker.mode == TrackingMode.HANDS and closed == []
    tracker.close()
    assert closed == [True]


def buffer():
    worker = FrameBuffer()
    worker.lock = threading.Lock()
    worker.ready = threading.Event()
    worker.latest_frame = None
    worker.publish_frame(np.zeros((8, 10, 3), np.uint8))
    return worker


def test_stale_and_skewed_camera_frames_are_excluded():
    first, second = buffer(), buffer()
    assert not first.get_frame(copy=False).flags.writeable
    first.frame_timestamp = time.perf_counter() - .1
    assert snapshots([first, second])[0] is None
    second.frame_timestamp = time.perf_counter() - 1
    assert second.get_snapshot()[0] is None


def test_legacy_detector_start_failure_releases_capture(monkeypatch):
    from handtrack.tracker import _hand_tracker as module
    from test_hand_tracker import FakeCapture
    cap = FakeCapture([])
    monkeypatch.setattr(module.cv2, 'VideoCapture', lambda *_: cap)
    def fail(**_):
        raise RuntimeError('model start failed')
    monkeypatch.setattr(module.mp.solutions.hands, 'Hands', fail)
    with pytest.raises(RuntimeError, match='model start'):
        module.HandTracker()
    assert cap._released


def test_gui_failed_camera_can_retry_and_releases_on_tracking_error(monkeypatch):
    from handtrack.applications import mediapipe_gui as module
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    closed = []
    class Tracker:
        mode = TrackingMode.HANDS
        def __init__(self, *_):
            pass
        def process(self, *_):
            raise RuntimeError('processing failed')
        def close(self):
            closed.append(True)
    class Capture:
        def __init__(self, opened):
            self.opened, self.released = opened, False
        def isOpened(self):
            return self.opened
        def read(self):
            return True, np.zeros((8, 8, 3), np.uint8)
        def release(self):
            self.released = True
    first, second = Capture(False), Capture(True)
    captures = iter((first, second))
    monkeypatch.setattr(module, 'MediaPipeModeTracker', Tracker)
    monkeypatch.setattr(module.cv2, 'VideoCapture', lambda *_: next(captures))
    window = module.MediaPipeTrackerGUI()
    assert first.released and window.capture is None
    window._start()
    assert window.timer.isActive()
    window._update_frame()
    assert second.released and not window.timer.isActive()
    assert 'processing failed' in window.preview.text()
    window.close()
    app.processEvents()
    assert closed == [True]
