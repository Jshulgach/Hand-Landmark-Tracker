"""Concurrency contracts: responsive Qt, native ownership and bounded previews."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from handtrack.applications import _preview_worker as workers
from handtrack.applications.mediapipe_gui import MediaPipeTrackerGUI
from handtrack.tracker import TrackingMode


def wait_until(predicate, app=None, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if app is not None:
            app.processEvents()
        if predicate():
            return
        time.sleep(.002)
    raise AssertionError('Timed out waiting for native worker')


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def factories(monkeypatch, *, init_gate=None, read_gate=None, process_gate=None,
              block_read_at=1, fail_mode=False, fail_process=False):
    state = dict(operations=[], reads=0, processed=0, closed=False, released=False)
    entered = threading.Event()
    def operation(name):
        state['operations'].append((name, threading.get_ident()))
    class Capture:
        def __init__(self, source):
            operation('capture_open')
        def isOpened(self):
            return True
        def set(self, *_):
            pass
        def get(self, *_):
            return 30
        def read(self):
            operation('capture_read')
            state['reads'] += 1
            if read_gate is not None and state['reads'] == block_read_at:
                entered.set()
                assert read_gate.wait(3)
            return True, np.full((16, 16, 3), state['reads'] % 255, np.uint8)
        def release(self):
            operation('capture_release')
            state['released'] = True
    class Tracker:
        def __init__(self, mode):
            operation('model_open')
            self.mode = TrackingMode.parse(mode)
            if init_gate is not None:
                entered.set()
                assert init_gate.wait(3)
        def set_mode(self, mode):
            operation('model_switch')
            if fail_mode:
                raise RuntimeError('model unavailable')
            self.mode = TrackingMode.parse(mode)
        def process(self, image):
            operation('model_process')
            state['processed'] += 1
            if process_gate is not None and state['processed'] == 1:
                entered.set()
                assert process_gate.wait(3)
            if fail_process:
                raise RuntimeError('inference failed')
            return SimpleNamespace(hands=[], faces=[], poses=[])
        def draw(self, frame, result):
            return frame
        def close(self):
            operation('model_close')
            state['closed'] = True
    monkeypatch.setattr(workers.cv2, 'VideoCapture', Capture)
    monkeypatch.setattr(workers, 'MediaPipeModeTracker', Tracker)
    return state, entered


def test_slow_start_and_stop_do_not_block_qt_event_loop(monkeypatch, app):
    gate = threading.Event()
    state, entered = factories(monkeypatch, init_gate=gate)
    window = MediaPipeTrackerGUI()
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(5)
    try:
        wait_until(entered.is_set, app)
        window.mode_combo.setCurrentText('Face')
        window._stop()
        wait_until(lambda: len(ticks) >= 4, app)
        assert window.worker is not None  # The model constructor is still blocked.
        gate.set()
        wait_until(lambda: window.worker is None, app)
        assert state['closed'] and state['released']
        assert window.toggle_button.isEnabled()
        assert all(owner != threading.get_ident() for _, owner in state['operations'])
        assert len({owner for _, owner in state['operations']}) == 1
    finally:
        gate.set()
        timer.stop()
        window.close()
        wait_until(lambda: window.worker is None, app)


def test_window_close_waits_asynchronously_for_blocked_native_read(monkeypatch, app):
    gate = threading.Event()
    state, entered = factories(monkeypatch, read_gate=gate)
    window = MediaPipeTrackerGUI()
    window.show()
    try:
        wait_until(entered.is_set, app)
        window.close()
        assert window._closing and window.worker is not None
        ticks = []
        QTimer.singleShot(0, lambda: ticks.append(1))
        wait_until(lambda: bool(ticks), app)
        assert not state['released']  # UI must never release a capture being read.
        gate.set()
        wait_until(lambda: window.worker is None, app)
        assert state['released'] and state['closed'] and not window.isVisible()
    finally:
        gate.set()
        window.close()
        wait_until(lambda: window.worker is None, app)


def test_preview_slot_keeps_latest_frame_without_backlog(monkeypatch):
    gate = threading.Event()
    state, entered = factories(monkeypatch, read_gate=gate, block_read_at=51)
    worker = workers.PreviewWorker()
    worker.start()
    try:
        assert entered.wait(3)
        packet, *_ = worker.poll()
        assert state['processed'] == 50
        assert packet.rgb[0, 0, 0] == 50
        assert not packet.rgb.flags.writeable
        assert worker.poll()[0] is None
        worker.request_stop()
        gate.set()
        worker.join(3)
        assert not worker.is_alive() and state['closed'] and state['released']
    finally:
        worker.request_stop()
        gate.set()
        worker.join(3)


def test_mode_switch_discards_inflight_previous_mode_frame(monkeypatch):
    process_gate = threading.Event()
    state, entered = factories(monkeypatch, process_gate=process_gate)
    worker = workers.PreviewWorker(mode='hands')
    worker.start()
    try:
        assert entered.wait(3)
        worker.request_mode('face')
        assert worker.poll()[0] is None
        process_gate.set()
        packets = []
        def new_packet():
            packet = worker.poll()[0]
            if packet is not None:
                packets.append(packet)
            return bool(packets)
        wait_until(new_packet)
        assert packets[0].mode is TrackingMode.FACE
        assert any(name == 'model_switch' for name, _ in state['operations'])
    finally:
        worker.request_stop()
        process_gate.set()
        worker.join(3)
        assert not worker.is_alive()


def test_failed_switch_keeps_previous_mode_and_reports_error(monkeypatch):
    state, _ = factories(monkeypatch, fail_mode=True)
    worker = workers.PreviewWorker()
    worker.start()
    try:
        wait_until(lambda: state['processed'] > 0)
        worker.request_mode('face')
        snapshots = []
        def error():
            snapshot = worker.poll()
            snapshots.append(snapshot)
            return snapshot[3] is not None
        wait_until(error)
        assert snapshots[-1][1] is TrackingMode.HANDS
        assert snapshots[-1][2] == 'running'
        assert 'Mode unavailable' in snapshots[-1][3]
    finally:
        worker.request_stop()
        worker.join(3)
        assert not worker.is_alive()


def test_inference_error_clears_preview_and_releases_resources(monkeypatch):
    state, _ = factories(monkeypatch, fail_process=True)
    worker = workers.PreviewWorker()
    worker.start()
    worker.join(3)
    packet, _, status, error, _ = worker.poll()
    assert packet is None and status == 'error' and 'inference failed' in error
    assert state['closed'] and state['released'] and not worker.is_alive()
