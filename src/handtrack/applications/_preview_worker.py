"""Single-owner native capture/model worker with one bounded preview slot."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from handtrack.tracker import MediaPipeModeTracker, TrackingMode


@dataclass(frozen=True)
class PreviewFrame:
    rgb: np.ndarray
    mode: TrackingMode
    detections: int
    fps: float


class PreviewWorker(threading.Thread):
    """All capture/model operations, including switches and cleanup, run here.

    GUI requests never wait for native calls. New previews replace the previous
    unread preview; no per-frame Qt signals or unbounded work queue are used.
    """

    def __init__(self, source=0, mode="hands"):
        super().__init__(daemon=True, name="MAVIS-preview")
        self.source = source
        self._lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._mode = TrackingMode.parse(mode)
        self._requested_mode = self._mode
        self._revision = 0
        self._latest = None
        self._state = "starting"
        self._error = None
        self._published_at = None

    def request_mode(self, mode):
        mode = TrackingMode.parse(mode)
        with self._lock:
            self._requested_mode = mode
            self._revision += 1
            self._latest = None
            self._error = None

    def request_stop(self):
        self._stop_requested.set()
        with self._lock:
            self._latest = None
            self._state = "stopping"

    def poll(self):
        with self._lock:
            latest, self._latest = self._latest, None
            return latest, self._mode, self._state, self._error, self._published_at

    def run(self):
        capture = tracker = None
        try:
            capture = cv2.VideoCapture(self.source)
            if not capture.isOpened():
                raise RuntimeError(f"Unable to open camera source {self.source!r}")
            if not isinstance(self.source, str):
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            tracker = MediaPipeModeTracker(self._requested_mode)
            rate = float(capture.get(cv2.CAP_PROP_FPS))
            period = 1 / rate if isinstance(self.source, str) and np.isfinite(rate) and rate > 0 else 0
            with self._lock:
                self._mode = tracker.mode
                self._state = "running"
                self._published_at = time.perf_counter()
            frames, started = 0, time.perf_counter()
            while not self._stop_requested.is_set():
                cycle_started = time.perf_counter()
                with self._lock:
                    mode, revision = self._requested_mode, self._revision
                if mode is not tracker.mode:
                    try:
                        tracker.set_mode(mode)
                        frames, started = 0, time.perf_counter()
                    except Exception as exc:  # noqa: BLE001 - Native worker boundary must report failures.
                        with self._lock:
                            if revision == self._revision:
                                self._error = f"Mode unavailable: {exc}"
                                self._requested_mode = tracker.mode
                        mode = tracker.mode
                    with self._lock:
                        self._mode = tracker.mode
                if self._stop_requested.is_set():
                    break
                ok, frame = capture.read()
                if not ok or frame is None:
                    with self._lock:
                        self._latest = None
                        self._state = "ended"
                    break
                if self._stop_requested.is_set():
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = tracker.process(rgb)
                tracker.draw(frame, result)
                display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                display.flags.writeable = False
                frames += 1
                preview = PreviewFrame(display, tracker.mode,
                    len(result.hands) + len(result.faces) + len(result.poses),
                    frames / max(time.perf_counter() - started, 1e-6))
                with self._lock:
                    if revision == self._revision and not self._stop_requested.is_set():
                        self._latest = preview
                        self._published_at = time.perf_counter()
                self._stop_requested.wait(max(0, period - (time.perf_counter() - cycle_started)))
        except Exception as exc:  # noqa: BLE001 - Native worker boundary must report failures.
            with self._lock:
                self._latest = None
                self._state = "error"
                self._error = f"Tracking stopped: {exc}"
        finally:
            for resource in (tracker, capture):
                if resource is not None:
                    try:
                        resource.close() if resource is tracker else resource.release()
                    except Exception as exc:  # noqa: BLE001 - Native worker boundary must report failures.
                        with self._lock:
                            self._state = "error"
                            self._error = f"Cleanup failed: {exc}"
            with self._lock:
                if self._state in ("starting", "running", "stopping"):
                    self._state = "stopped"
