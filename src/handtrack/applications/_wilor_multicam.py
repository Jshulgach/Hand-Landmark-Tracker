"""Asynchronous OptiTrack/WiLoR preview with one model and bounded results."""
from __future__ import annotations

import sys
import threading
import time
import traceback
from dataclasses import dataclass

from mavis_track.wilor import WiLoRMultiViewTracker


def camera_manager():
    from unity_hand_tracking.optitrack_cam_py.multi_mjpeg import CameraManager
    return CameraManager()


@dataclass(frozen=True)
class ViewPacket:
    frames: tuple
    results: tuple
    received_at: tuple
    finished_at: float
    inference_seconds: float


class WiLoRCameraSession(threading.Thread):
    """Own startup, inference and shutdown; camera workers own native capture.

    The GUI reads fresh camera snapshots independently while one shared WiLoR
    model loads/processes. Inference frames always stay paired with their output.
    No triangulation, recording or network streaming runs in this session.
    """

    calibration_available = False
    calibration_error = "WiLoR per-camera preview; calibrated reconstruction is not enabled."

    def __init__(self, *, root=None, device="auto", fast=False, batch_size=2,
                 confidence=.3, exposure=10):
        super().__init__(daemon=True, name="WiLoR-multicam")
        self.options = {"root": root, "device": device, "fast": fast,
                        "batch_size": batch_size, "confidence": confidence,
                        "source_id": "optitrack"}
        self.confidence = confidence
        self.exposure = exposure
        self._stop_requested = threading.Event()
        self.ready = threading.Event()
        self._lock = threading.RLock()
        self._manager = None
        self._packet = None
        self.faces = None
        self.num_cameras = 0
        self.camera_indices = ()
        self.image_sizes = ()
        self.state = "starting cameras"
        self.error = None
        self.device = None
        self.completed_batches = 0
        self._stage_started = time.perf_counter()

    @property
    def stage_seconds(self):
        with self._lock:
            return time.perf_counter() - self._stage_started

    def _set_state(self, state, *, announce=False):
        with self._lock:
            if self._stop_requested.is_set():
                return
            if self.state != state:
                self._stage_started = time.perf_counter()
            self.state = state
        if announce:
            print(f"[WiLoR] {state}", flush=True)

    def _progress(self, stage):
        self._set_state(stage, announce=True)

    def _failure(self, stage, exc):
        with self._lock:
            self.error = str(exc)
            self.state = stage
        print(f"[WiLoR] {stage}: {exc}\n[WiLoR] Python: {sys.executable}",
              file=sys.stderr, flush=True)
        traceback.print_exception(type(exc), exc, exc.__traceback__)

    @property
    def cam_mgr(self):
        return self if self.ready.is_set() else None

    def snapshot(self):
        with self._lock:
            return self._packet, self.state, self.error

    def raw_snapshots(self):
        with self._lock:
            if self._manager is None or self._stop_requested.is_set():
                return tuple((None, None) for _ in range(self.num_cameras))
            # Each buffer applies its own age limit. Independent views do not
            # imply synchronized exposure or require cross-view receipt skew.
            return tuple(worker.get_snapshot() for worker in list(self._manager.workers))

    def _control_manager(self):
        if self._manager is None or self._stop_requested.is_set():
            raise RuntimeError("Camera session is stopping or unavailable")
        return self._manager

    def get_resolution(self, index):
        return self.image_sizes[index]

    def set_exposure(self, index, value):
        with self._lock:
            self._control_manager().set_exposure(index, value)

    def get_illumination(self, index):
        with self._lock:
            return self._control_manager().get_illumination(index)

    def set_illumination(self, index, **kwargs):
        with self._lock:
            return self._control_manager().set_illumination(index, **kwargs)

    def set_enabled_cameras(self, *_):
        pass  # Legacy reconstruction controls are disabled for per-camera preview.

    def update_mp_params(self, confidence, _tracking_confidence):
        with self._lock:
            self.confidence = float(confidence)

    def request_stop(self):
        self._stop_requested.set()
        with self._lock:
            self.state = "stopping"
            self._packet = None

    def run(self):
        manager = model = None
        try:
            manager = camera_manager()
            if self._stop_requested.is_set():
                return
            if manager.num_cameras < 1:
                raise RuntimeError("No OptiTrack cameras detected")
            manager.start_all(exposure=self.exposure)
            manager.wait_ready()
            with self._lock:
                self.num_cameras = manager.num_cameras
                self.camera_indices = tuple(getattr(manager, "camera_indices", range(self.num_cameras)))
                self.image_sizes = tuple(manager.get_resolution(i) for i in range(self.num_cameras))
                self._manager = manager
                self._set_state("Loading WiLoR", announce=True)
                self.ready.set()
            if self._stop_requested.is_set():
                return
            try:
                print(f"[WiLoR] Python: {sys.executable}; requested device: "
                      f"{self.options['device']}; fast: {self.options['fast']}", flush=True)
                model = WiLoRMultiViewTracker(**self.options, progress=self._progress)
                self.faces = model.faces
                self.device = model.device
                self._progress(f"Model ready on {self.device.upper()}; waiting for first inference")
            except Exception as exc:  # noqa: BLE001 - Native inference boundary.
                self._failure("WiLoR unavailable; live cameras remain available", exc)
                self._stop_requested.wait()
                return
            last_received = None
            while not self._stop_requested.is_set():
                snapshots = self.raw_snapshots()
                frames = tuple(frame for frame, _ in snapshots)
                if not any(frame is not None for frame in frames):
                    self._set_state("Waiting for fresh camera frames")
                    self._stop_requested.wait(.02)
                    continue
                received = tuple(stamp if frame is not None else None
                                 for frame, stamp in snapshots)
                if received == last_received:
                    self._stop_requested.wait(.002)
                    continue
                last_received = received
                with self._lock:
                    confidence = self.confidence
                self._set_state(
                    "First inference batch" if self.completed_batches == 0 else "Processing WiLoR",
                    announce=self.completed_batches == 0)
                started = time.perf_counter()
                try:
                    model.confidence = confidence
                    results = model.process_views(frames, timestamp=started)
                except Exception as exc:  # noqa: BLE001 - Native inference boundary.
                    with self._lock:
                        self._packet = None
                    self._failure("WiLoR inference failed; live cameras remain available", exc)
                    self._stop_requested.wait()
                    break
                finished = time.perf_counter()
                packet = ViewPacket(
                    frames, results, tuple(stamp for _, stamp in snapshots), finished,
                    finished - started)
                with self._lock:
                    if not self._stop_requested.is_set():
                        self._packet = packet
                        self.completed_batches += 1
                        self._set_state("Running WiLoR")
                if self.completed_batches == 1:
                    counts = [len(result.hands) if result is not None else 0 for result in results]
                    print(f"[WiLoR] First batch completed on {self.device.upper()} in "
                          f"{packet.inference_seconds:.2f}s; hands per camera: {counts}", flush=True)
                self._stop_requested.wait(.001)
        except Exception as exc:  # noqa: BLE001 - Native camera boundary.
            self._failure("Camera startup failed", exc)
        finally:
            with self._lock:
                self.ready.clear()
                self._manager = None
            try:
                if model is not None:
                    model.close()
            finally:
                try:
                    if manager is not None:
                        manager.stop_all()
                except Exception as exc:  # noqa: BLE001 - Native cleanup boundary.
                    with self._lock:
                        self.error = f"Camera cleanup failed: {exc}"
                with self._lock:
                    self._packet = None
                    if self._stop_requested.is_set() and not self.error:
                        self.state = "stopped"
