"""
multi_mjpeg.py — Reusable OptiTrack multi-camera module.

Provides:
  - CameraWorker: threaded per-camera frame grabber
  - CameraManager: initializes SDK, manages workers, builds grid frames
  - main(): standalone demo viewer (run this file directly)

Usage from another script:
    from multi_mjpeg import CameraManager

    mgr = CameraManager()
    mgr.start_all()

    # Grab individual frames
    frame = mgr.get_frame(cam_index=0)

    # Or build a stitched grid
    grid = mgr.get_grid()

    mgr.stop_all()
"""

import threading
import time
from dataclasses import asdict

from mavis_track._capture import FrameBuffer, snapshots
from mavis_track._optitrack_runtime import capture_runtime, shutdown_capture_sdks

import cv2
import numpy as np

try:
    from .config import (
        CAMERA_EXPOSURE,
        DISPLAY_SCALE,
        GRID_COLS,
        MJPEG_MODE,
        SDK_IMPORT_ERROR,
        optitrack_cam,
    )
except ImportError:
    from config import (
        CAMERA_EXPOSURE,
        DISPLAY_SCALE,
        GRID_COLS,
        MJPEG_MODE,
        SDK_IMPORT_ERROR,
        optitrack_cam,
    )


# ---------------------------------------------------------------------------
# Per-camera threaded worker
# ---------------------------------------------------------------------------
class CameraWorker(FrameBuffer, threading.Thread):
    """
    Dedicated thread that continuously drains a single camera's buffer
    so the most recent frame is always available via get_frame().
    """

    def __init__(self, camera, index, exposure=None, mjpeg_mode=MJPEG_MODE, manager=None):
        super().__init__(daemon=True)
        self.camera = camera
        self.manager = manager
        self.index = index
        self.name = f"Camera #{index}"
        self.exposure = exposure  # None = let camera auto-manage
        self.mjpeg_mode = mjpeg_mode

        self.running = True
        self.latest_frame = None
        self.lock = threading.Lock()
        self.width = camera.width()
        self.height = camera.height()
        self.frame_timestamp = None
        self.capture_settings = {"camera_index": index, "width": self.width, "height": self.height}
        self.status = "starting"
        self.error = ""
        self.ready = threading.Event()
        self.capture_started = threading.Event()

    # -- public helpers (safe to call from any thread) ----------------------

    def get_resolution(self):
        return self.width, self.height

    def set_exposure(self, value):
        """Update exposure at runtime (thread-safe via SDK)."""
        self.exposure = value
        self.camera.set_exposure(value)

    def stop(self):
        self.running = False
        # The capture thread owns SDK handles and releases them in its finally block.

    # -- thread body --------------------------------------------------------

    def run(self):
        try:
            self._capture()
        except Exception as exc:
            self.error = str(exc)
            self.status = "failed"
        finally:
            self.running = False
            self.ready.set()
            try:
                self.camera.stop()
            finally:
                self.camera.release()

    def _capture(self):
        cam = self.camera
        cam.set_video_type(self.mjpeg_mode)
        cam.set_exposure(self.exposure)
        cam.set_aec(False)
        cam.set_agc(False)
        cam.set_text_overlay(False)
        cam.start()

        time.sleep(0.2)  # let settings take effect

        actual_exp = cam.get_exposure()
        self.capture_settings.update(exposure=actual_exp, video_mode=self.mjpeg_mode)
        if hasattr(cam, "capture_info"):
            self.capture_settings.update(cam.capture_info())
        from mavis_track._illumination import read_illumination
        self.capture_settings['illumination'] = asdict(read_illumination(cam))
        self.status = "waiting for frames"
        self.capture_started.set()
        print(
            f"[{self.name}] Exposure={self.exposure}, "
            f"AEC=False, AGC=False | Actual={actual_exp}"
        )

        while self.running:
            recorder = self.manager.recorder if self.manager is not None else None
            # Preserve SDK frames while recording; preview can still replace its slot.
            frame_obj = cam.get_frame() if recorder is not None else cam.get_latest_frame()
            if frame_obj:
                try:
                    received = time.perf_counter()
                    metadata = {"host_received_monotonic": received,
                                "sdk_timing_available": hasattr(frame_obj, "timing"),
                                "exposure_setting": self.exposure,
                                "inference_enabled": getattr(self.manager, "inference_enabled", True)}
                    if hasattr(frame_obj, "timing"):
                        metadata.update(frame_obj.timing())
                    img_rgba = frame_obj.rasterize(self.width, self.height)
                    img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
                    self.publish_frame(img_bgr)
                    if recorder is not None:
                        recorder.frame(self.index, img_bgr, metadata)
                finally:
                    frame_obj.release()
            # Always yield CPU — prevents starving other processes.
            # ~2ms sleep still allows >200 FPS capture which far exceeds
            # the ~30 FPS processing rate.
            time.sleep(0.002)


# ---------------------------------------------------------------------------
# High-level manager
# ---------------------------------------------------------------------------
class CameraManager:
    """
    Convenience wrapper: discovers cameras, creates CameraWorkers, and
    provides helpers for grid display.
    """

    def __init__(self, discovery_wait: float = 2.0):
        """Initialise the OptiTrack SDK and discover cameras."""
        if optitrack_cam is None:
            raise RuntimeError(
                f"OptiTrack Python binding could not load ({SDK_IMPORT_ERROR}). "
                "From a source checkout run: python external/setup_optitrack.py --from-history "
                "(Windows x64 / Python 3.11), then restart MAVIS. "
                "For another compatible binding, set OPTITRACK_CAM_PY_PATH to its folder."
            )
        self.workers: list[CameraWorker] = []
        self._shutdown = False  # Capture stopped; the process SDK remains alive.
        self._stop_lock = threading.RLock()
        self.num_cameras = 0
        self.recorder = None
        self.inference_enabled = True
        self._sync_stop = threading.Event()
        self._sync_lock = threading.RLock()
        self._expected_sync_hz = None
        self._sync_thread = None
        self._sync_status = {"available": False, "reason": "Cameras not started"}
        self.camera_indices = []
        self._runtime = capture_runtime(optitrack_cam)
        with self._stop_lock:
            fresh = self._runtime.acquire(self)
            try:
                print("Initializing OptiTrack SDK..." if fresh else "Reusing OptiTrack SDK...")
                time.sleep(discovery_wait)
                self.num_cameras = optitrack_cam.camera_count()
                self.camera_indices = list(range(self.num_cameras))
                print(f"Detected {self.num_cameras} cameras.")
            except BaseException:
                self._shutdown = True
                self._runtime.release(self)
                raise

    # -- lifecycle ----------------------------------------------------------

    def start_all(
        self,
        exposure=CAMERA_EXPOSURE,
        mjpeg_mode=MJPEG_MODE,
        acquisition_rounds: int = 10,
        retry_delay: float = 0.5,
    ):
        """Create and start workers while holding the exclusive capture lease."""
        with self._stop_lock:
            if self._shutdown:
                raise RuntimeError("This camera session is stopped; create a new CameraManager")
            return self._start_all(exposure, mjpeg_mode, acquisition_rounds, retry_delay)

    def _start_all(self, exposure, mjpeg_mode, acquisition_rounds, retry_delay):
        if self.workers:
            return self.workers

        # Cameras can appear in camera_count() before all SDK handles are
        # ready. Retry every unresolved index in rounds so early indices are
        # revisited instead of giving each one a single, premature window.
        handles = {}
        pending = set(range(self.num_cameras))
        for attempt in range(max(1, acquisition_rounds)):
            for index in sorted(pending):
                camera = optitrack_cam.get_camera_by_index(index)
                if camera is not None:
                    handles[index] = camera
            pending.difference_update(handles)
            if not pending:
                break
            if attempt < acquisition_rounds - 1:
                time.sleep(retry_delay)

        if pending:
            for camera in handles.values():
                try:
                    camera.release()
                except Exception:
                    pass
            raise RuntimeError(
                "OptiTrack detected "
                f"{self.num_cameras} cameras but could not acquire handles for "
                f"indices {sorted(pending)} after {max(1, acquisition_rounds)} "
                "attempts. Close other camera applications, power-cycle any "
                "missing cameras, wait a few seconds, and try again."
            )

        unstarted = dict(handles)
        try:
            for index in range(self.num_cameras):
                worker = CameraWorker(
                    handles[index], index, exposure=exposure, mjpeg_mode=mjpeg_mode
                )
                worker.manager = self
                worker.start()
                self.workers.append(worker)
                del unstarted[index]
                print(f"Started worker for Camera {index}")
        except BaseException:
            for camera in unstarted.values():
                camera.release()
            raise

        self._sync_thread = threading.Thread(target=self._monitor_sync, name="OptiHub-status", daemon=True)
        self._sync_thread.start()
        return self.workers

    def _monitor_sync(self):
        while not self._sync_stop.is_set():
            self._sample_sync_status()
            self._sync_stop.wait(.2)

    def _sample_sync_status(self):
        with self._sync_lock:
            status = {"host_sampled_monotonic": time.perf_counter(),
                      "input_level": None, "input_edges": None,
                      "input_edge_recording_supported": False,
                      "configurable": hasattr(optitrack_cam, "configure_sync_preset"),
                      "exposure_output_configurable": hasattr(optitrack_cam, "configure_exposure_output"),
                      "expected_frequency_hz": self._expected_sync_hz}
            try:
                if hasattr(optitrack_cam, "get_sync_status"):
                    status.update(optitrack_cam.get_sync_status())
                else:
                    status.update(available=False, reason="Rebuild binding for OptiHub sync configuration")
            except Exception as exc:
                status.update(available=False, reason=str(exc))
            status["cameras"] = [dict(camera_index=worker.index,
                                      status=getattr(worker, "status", "unknown"),
                                      error=getattr(worker, "error", ""),
                                      last_frame_host_monotonic=getattr(worker, "frame_timestamp", None))
                                 for worker in self.workers]
            self._sync_status = status
            recorder = self.recorder
            if recorder is not None:
                recorder.sync(status)
            return dict(status)

    def get_sync_status(self):
        return dict(self._sync_status)

    def configure_sync_preset(self, preset):
        """Explicit mode change, serialized with SDK reads and session shutdown."""
        if preset not in ("vicon_external_50hz", "internal_50hz"):
            raise ValueError("Unknown OptiHub sync preset")
        with self._stop_lock:
            if self._shutdown or not self.workers:
                raise RuntimeError("Start cameras before configuring sync")
            if self.recorder is not None:
                raise RuntimeError("Stop recording before changing camera sync")
            if not hasattr(optitrack_cam, "configure_sync_preset"):
                raise RuntimeError("Rebuild the native binding to enable OptiHub sync controls")
            with self._sync_lock:
                try:
                    optitrack_cam.configure_sync_preset(preset)
                except Exception:
                    self._sample_sync_status()
                    raise
                self._expected_sync_hz = 50.
                return self._sample_sync_status()

    def start_recording(self, directory, *, nominal_fps=100., trial_name="trial", session_metadata=None,
                        calibration_path=None, codec="MJPG"):
        from mavis_track._camera_recording import CameraRecording
        with self._stop_lock:
            if self._shutdown or not self.workers:
                raise RuntimeError("Start cameras before recording")
            if self.recorder is not None:
                raise RuntimeError("A recording is already active")
            metadata = dict(session_metadata or {})
            metadata['binding_capabilities'] = {
                name: getattr(optitrack_cam, name, None) for name in (
                    'BINDING_API_VERSION', 'CAMERA_RECORDING_API_VERSION', 'OPTIHUB_SYNC_CONTROL_API_VERSION')}
            metadata['sync_at_recording_start'] = self.get_sync_status()
            self.recorder = CameraRecording(directory, nominal_fps=nominal_fps,
                                            camera_ids=self.camera_indices,
                                            camera_settings=[dict(worker.capture_settings) for worker in self.workers],
                                            trial_name=trial_name, session_metadata=metadata,
                                            calibration_path=calibration_path, codec=codec)
            self.recorder.sync(self.get_sync_status())
            return self.recorder

    def configure_exposure_output(self):
        with self._stop_lock:
            if self._shutdown or not self.workers:
                raise RuntimeError("Start cameras before configuring the exposure output")
            if self.recorder is not None:
                raise RuntimeError("Stop recording before changing the exposure output")
            if not hasattr(optitrack_cam, 'configure_exposure_output'):
                raise RuntimeError("Rebuild the binding to configure exposure output")
            with self._sync_lock:
                try:
                    optitrack_cam.configure_exposure_output()
                finally:
                    self._sample_sync_status()
            return self.get_sync_status()

    def stop_recording(self):
        with self._stop_lock:
            recorder = self.recorder
            if recorder is not None:
                try:
                    recorder.close()
                finally:
                    if not recorder._thread.is_alive():
                        self.recorder = None

    def wait_ready(self, timeout=3):
        status = self._sample_sync_status()
        external = status.get("available") and status.get("input_source") in (2, 3, 4, 5, 6)
        for worker in self.workers:
            if external and hasattr(worker, "capture_started"):
                if worker.capture_started.wait(timeout) and worker.is_alive() and not worker.error:
                    continue  # A missing external pulse train is not a camera startup failure.
            if not worker.ready.wait(timeout) or worker.frame_timestamp is None:
                raise RuntimeError(f"{worker.name} did not supply a frame: {worker.error}")

    def stop_all(self):
        """Stop/release capture handles; retain the SDK for the next session."""
        with self._stop_lock:
            if self._shutdown:
                return
            recording_error = None
            try:
                self.stop_recording()
            except RuntimeError as exc:
                if self.recorder is not None:
                    raise  # Writer still owns queued frames; preserve session for retry.
                recording_error = exc
            self._sync_stop.set()
            if self._sync_thread is not None:
                self._sync_thread.join(timeout=3)
                if self._sync_thread.is_alive():
                    raise RuntimeError("Sync status worker did not stop")
            print("Stopping camera workers...")
            for worker in self.workers:
                worker.stop()
            for worker in self.workers:
                worker.join(timeout=3)
            alive = [worker.name for worker in self.workers if worker.is_alive()]
            if alive:
                # Keep ownership: another model must not reopen the cameras
                # or shut down the SDK underneath capture still in flight.
                raise RuntimeError(f"Camera workers did not stop: {alive}")
            self._shutdown = True
            self.workers.clear()
            self._runtime.release(self)
            print("Camera capture cleanup complete.")
            if recording_error is not None:
                raise recording_error

    # -- frame access -------------------------------------------------------

    def set_exposure(self, cam_index: int, value: int):
        """Set exposure for a specific camera at runtime."""
        with self._stop_lock:
            if not self._shutdown and 0 <= cam_index < len(self.workers):
                self.workers[cam_index].set_exposure(value)
                self.workers[cam_index].capture_settings["exposure"] = value
                if self.recorder is not None:
                    self.recorder.event('exposure_changed', camera_index=cam_index, requested_value=value)

    def _illumination_camera(self, cam_index):
        if self._shutdown:
            raise RuntimeError("Camera capture session is stopped")
        if type(cam_index) is not int or not 0 <= cam_index < len(self.workers):
            raise ValueError("Camera index is outside the active capture session")
        return self.workers[cam_index].camera

    def get_illumination(self, cam_index):
        from mavis_track._illumination import read_illumination
        with self._stop_lock:
            return read_illumination(self._illumination_camera(cam_index))

    def set_illumination(self, cam_index, *, intensity=None, high_power=None, maximize=False):
        from mavis_track._illumination import set_illumination
        with self._stop_lock:
            state = set_illumination(self._illumination_camera(cam_index), intensity=intensity,
                                     high_power=high_power, maximize=maximize)
            self.workers[cam_index].capture_settings['illumination'] = asdict(state)
            if self.recorder is not None:
                self.recorder.event('illumination_changed', camera_index=cam_index, state=asdict(state))
            return state

    def get_frame(self, cam_index: int):
        """Return the latest BGR frame from a specific camera."""
        return self.workers[cam_index].get_frame()

    def get_all_frames(self, copy=True):
        """Return a list of BGR frames, one per camera (in index order)."""
        frames = []
        for worker in self.workers:
            frame, _ = worker.get_snapshot()
            if frame is None:
                frame = np.zeros((worker.height, worker.width, 3), np.uint8)
            frames.append(frame.copy() if copy else frame)
        return frames

    def get_fresh_frames(self):
        return snapshots(self.workers)

    def get_grid(
        self,
        grid_cols: int = GRID_COLS,
        scale: float = DISPLAY_SCALE,
        overlay_labels: bool = True,
    ):
        """
        Stitch all camera frames into a single grid image.

        Parameters
        ----------
        grid_cols : int
            Number of columns in the grid.
        scale : float
            Resize factor applied to the final grid (1.0 = no resize).
        overlay_labels : bool
            If True, burn "Camera #N" text into each cell.

        Returns
        -------
        numpy.ndarray  (BGR)
        """
        frames = self.get_all_frames()
        if not frames:
            return np.zeros((480, 640, 3), np.uint8)

        width = max(frame.shape[1] for frame in frames)
        height = max(frame.shape[0] for frame in frames)
        frames = [cv2.resize(frame, (width, height)) for frame in frames]
        if overlay_labels:
            for i, f in enumerate(frames):
                cv2.putText(
                    f,
                    f"Camera #{i}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

        # Build row-major grid, padding incomplete rows with black
        rows = []
        for start in range(0, len(frames), grid_cols):
            row = frames[start : start + grid_cols]
            while len(row) < grid_cols:
                row.append(np.zeros_like(frames[0]))
            rows.append(np.hstack(row))

        grid = np.vstack(rows)

        if scale != 1.0:
            grid = cv2.resize(grid, None, fx=scale, fy=scale)

        return grid

    # -- info ---------------------------------------------------------------

    def get_resolution(self, cam_index: int = 0):
        """Return (width, height) for a given camera."""
        if not self.workers:
            raise RuntimeError(
                "No OptiTrack camera workers are running. Call start_all() first and ensure camera handles were acquired."
            )
        return self.workers[cam_index].get_resolution()


# ---------------------------------------------------------------------------
# Standalone demo viewer (equivalent to the original multi_mjpeg.py)
# ---------------------------------------------------------------------------
def main():
    mgr = CameraManager()
    if mgr.num_cameras == 0:
        print("No cameras found. Exiting.")
        mgr.stop_all()
        shutdown_capture_sdks()
        return

    mgr.start_all()

    window_name = "OptiTrack Multi-Camera Live Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            grid = mgr.get_grid()
            cv2.imshow(window_name, grid)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
    finally:
        mgr.stop_all()
        shutdown_capture_sdks()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
