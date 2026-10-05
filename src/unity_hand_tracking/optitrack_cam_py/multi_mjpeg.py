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

from mavis_track._capture import FrameBuffer, snapshots

import cv2
import numpy as np

try:
    from .config import (
        CAMERA_EXPOSURE,
        DISPLAY_SCALE,
        GRID_COLS,
        MJPEG_MODE,
        optitrack_cam,
    )
except ImportError:
    from config import (
        CAMERA_EXPOSURE,
        DISPLAY_SCALE,
        GRID_COLS,
        MJPEG_MODE,
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

    def __init__(self, camera, index, exposure=None, mjpeg_mode=MJPEG_MODE):
        super().__init__(daemon=True)
        self.camera = camera
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
        self.status = "starting"
        self.error = ""
        self.ready = threading.Event()

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
        cam.set_text_overlay(True)
        cam.start()

        time.sleep(0.2)  # let settings take effect

        actual_exp = cam.get_exposure()
        print(
            f"[{self.name}] Exposure={self.exposure}, "
            f"AEC=False, AGC=False | Actual={actual_exp}"
        )

        while self.running:
            frame_obj = cam.get_latest_frame()
            if frame_obj:
                try:
                    img_rgba = frame_obj.rasterize(self.width, self.height)
                    img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
                    self.publish_frame(img_bgr)
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
                "optitrack_cam module not found. "
                "Install/build the OptiTrack Python SDK and either place it in "
                "unity_hand_tracking/optitrack_cam_py/Release or set "
                "OPTITRACK_CAM_PY_PATH to its folder."
            )
        print("Initializing OptiTrack SDK...")
        optitrack_cam.initialize_sdk()
        time.sleep(discovery_wait)

        self.num_cameras = optitrack_cam.camera_count()
        print(f"Detected {self.num_cameras} cameras.")

        self.workers: list[CameraWorker] = []
        self._shutdown = False
        self.camera_indices = list(range(self.num_cameras))

    # -- lifecycle ----------------------------------------------------------

    def start_all(
        self,
        exposure=CAMERA_EXPOSURE,
        mjpeg_mode=MJPEG_MODE,
        acquisition_rounds: int = 10,
        retry_delay: float = 0.5,
    ):
        """Create and start a CameraWorker for every detected camera."""
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

        for index in range(self.num_cameras):
            worker = CameraWorker(
                handles[index], index, exposure=exposure, mjpeg_mode=mjpeg_mode
            )
            worker.start()
            self.workers.append(worker)
            print(f"Started worker for Camera {index}")

        return self.workers

    def wait_ready(self, timeout=3):
        for worker in self.workers:
            if not worker.ready.wait(timeout) or worker.frame_timestamp is None:
                raise RuntimeError(f"{worker.name} did not supply a frame: {worker.error}")

    def stop_all(self):
        """Stop every worker and shut down the SDK."""
        if self._shutdown:
            return
        print("Shutting down workers...")
        for w in self.workers:
            w.stop()
        for w in self.workers:
            w.join(timeout=3)
        alive = [worker.name for worker in self.workers if worker.is_alive()]
        if alive:
            raise RuntimeError(f"Camera workers did not stop: {alive}")
        optitrack_cam.shutdown_sdk()
        self._shutdown = True
        self.workers.clear()
        print("Cleanup complete.")

    # -- frame access -------------------------------------------------------

    def set_exposure(self, cam_index: int, value: int):
        """Set exposure for a specific camera at runtime."""
        if 0 <= cam_index < len(self.workers):
            self.workers[cam_index].set_exposure(value)

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
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
