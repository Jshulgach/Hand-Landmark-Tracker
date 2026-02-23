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

import sys
import threading
import time

import cv2
import numpy as np

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
class CameraWorker(threading.Thread):
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

    # -- public helpers (safe to call from any thread) ----------------------

    def get_frame(self):
        """Return the latest BGR frame (copy), or a black placeholder."""
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        return np.zeros((self.height, self.width, 3), np.uint8)

    def get_resolution(self):
        return self.width, self.height

    def set_exposure(self, value):
        """Update exposure at runtime (thread-safe via SDK)."""
        self.exposure = value
        self.camera.set_exposure(value)

    def stop(self):
        self.running = False
        self.camera.stop()
        self.camera.release()

    # -- thread body --------------------------------------------------------

    def run(self):
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
                img_rgba = frame_obj.rasterize(self.width, self.height)
                img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
                with self.lock:
                    self.latest_frame = img_bgr
                frame_obj.release()
            else:
                time.sleep(0.001)


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
        print("Initializing OptiTrack SDK...")
        optitrack_cam.initialize_sdk()
        time.sleep(discovery_wait)

        self.num_cameras = optitrack_cam.camera_count()
        print(f"Detected {self.num_cameras} cameras.")

        self.workers: list[CameraWorker] = []

    # -- lifecycle ----------------------------------------------------------

    def start_all(self, exposure=CAMERA_EXPOSURE, mjpeg_mode=MJPEG_MODE):
        """Create and start a CameraWorker for every detected camera."""
        for i in range(self.num_cameras):
            cam = optitrack_cam.get_camera_by_index(i)
            if cam:
                worker = CameraWorker(cam, i, exposure=exposure, mjpeg_mode=mjpeg_mode)
                worker.start()
                self.workers.append(worker)
                print(f"Started worker for Camera {i}")
        return self.workers

    def stop_all(self):
        """Stop every worker and shut down the SDK."""
        print("Shutting down workers...")
        for w in self.workers:
            w.stop()
        for w in self.workers:
            w.join(timeout=3)
        optitrack_cam.shutdown_sdk()
        print("Cleanup complete.")

    # -- frame access -------------------------------------------------------

    def set_exposure(self, cam_index: int, value: int):
        """Set exposure for a specific camera at runtime."""
        if 0 <= cam_index < len(self.workers):
            self.workers[cam_index].set_exposure(value)

    def get_frame(self, cam_index: int):
        """Return the latest BGR frame from a specific camera."""
        return self.workers[cam_index].get_frame()

    def get_all_frames(self):
        """Return a list of BGR frames, one per camera (in index order)."""
        return [w.get_frame() for w in self.workers]

    def get_grid(self, grid_cols: int = GRID_COLS, scale: float = DISPLAY_SCALE,
                 overlay_labels: bool = True):
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

        if overlay_labels:
            for i, f in enumerate(frames):
                cv2.putText(
                    f, f"Camera #{i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                )

        # Build row-major grid, padding incomplete rows with black
        rows = []
        for start in range(0, len(frames), grid_cols):
            row = frames[start: start + grid_cols]
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
