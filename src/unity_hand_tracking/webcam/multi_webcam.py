"""
multi_webcam.py - Reusable webcam multi-camera module.

Provides:
  - CameraWorker: threaded per-camera frame grabber (cv2.VideoCapture)
  - CameraManager: discovers webcams, manages workers, builds grid frames
  - main(): standalone demo viewer (run this file directly)
"""

import threading
import time
import math

import cv2
import numpy as np

try:
    from .config import (
        CAMERA_EXPOSURE,
        DISPLAY_SCALE,
        GRID_COLS,
        WEBCAM_INDICES,
    )
except ImportError:
    from config import (
        CAMERA_EXPOSURE,
        DISPLAY_SCALE,
        GRID_COLS,
        WEBCAM_INDICES,
    )


class CameraWorker(threading.Thread):
    """Dedicated thread that continuously grabs the latest frame for one webcam."""

    def __init__(self, cam_index, exposure=None):
        super().__init__(daemon=True)
        self.cam_index = cam_index
        self.name = f"Camera #{cam_index}"
        self.exposure = exposure
        self.running = True
        self.latest_frame = None
        self.lock = threading.Lock()
        self.cap = None
        self.width = 640
        self.height = 480

    def get_frame(self):
        """Return the latest BGR frame (copy), or a black placeholder."""
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        return np.zeros((self.height, self.width, 3), np.uint8)

    def get_resolution(self):
        return self.width, self.height

    def set_exposure(self, value):
        self.exposure = value
        if self.cap is not None and self.cap.isOpened():
            # None means keep auto exposure enabled.
            if value is None:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                return

            # Not all webcam drivers expose CAP_PROP_EXPOSURE. Ignore failures.
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, float(value))

    def stop(self):
        self.running = False

    def run(self):
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.cam_index)

        if not cap.isOpened():
            print(f"[{self.name}] Failed to open webcam index {self.cam_index}")
            self.running = False
            return

        self.cap = cap

        # Intentionally do not force camera properties (FPS/resolution/buffer/exposure)
        # so each webcam uses its native driver defaults.

        if self.exposure is not None:
            self.set_exposure(self.exposure)

        ok, frame = cap.read()
        if ok and frame is not None:
            self.height, self.width = frame.shape[:2]
            with self.lock:
                self.latest_frame = frame

        print(f"[{self.name}] Started at {self.width}x{self.height}")

        while self.running:
            ok, frame = cap.read()
            if ok and frame is not None:
                with self.lock:
                    self.latest_frame = frame
            time.sleep(0.002)

        cap.release()
        self.cap = None


class CameraManager:
    """Discovers webcams, starts CameraWorkers, and builds grid images."""

    @staticmethod
    def _choose_grid_cols(num_frames, preferred_cols):
        """Choose compact columns to minimize empty black tiles."""
        if num_frames <= 0:
            return 1
        if preferred_cols is None:
            preferred_cols = num_frames

        max_cols = max(1, min(preferred_cols, num_frames))
        best_cols = 1
        best_empty = 10**9
        best_balance = 10**9

        for cols in range(1, max_cols + 1):
            rows = math.ceil(num_frames / cols)
            empty = rows * cols - num_frames
            balance = abs(cols - rows)
            if (
                empty < best_empty
                or (
                    empty == best_empty
                    and (
                        balance < best_balance
                        or (balance == best_balance and cols > best_cols)
                    )
                )
            ):
                best_cols = cols
                best_empty = empty
                best_balance = balance
        return best_cols

    def __init__(self, indices=None):
        # If WEBCAM_INDICES is empty, probe a few default indices.
        candidate_indices = list(indices) if indices is not None else list(WEBCAM_INDICES)
        if not candidate_indices:
            candidate_indices = list(range(6))

        self.camera_indices = []
        for idx in candidate_indices:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)
            opened = cap.isOpened()
            cap.release()
            if opened:
                self.camera_indices.append(idx)

        self.num_cameras = len(self.camera_indices)
        print(f"Detected {self.num_cameras} webcam(s): {self.camera_indices}")

        self.workers = []

    def start_all(self, exposure=CAMERA_EXPOSURE, mjpeg_mode=None):
        """Create and start a CameraWorker for every detected webcam."""
        _ = mjpeg_mode
        for cam_index in self.camera_indices:
            worker = CameraWorker(cam_index, exposure=exposure)
            worker.start()
            self.workers.append(worker)
            print(f"Started worker for webcam index {cam_index}")
        return self.workers

    def stop_all(self):
        """Stop every worker and wait for cleanup."""
        print("Shutting down webcam workers...")
        for worker in self.workers:
            worker.stop()
        for worker in self.workers:
            worker.join(timeout=3)
        print("Cleanup complete.")

    def set_exposure(self, cam_index, value):
        """Set exposure for a worker by its local index in camera_indices."""
        if 0 <= cam_index < len(self.workers):
            self.workers[cam_index].set_exposure(value)

    def get_frame(self, cam_index):
        """Return latest frame from a worker by local index."""
        return self.workers[cam_index].get_frame()

    def get_all_frames(self):
        """Return all latest frames in worker order."""
        return [worker.get_frame() for worker in self.workers]

    def get_grid(self, grid_cols=GRID_COLS, scale=DISPLAY_SCALE, overlay_labels=True):
        """Stitch all camera frames into a single grid image."""
        frames = self.get_all_frames()
        if not frames:
            return np.zeros((480, 640, 3), np.uint8)

        grid_cols = self._choose_grid_cols(len(frames), grid_cols)

        if overlay_labels:
            for i, frame in enumerate(frames):
                source_idx = self.camera_indices[i]
                cv2.putText(
                    frame,
                    f"Camera #{i} (src {source_idx})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

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

    def get_resolution(self, cam_index=0):
        """Return (width, height) for a given worker index."""
        return self.workers[cam_index].get_resolution()


def main():
    mgr = CameraManager()
    if mgr.num_cameras == 0:
        print("No webcams found. Exiting.")
        return

    mgr.start_all()
    window_name = "Webcam Multi-Camera Live Feed"
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
