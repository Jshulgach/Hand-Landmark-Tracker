import threading
import time

import cv2
import numpy as np
import optitrack_cam

# --- Configuration ---
DISPLAY_SCALE = 0.5  # Adjust this to fit 6 cameras on your screen
EXPOSURE = (
    10  # Low exposure for maximum FPS (valid range: 1-480, lower = faster/darker)
)
MJPEG_MODE = 6  # Core::MJPEGMode
GRID_COLS = 3  # 3x2 grid for 6 cameras


class CameraWorker(threading.Thread):
    """
    Dedicated thread for each camera to drain the buffer continuously.
    This ensures the 'latest' frame is always ready for the UI.
    """

    def __init__(self, camera, index):
        super().__init__()
        self.camera = camera
        self.index = index
        self.name = f"Camera #{index}"
        self.running = True
        self.latest_frame = None
        self.lock = threading.Lock()
        self.width = camera.width()
        self.height = camera.height()

    def run(self):
        # Configure Camera inside the thread
        self.camera.set_video_type(MJPEG_MODE)

        # CRITICAL: Set manual exposure BEFORE disabling AEC
        self.camera.set_exposure(EXPOSURE)
        self.camera.set_aec(False)  # Disable auto-exposure after setting value

        self.camera.set_agc(True)
        self.camera.set_text_overlay(True)
        self.camera.start()

        # Wait a moment for camera to apply settings
        time.sleep(0.2)

        # Verify actual exposure value
        actual_exposure = self.camera.get_exposure()
        print(
            f"[{self.name}] Configured: Exposure={EXPOSURE}, AEC=False, AGC=True | Actual Exposure={actual_exposure}"
        )

        while self.running:
            # Drain the queue to remove delay (same logic as get_latest_frame)
            frame_obj = self.camera.get_latest_frame()

            if frame_obj:
                # Process the newest frame
                img_rgba = frame_obj.rasterize(self.width, self.height)
                img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)

                # Overlay label
                cv2.putText(
                    img_bgr,
                    self.name,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                # Thread-safe update of the frame to be displayed
                with self.lock:
                    self.latest_frame = img_bgr

                frame_obj.release()
            else:
                time.sleep(0.001)  # Small sleep if no frame to prevent CPU maxing

    def get_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            # Return placeholder if no frame yet
            return np.zeros((self.height, self.width, 3), np.uint8)

    def stop(self):
        self.running = False
        self.camera.stop()
        self.camera.release()


def main():
    print("Initializing SDK for Multi-Camera View...")
    optitrack_cam.initialize_sdk()
    time.sleep(2)  # Discovery time

    num_cameras = optitrack_cam.camera_count()
    print(f"Detected {num_cameras} cameras.")

    if num_cameras == 0:
        print("No cameras found. Exiting.")
        return

    workers = []
    for i in range(num_cameras):
        cam = optitrack_cam.get_camera_by_index(i)
        if cam:
            worker = CameraWorker(cam, i)
            worker.start()
            workers.append(worker)
            print(f"Started worker for Camera {i}")

    window_name = "OptiTrack 6-Camera Live Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            frames = [w.get_frame() for w in workers]

            # Create the Grid (3 columns)
            # Split frames into rows of 3
            rows_data = [
                frames[i : i + GRID_COLS] for i in range(0, len(frames), GRID_COLS)
            ]

            full_rows = []
            for r in rows_data:
                # If a row is incomplete (e.g. 5 cameras), pad it with black frames
                while len(r) < GRID_COLS:
                    r.append(np.zeros_like(frames[0]))
                full_rows.append(np.hstack(r))

            grid = np.vstack(full_rows)

            # Resize for screen fit
            if DISPLAY_SCALE != 1.0:
                grid = cv2.resize(grid, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

            cv2.imshow(window_name, grid)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break
    finally:
        print("Shutting down workers...")
        for w in workers:
            w.stop()
        for w in workers:
            w.join()

        optitrack_cam.shutdown_sdk()
        cv2.destroyAllWindows()
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
