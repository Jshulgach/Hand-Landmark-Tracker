import threading

import numpy as np

from handtrack.cameras.optitrack.mocap_tracker import MultiCameraTracker
from handtrack.cameras.optitrack.multi_mjpeg import CameraWorker


class RecordingDetector:
    def __init__(self):
        self.shape = None
        self.writeable = None

    def process(self, image):
        self.shape = image.shape
        self.writeable = image.flags.writeable
        return object()


def test_mediapipe_input_is_downscaled_and_read_only():
    detector = RecordingDetector()
    frame = np.zeros((1024, 1280, 3), dtype=np.uint8)

    result = MultiCameraTracker._detect_single_camera(detector, frame, 0)

    assert result is not None
    assert detector.shape == (512, 640, 3)
    assert detector.writeable is False


def test_camera_worker_can_return_zero_copy_frame_view():
    worker = CameraWorker.__new__(CameraWorker)
    worker.lock = threading.Lock()
    worker.latest_frame = np.zeros((8, 10, 3), dtype=np.uint8)
    worker.status = "streaming"
    worker.error = ""

    view = worker.get_frame(copy=False)
    copied = worker.get_frame()

    assert view is worker.latest_frame
    assert copied is not worker.latest_frame
    assert np.array_equal(copied, worker.latest_frame)


def test_tracker_requests_zero_copy_camera_frames():
    requested = []
    frames = [np.zeros((4, 4, 3), dtype=np.uint8)]

    class CameraManager:
        def get_all_frames(self, copy=True):
            requested.append(copy)
            return frames

    tracker = MultiCameraTracker()
    tracker.cam_mgr = CameraManager()

    assert tracker.capture_frames() is frames
    assert requested == [False]
