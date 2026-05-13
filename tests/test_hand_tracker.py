import numpy as np

from handtrack.tracker._hand_tracker import HandTracker


class FakeCapture:
    def __init__(self, frames, reported_total_frames=None, fps=30.0, width=640, height=480):
        self._frames = list(frames)
        self._reported_total_frames = len(self._frames) if reported_total_frames is None else reported_total_frames
        self._fps = fps
        self._width = width
        self._height = height
        self._read_index = 0
        self._released = False

    def isOpened(self):
        return not self._released

    def read(self):
        if self._released or self._read_index >= len(self._frames):
            return False, None
        frame = self._frames[self._read_index]
        self._read_index += 1
        return True, frame.copy()

    def set(self, prop, value):
        if prop == 3:
            self._width = value
        elif prop == 4:
            self._height = value
        elif prop == 5:
            self._fps = value
        return True

    def get(self, prop):
        if prop == 7:
            return self._reported_total_frames
        if prop == 5:
            return self._fps
        if prop == 3:
            return self._width
        if prop == 4:
            return self._height
        return 0

    def release(self):
        self._released = True


class FakeHandsProcessor:
    def __init__(self, results):
        self._results = list(results)
        self._index = 0

    def process(self, rgb):
        del rgb
        if self._index < len(self._results):
            result = self._results[self._index]
            self._index += 1
            return result
        return FakeResults([])


class FakeResults:
    def __init__(self, hand_landmarks):
        self.multi_hand_landmarks = hand_landmarks


class FakeLandmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class FakeHandLandmarks:
    def __init__(self, coords):
        self.landmark = [FakeLandmark(*coord) for coord in coords]


class FakeKalman3D:
    next_id = 0

    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.filter_id = FakeKalman3D.next_id
        FakeKalman3D.next_id += 1

    def update(self, z):
        return np.asarray(z, dtype=np.float32) + np.float32(self.filter_id)


def build_hand(base_value):
    return FakeHandLandmarks(
        [(base_value + index, base_value + index + 0.1, base_value + index + 0.2) for index in range(21)]
    )


def build_tracker(monkeypatch, frames, results, *, max_hands=1, apply_kalman=True, reported_total_frames=None, fps=30.0):
    from handtrack.tracker import _hand_tracker as hand_tracker_module

    capture = FakeCapture(frames, reported_total_frames=reported_total_frames, fps=fps)
    monkeypatch.setattr(hand_tracker_module.cv2, "VideoCapture", lambda source: capture)
    monkeypatch.setattr(hand_tracker_module.mp.solutions.hands, "Hands", lambda **kwargs: FakeHandsProcessor(results))
    tracker = HandTracker(source="fake.mp4", max_hands=max_hands, apply_kalman=apply_kalman)
    return tracker, capture


def test_extract_landmarks_uses_processed_frame_count(monkeypatch):
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
    results = [FakeResults([]), FakeResults([]), FakeResults([])]
    tracker, capture = build_tracker(
        monkeypatch,
        frames,
        results,
        max_hands=1,
        apply_kalman=False,
        reported_total_frames=9,
        fps=25.0,
    )

    landmarks, metadata = tracker.extract_landmarks(visualize=False, save_video=False)

    assert landmarks.shape == (3, 21, 3)
    assert metadata["total_frames"] == 3
    assert metadata["sampling_rate"] == 25.0
    assert metadata["time_vector"].shape == (3,)
    assert np.allclose(metadata["time_vector"], np.array([0.0, 1.0 / 25.0, 2.0 / 25.0], dtype=np.float32))
    assert capture._released is True


def test_extract_landmarks_preserves_multi_hand_shape(monkeypatch):
    frame = np.zeros((6, 6, 3), dtype=np.uint8)
    results = [FakeResults([build_hand(10.0), build_hand(100.0)])]
    tracker, _ = build_tracker(monkeypatch, [frame], results, max_hands=2, apply_kalman=False)

    landmarks, metadata = tracker.extract_landmarks(visualize=False, save_video=False)

    assert landmarks.shape == (1, 2, 21, 3)
    assert metadata["total_frames"] == 1
    assert np.allclose(landmarks[0, 0, 0], np.array([10.0, 10.1, 10.2], dtype=np.float32))
    assert np.allclose(landmarks[0, 1, 0], np.array([100.0, 100.1, 100.2], dtype=np.float32))


def test_prepare_landmarks_uses_distinct_filter_bank_per_hand(monkeypatch):
    from handtrack.tracker import _hand_tracker as hand_tracker_module

    FakeKalman3D.next_id = 0
    monkeypatch.setattr(hand_tracker_module, "Kalman3D", FakeKalman3D)

    tracker, _ = build_tracker(
        monkeypatch,
        [np.zeros((4, 4, 3), dtype=np.uint8)],
        [FakeResults([])],
        max_hands=2,
        apply_kalman=True,
    )

    raw_landmarks, filtered_landmarks, detected_count = tracker._prepare_landmarks([build_hand(1.0), build_hand(10.0)])

    assert detected_count == 2
    assert raw_landmarks.shape == (2, 21, 3)
    assert filtered_landmarks.shape == (2, 21, 3)
    assert np.allclose(filtered_landmarks[0, 0], np.array([1.0, 1.1, 1.2], dtype=np.float32))
    assert np.allclose(filtered_landmarks[1, 0], np.array([31.0, 31.1, 31.2], dtype=np.float32))
