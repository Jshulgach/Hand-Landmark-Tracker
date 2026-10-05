from types import SimpleNamespace
import numpy as np
import pytest

from mavis_track import HandTracker


def detection(x, label):
    points = SimpleNamespace(landmark=[SimpleNamespace(x=x + i * .002, y=.2 + i * .003, z=.001 * i) for i in range(21)])
    handedness = SimpleNamespace(classification=[SimpleNamespace(label=label, score=.95)])
    return points, handedness


def results(*hands):
    return SimpleNamespace(multi_hand_landmarks=[hand[0] for hand in hands],
                           multi_handedness=[hand[1] for hand in hands])


def solution(monkeypatch, outputs):
    import mediapipe as mp
    queue = iter(outputs)
    state = {"closed": False}
    def process(image):
        assert not image.flags.writeable
        return next(queue)
    fake = SimpleNamespace(process=process, close=lambda: state.update(closed=True))
    monkeypatch.setattr(mp.solutions.hands, "Hands", lambda **kwargs: fake)
    return state


def test_frame_api_retains_identity_when_detection_order_changes(monkeypatch):
    a, b = detection(.2, "Left"), detection(.7, "Right")
    state = solution(monkeypatch, [results(a, b), results(b, a), results(), results(a)])
    image = np.zeros((64, 64, 3), np.uint8)
    with HandTracker(mirrored=True, smoothing=True) as tracker:
        first = tracker.process(image, timestamp=0)
        second = tracker.process(image, timestamp=.1)
        assert [(hand.handedness, hand.track_id) for hand in first.hands] == [("Left", 0), ("Right", 1)]
        assert [hand.track_id for hand in second.hands] == [1, 0]
        assert tracker.process(image, timestamp=.2).hands == ()
        assert tracker.process(image, timestamp=.3).hands[0].track_id == 2
    assert state["closed"]
    assert image.flags.writeable
    assert not first.hands[0].landmarks.flags.writeable
    assert np.allclose(first.hands[0].landmarks, first.hands[0].raw_landmarks)


def test_frame_api_owns_no_capture_and_corrects_handedness(monkeypatch):
    import cv2
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: pytest.fail("Frame tracker opened a camera"))
    solution(monkeypatch, [results(detection(.2, "Left"))])
    with HandTracker() as tracker:
        result = tracker.process(np.zeros((8, 9, 3), np.uint8), color="rgb")
    assert result.hands[0].handedness == "Right"
    assert result.image_size == (9, 8)


def test_frame_api_rejects_bad_images_clock_and_use_after_close(monkeypatch):
    solution(monkeypatch, [results()])
    tracker = HandTracker()
    with pytest.raises(ValueError, match="uint8"):
        tracker.process(np.zeros((4, 4, 3)))
    image = np.zeros((4, 4, 3), np.uint8)
    tracker.process(image, timestamp=1)
    with pytest.raises(ValueError, match="increase"):
        tracker.process(image, timestamp=1)
    tracker.close()
    tracker.close()
    with pytest.raises(RuntimeError, match="closed"):
        tracker.process(image)


def test_context_closes_after_processing_failure(monkeypatch):
    state = solution(monkeypatch, [])
    with pytest.raises(StopIteration):
        with HandTracker() as tracker:
            tracker.process(np.zeros((4, 4, 3), np.uint8))
    assert state["closed"]
