from types import SimpleNamespace

import pytest

from handtrack.tracker._mediapipe_modes import (
    MediaPipeModeTracker,
    TrackingMode,
)


@pytest.mark.parametrize("value", ["hands", "FACE", TrackingMode.POSE, " holistic "])
def test_tracking_mode_parse(value):
    assert isinstance(TrackingMode.parse(value), TrackingMode)


def test_tracking_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unknown tracking mode"):
        TrackingMode.parse("body")


def test_holistic_result_is_normalized(monkeypatch):
    raw = SimpleNamespace(
        left_hand_landmarks="left",
        right_hand_landmarks=None,
        face_landmarks="face",
        pose_landmarks="pose",
    )
    solution = SimpleNamespace(process=lambda _frame: raw, close=lambda: None)
    monkeypatch.setattr(MediaPipeModeTracker, "_create_solution", lambda self: setattr(self, "_solution", solution))

    tracker = MediaPipeModeTracker("holistic")
    result = tracker.process(object())

    assert result.mode is TrackingMode.HOLISTIC
    assert result.hands == ["left"]
    assert result.faces == ["face"]
    assert result.poses == ["pose"]


def test_switching_mode_closes_previous_solution(monkeypatch):
    closed = []

    def create(self):
        self._solution = SimpleNamespace(close=lambda mode=self.mode: closed.append(mode))

    monkeypatch.setattr(MediaPipeModeTracker, "_create_solution", create)
    tracker = MediaPipeModeTracker("hands")
    tracker.set_mode("face")

    assert closed == [TrackingMode.HANDS]
    assert tracker.mode is TrackingMode.FACE
