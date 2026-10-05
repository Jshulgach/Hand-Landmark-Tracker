"""Hand tracking for caller-owned images and event loops."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class HandResult:
    """One detection. World landmarks are hand-centered model estimates in meters.

    ``landmarks`` uses image-normalized x/y and MediaPipe's wrist-relative z.
    ``track_id`` is a short-lived association, not biometric identity.
    Arrays are owned by this result and are read-only.
    """

    track_id: int
    handedness: str
    handedness_score: float
    landmarks: np.ndarray
    raw_landmarks: np.ndarray
    world_landmarks: np.ndarray | None
    angles: dict[str, float]
    angle_space: str


@dataclass(frozen=True, slots=True)
class FrameResult:
    frame_id: int
    timestamp: float
    source_id: str
    image_size: tuple[int, int]
    hands: tuple[HandResult, ...]
    coordinate_space: str = "image_normalized"
    timestamp_clock: str = "monotonic"


def _points(landmarks) -> np.ndarray:
    points = np.array([(p.x, p.y, p.z) for p in landmarks.landmark], dtype=np.float32)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("A hand must contain 21 finite three-dimensional landmarks")
    return points


class HandTracker:
    """Process BGR/RGB uint8 frames without opening capture or showing a window.

    Pass monotonically increasing seconds in ``timestamp`` when your source has
    its own clock. Otherwise ``time.perf_counter()`` is used. Set ``mirrored``
    to true only when input frames have been horizontally mirrored.
    """

    def __init__(self, *, max_hands: int = 2, confidence: float = 0.5,
                 tracking_confidence: float = 0.5, mirrored: bool = False,
                 source_id: str = "frames", smoothing: bool = False) -> None:
        if isinstance(max_hands, bool) or not isinstance(max_hands, (int, np.integer)) or max_hands not in (1, 2):
            raise ValueError("max_hands must be 1 or 2")
        for name, value in (("confidence", confidence),
                            ("tracking_confidence", tracking_confidence)):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        import mediapipe as mp
        self._solution = mp.solutions.hands.Hands(
            max_num_hands=max_hands, min_detection_confidence=confidence,
            min_tracking_confidence=tracking_confidence)
        self.max_hands = max_hands
        self.mirrored = mirrored
        self.source_id = str(source_id)
        self.smoothing = smoothing
        self._frame_id = 0
        self._last_timestamp: float | None = None
        self._tracks: dict[int, tuple[np.ndarray, str, float]] = {}
        self._filters: dict[int, list] = {}
        self._next_id = 0

    def process(self, frame: np.ndarray, *, color: Literal["bgr", "rgb"] = "bgr",
                timestamp: float | None = None) -> FrameResult:
        if self._solution is None:
            raise RuntimeError("Tracker is closed; create a new HandTracker")
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1):
            raise ValueError("frame must be a nonempty uint8 image with three channels")
        if color not in ("bgr", "rgb"):
            raise ValueError("color must be 'bgr' or 'rgb'")
        stamp = time.perf_counter() if timestamp is None else float(timestamp)
        if not np.isfinite(stamp) or stamp < 0:
            raise ValueError("timestamp must be finite, nonnegative seconds")
        if self._last_timestamp is not None and stamp <= self._last_timestamp:
            raise ValueError("timestamps must increase between frames")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if color == "bgr" else frame.copy()
        rgb = np.ascontiguousarray(rgb)
        rgb.flags.writeable = False
        raw = self._solution.process(rgb)
        height, width = frame.shape[:2]
        detections = list(raw.multi_hand_landmarks or [])[:self.max_hands]
        handedness = list(getattr(raw, "multi_handedness", None) or [])
        worlds = list(getattr(raw, "multi_hand_world_landmarks", None) or [])
        # Only adjacent-frame associations retain filter state. A missing hand
        # never silently reuses another hand's smoother after reappearance.
        old = self._tracks if (self._last_timestamp is not None
                              and stamp - self._last_timestamp <= 0.5) else {}
        available = set(old)
        tracks = {}
        hands = []
        from handtrack.processing._joint_angles import compute_all_joint_angles
        for index, detection in enumerate(detections):
            points = _points(detection)
            raw_points = points.copy()
            label, score = "Unknown", 0.0
            if index < len(handedness) and handedness[index].classification:
                classification = handedness[index].classification[0]
                label, score = classification.label, float(classification.score)
                if not self.mirrored and label in {"Left", "Right"}:
                    label = "Right" if label == "Left" else "Left"
            candidates = [(float(np.linalg.norm(points[0, :2] - old[key][0][0, :2])), key)
                          for key in available if old[key][1] == label]
            distance, key = min(candidates, default=(float("inf"), -1))
            if distance > 0.25:
                key = self._next_id
                self._next_id += 1
            else:
                available.remove(key)
            tracks[key] = (points.copy(), label, stamp)
            if self.smoothing:
                from handtrack.processing._kalman_filter import Kalman3D
                if key not in self._filters:
                    self._filters[key] = [Kalman3D() for _ in range(21)]
                dt = stamp - self._last_timestamp if self._last_timestamp is not None else 1 / 30
                points = np.array([f.update(p, dt=dt) for f, p in
                                   zip(self._filters[key], points)], dtype=np.float32)
            world = _points(worlds[index]) if index < len(worlds) else None
            angle_points = world if world is not None else points * [width, height, width]
            angles = compute_all_joint_angles(angle_points)
            points.flags.writeable = False
            raw_points.flags.writeable = False
            if world is not None:
                world.flags.writeable = False
            hands.append(HandResult(key, label, score, points, raw_points, world, angles,
                                    "hand_world_meters" if world is not None else "image_scaled"))
        self._tracks = tracks
        self._filters = {key: value for key, value in self._filters.items() if key in tracks}
        result = FrameResult(self._frame_id, stamp, self.source_id, (width, height), tuple(hands),
                             timestamp_clock="source" if timestamp is not None else "monotonic")
        self._frame_id += 1
        self._last_timestamp = stamp
        return result

    def close(self) -> None:
        if self._solution is not None:
            solution, self._solution = self._solution, None
            solution.close()
        self._tracks.clear()
        self._filters.clear()

    def __enter__(self) -> HandTracker:  # noqa: PYI034 - Python 3.10 supports no typing.Self
        if self._solution is None:
            raise RuntimeError("Tracker is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
