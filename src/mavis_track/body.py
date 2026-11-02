"""Single-person hand/body tracking on one caller-owned frame and clock."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import IntFlag
from typing import Literal

import cv2
import numpy as np

from .tracker import FrameResult, HandResult

BODY_LANDMARK_NAMES = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index",
    "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
)
QUALITY_REGION_NAMES = ("left_hand", "right_hand", "body")


class QualityFlag(IntFlag):
    """Observable tracking conditions, not calibrated accuracy probabilities."""

    OK = 0
    MISSING = 1
    LOW_VISIBILITY = 2
    OUT_OF_FRAME = 4
    NONFINITE = 8
    TEMPORAL_GAP = 16
    TRACKING_RESET = 32


@dataclass(frozen=True, slots=True)
class TrackingQuality:
    left_hand: QualityFlag = QualityFlag.MISSING
    right_hand: QualityFlag = QualityFlag.MISSING
    body: QualityFlag = QualityFlag.MISSING


@dataclass(frozen=True, slots=True)
class BodyResult:
    """33 image-normalized points; world estimates are hip-centered meters.

    Visibility is a model output, not a localization-error estimate. Arrays are
    read-only snapshots. Neither coordinate space is calibrated scene geometry.
    """

    landmarks: np.ndarray
    world_landmarks: np.ndarray | None
    visibility: np.ndarray
    landmark_names: tuple[str, ...] = BODY_LANDMARK_NAMES


@dataclass(frozen=True, slots=True)
class HandBodyFrameResult(FrameResult):
    body: BodyResult | None = None
    quality: TrackingQuality = TrackingQuality()


def _readonly(array):
    array.flags.writeable = False
    return array


def _region(landmarks, count):
    if landmarks is None:
        return None, QualityFlag.MISSING
    points = np.array([(p.x, p.y, p.z) for p in landmarks.landmark], dtype=np.float32)
    if points.shape != (count, 3) or not np.isfinite(points).all():
        return None, QualityFlag.MISSING | QualityFlag.NONFINITE
    flags = QualityFlag.OK
    if np.any((points[:, :2] < 0) | (points[:, :2] > 1)):
        flags |= QualityFlag.OUT_OF_FRAME
    return _readonly(points), flags


class HandBodyTracker:
    """Track one person's body and two hands using MediaPipe Holistic.

    The caller owns capture and scheduling. Each process call runs exactly one
    Holistic inference. Input is never modified. No camera, GUI or face results
    are exposed. Face inference remains internal to the Holistic model.

    ``mirrored=True`` means input pixels were flipped horizontally: swap the
    anatomical side labels and body left/right indices back to subject sides.
    Hand-world estimates are unavailable from this backend. Hand angles use
    image-scaled geometry, not calibrated joint measurements. Only hands are
    optionally smoothed; quality always describes the raw observations.
    """

    def __init__(self, *, confidence: float = 0.5, tracking_confidence: float = 0.5,
                 visibility_threshold: float = 0.5, max_gap: float = 0.5,
                 mirrored: bool = False, source_id: str = "frames",
                 smoothing: bool = False) -> None:
        for name, value in (("confidence", confidence),
                            ("tracking_confidence", tracking_confidence),
                            ("visibility_threshold", visibility_threshold)):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if not np.isfinite(max_gap) or max_gap <= 0:
            raise ValueError("max_gap must be positive finite seconds")
        import mediapipe as mp
        self._solution = mp.solutions.holistic.Holistic(
            smooth_landmarks=False, min_detection_confidence=confidence,
            min_tracking_confidence=tracking_confidence)
        self.mirrored = mirrored
        self.source_id = str(source_id)
        self.smoothing = smoothing
        self.visibility_threshold = float(visibility_threshold)
        self.max_gap = float(max_gap)
        self._frame_id = 0
        self._last_timestamp: float | None = None
        self._tracks: dict[str, tuple[int, np.ndarray]] = {}
        self._seen: set[str] = set()
        self._filters: dict[str, list] = {}
        self._next_id = 0
        # Reflection changes semantic left/right labels, not coordinate axes.
        self._body_order = list(range(33))
        if mirrored:
            for index, name in enumerate(BODY_LANDMARK_NAMES):
                opposite = ("right_" + name[5:] if name.startswith("left_") else
                            "left_" + name[6:] if name.startswith("right_") else
                            "mouth_right" if name == "mouth_left" else
                            "mouth_left" if name == "mouth_right" else name)
                self._body_order[index] = BODY_LANDMARK_NAMES.index(opposite)

    def process(self, frame: np.ndarray, *, color: Literal["bgr", "rgb"] = "bgr",
                timestamp: float | None = None) -> HandBodyFrameResult:
        if self._solution is None:
            raise RuntimeError("Tracker is closed; create a new HandBodyTracker")
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
        gap = self._last_timestamp is not None and stamp - self._last_timestamp > self.max_gap
        gap_flag = QualityFlag.TEMPORAL_GAP if gap else QualityFlag.OK
        old = {} if gap else self._tracks
        tracks = {}
        hands = []
        quality = {}
        from handtrack.processing._joint_angles import compute_all_joint_angles
        for side, attribute in (("Left", "left_hand_landmarks"),
                                ("Right", "right_hand_landmarks")):
            label = ({"Left": "Right", "Right": "Left"}[side] if self.mirrored else side)
            points, flags = _region(getattr(raw, attribute, None), 21)
            flags |= gap_flag
            if points is not None:
                previous = old.get(label)
                if previous is None or np.linalg.norm(points[0, :2] - previous[1][:2]) > 0.25:
                    key = self._next_id
                    self._next_id += 1
                    self._filters.pop(label, None)
                    if label in self._seen:
                        flags |= QualityFlag.TRACKING_RESET
                else:
                    key = previous[0]
                tracks[label] = (key, points[0].copy())
                self._seen.add(label)
                filtered = points.copy()
                if self.smoothing:
                    from handtrack.processing._kalman_filter import Kalman3D
                    if label not in self._filters:
                        self._filters[label] = [Kalman3D() for _ in range(21)]
                    dt = stamp - self._last_timestamp if self._last_timestamp is not None else 1 / 30
                    filtered = np.array([f.update(p, dt=dt) for f, p in
                                         zip(self._filters[label], points)], dtype=np.float32)
                angles = compute_all_joint_angles(filtered * [width, height, width])
                hands.append(HandResult(key, label, float("nan"), _readonly(filtered),
                                        points, None, angles, "image_scaled"))
            quality[label] = flags
        body_points, body_flags = _region(getattr(raw, "pose_landmarks", None), 33)
        body_flags |= gap_flag
        body = None
        if body_points is not None:
            body_points = _readonly(body_points[self._body_order].copy())
            visibility = np.array([getattr(p, "visibility", np.nan)
                                   for p in raw.pose_landmarks.landmark], dtype=np.float32)[self._body_order]
            if not np.isfinite(visibility).all() or np.any((visibility < 0) | (visibility > 1)):
                visibility[:] = np.nan
                body_flags |= QualityFlag.NONFINITE | QualityFlag.LOW_VISIBILITY
            elif np.any(visibility < self.visibility_threshold):
                body_flags |= QualityFlag.LOW_VISIBILITY
            world = None
            world_raw = getattr(raw, "pose_world_landmarks", None)
            if world_raw is not None:
                world_points, _ = _region(world_raw, 33)
                if world_points is None:
                    body_flags |= QualityFlag.NONFINITE
                else:
                    world = _readonly(world_points[self._body_order].copy())
            body = BodyResult(body_points, world, _readonly(visibility))
        self._tracks = tracks
        self._filters = {key: value for key, value in self._filters.items() if key in tracks}
        self._last_timestamp = stamp
        result = HandBodyFrameResult(
            self._frame_id, stamp, self.source_id, (width, height), tuple(hands),
            timestamp_clock="source" if timestamp is not None else "monotonic", body=body,
            quality=TrackingQuality(quality["Left"], quality["Right"], body_flags))
        self._frame_id += 1
        return result

    def close(self) -> None:
        if self._solution is not None:
            solution, self._solution = self._solution, None
            try:
                solution.close()
            finally:
                self._tracks.clear()
                self._filters.clear()
                self._seen.clear()

    def __enter__(self) -> HandBodyTracker:  # noqa: PYI034
        if self._solution is None:
            raise RuntimeError("Tracker is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
