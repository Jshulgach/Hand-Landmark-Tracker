"""Unified MediaPipe tracking modes used by the public webcam GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import mediapipe as mp


class TrackingMode(str, Enum):
    HANDS = "hands"
    FACE = "face"
    POSE = "pose"
    HOLISTIC = "holistic"

    @classmethod
    def parse(cls, value: str | "TrackingMode") -> "TrackingMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            choices = ", ".join(mode.value for mode in cls)
            raise ValueError(f"Unknown tracking mode {value!r}; choose {choices}.") from exc


@dataclass(slots=True)
class TrackingResult:
    """Landmarks detected in one frame, grouped by body region."""

    mode: TrackingMode
    hands: list[Any] = field(default_factory=list)
    faces: list[Any] = field(default_factory=list)
    poses: list[Any] = field(default_factory=list)
    raw: Any = None


class MediaPipeModeTracker:
    """Own one MediaPipe solution and expose a stable multi-mode interface."""

    def __init__(
        self,
        mode: str | TrackingMode = TrackingMode.HANDS,
        *,
        max_hands: int = 2,
        max_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.max_hands = max(1, int(max_hands))
        self.max_faces = max(1, int(max_faces))
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.mode = TrackingMode.parse(mode)
        self._solution: Any = None
        self._create_solution()

    def _create_solution(self) -> None:
        common = {
            "min_detection_confidence": self.min_detection_confidence,
            "min_tracking_confidence": self.min_tracking_confidence,
        }
        if self.mode is TrackingMode.HANDS:
            self._solution = mp.solutions.hands.Hands(
                max_num_hands=self.max_hands,
                **common,
            )
        elif self.mode is TrackingMode.FACE:
            self._solution = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=self.max_faces,
                refine_landmarks=True,
                **common,
            )
        elif self.mode is TrackingMode.POSE:
            self._solution = mp.solutions.pose.Pose(**common)
        else:
            self._solution = mp.solutions.holistic.Holistic(
                refine_face_landmarks=True,
                **common,
            )

    def set_mode(self, mode: str | TrackingMode) -> None:
        new_mode = TrackingMode.parse(mode)
        if new_mode is self.mode:
            return
        self.close()
        self.mode = new_mode
        self._create_solution()

    def process(self, rgb_frame: Any) -> TrackingResult:
        raw = self._solution.process(rgb_frame)
        result = TrackingResult(mode=self.mode, raw=raw)

        if self.mode is TrackingMode.HANDS:
            result.hands = list(raw.multi_hand_landmarks or [])
        elif self.mode is TrackingMode.FACE:
            result.faces = list(raw.multi_face_landmarks or [])
        elif self.mode is TrackingMode.POSE:
            if raw.pose_landmarks is not None:
                result.poses = [raw.pose_landmarks]
        else:
            result.hands = [
                landmarks
                for landmarks in (raw.left_hand_landmarks, raw.right_hand_landmarks)
                if landmarks is not None
            ]
            if raw.face_landmarks is not None:
                result.faces = [raw.face_landmarks]
            if raw.pose_landmarks is not None:
                result.poses = [raw.pose_landmarks]

        return result

    @staticmethod
    def draw(frame: Any, result: TrackingResult) -> Any:
        drawing = mp.solutions.drawing_utils
        styles = mp.solutions.drawing_styles

        for hand in result.hands:
            drawing.draw_landmarks(
                frame,
                hand,
                mp.solutions.hands.HAND_CONNECTIONS,
                styles.get_default_hand_landmarks_style(),
                styles.get_default_hand_connections_style(),
            )
        for face in result.faces:
            drawing.draw_landmarks(
                frame,
                face,
                mp.solutions.face_mesh.FACEMESH_CONTOURS,
                connection_drawing_spec=styles.get_default_face_mesh_contours_style(),
            )
            drawing.draw_landmarks(
                frame,
                face,
                mp.solutions.face_mesh.FACEMESH_IRISES,
                connection_drawing_spec=styles.get_default_face_mesh_iris_connections_style(),
            )
        for pose in result.poses:
            drawing.draw_landmarks(
                frame,
                pose,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=styles.get_default_pose_landmarks_style(),
            )
        return frame

    def close(self) -> None:
        if self._solution is not None:
            self._solution.close()
            self._solution = None

    def __enter__(self) -> "MediaPipeModeTracker":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
