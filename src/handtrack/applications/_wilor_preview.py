"""Official WiLoR live overlay; imported only when selected."""
from __future__ import annotations

import cv2
import numpy as np

from handtrack.tracker import TrackingMode
from mavis_track.wilor import WiLoRTracker

_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4)]
for _base in (5, 9, 13, 17):
    _CONNECTIONS.extend([(0, _base), (_base, _base + 1),
                         (_base + 1, _base + 2), (_base + 2, _base + 3)])


class WiLoRModeTracker:
    mode = TrackingMode.HANDS

    def __init__(self, *, root=None, device="auto", fast=False):
        self.tracker = WiLoRTracker(root=root, device=device, fast=fast)

    def set_mode(self, mode):
        if TrackingMode.parse(mode) is not TrackingMode.HANDS:
            raise ValueError("WiLoR supports hands only; select MediaPipe for body or face")

    def process(self, rgb):
        return self.tracker.process(rgb, color="rgb")

    def draw(self, frame, result):
        # Full-image projections only. Display clipping never changes inference.
        height, width = frame.shape[:2]
        limit = max(height, width) * 4
        for hand in result.hands:
            mesh = np.rint(np.clip(hand.image_vertices, -limit, limit)).astype(np.int32)
            triangles = mesh[self.tracker.faces]
            cv2.polylines(frame, list(triangles), True, (130, 90, 40), 1, cv2.LINE_AA)
            points = np.rint(np.clip(hand.image_landmarks, -limit, limit)).astype(np.int32)
            for start, end in _CONNECTIONS:
                cv2.line(frame, tuple(points[start]), tuple(points[end]),
                         (80, 220, 160), 2, cv2.LINE_AA)
            for point in points:
                cv2.circle(frame, tuple(point), 3, (230, 230, 230), -1, cv2.LINE_AA)
            x, y = np.rint(hand.bbox[:2]).astype(int)
            x, y = int(np.clip(x, 0, width - 1)), int(np.clip(y - 8, 16, max(16, height - 1)))
            cv2.putText(frame, f"{hand.handedness} | box {hand.detection_score:.2f}",
                        (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (80, 220, 160), 1, cv2.LINE_AA)

    def close(self):
        self.tracker.close()
