"""Replay saved landmark sessions from a recorded HandTrack bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from handtrack.io import SessionLoader
from handtrack.tracker import get_hand_connections


def _draw_landmarks(canvas: np.ndarray, landmarks: np.ndarray) -> None:
    height, width = canvas.shape[:2]
    points: list[tuple[int, int]] = []
    for x_coord, y_coord, _z_coord in landmarks:
        x_pixel = int(np.clip(x_coord, 0.0, 1.0) * (width - 1))
        y_pixel = int(np.clip(y_coord, 0.0, 1.0) * (height - 1))
        points.append((x_pixel, y_pixel))

    for start_idx, end_idx in get_hand_connections():
        cv2.line(canvas, points[start_idx], points[end_idx], (0, 200, 255), 2)

    for point in points:
        cv2.circle(canvas, point, 4, (255, 255, 255), -1)


def _build_canvas(width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (24, 24, 24)
    return canvas


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay a recorded landmarks session as a 2D skeleton animation.",
    )
    parser.add_argument(
        "session",
        help="path to a session folder containing landmarks.npz",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="override playback rate; defaults to the session sampling rate",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="window width in pixels",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="window height in pixels",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="loop playback until Escape is pressed",
    )
    args = parser.parse_args(argv)

    session_dir = Path(args.session)
    if not session_dir.exists():
        print(f"[handtracker] Session folder not found: {session_dir}")
        return 1

    loader = SessionLoader(str(session_dir), label="")
    landmarks, sampling_rate, _time_vector = loader.load_landmarks()
    if landmarks is None or sampling_rate is None:
        print(f"[handtracker] No landmarks session found in: {session_dir}")
        return 1

    playback_fps = args.fps or sampling_rate
    frame_delay_ms = max(1, int(round(1000.0 / max(playback_fps, 1e-6))))
    landmarks = np.asarray(landmarks, dtype=np.float32)

    window_name = "HandTrack Replay"
    try:
        while True:
            for frame_index, frame_landmarks in enumerate(landmarks, start=1):
                canvas = _build_canvas(args.width, args.height)
                _draw_landmarks(canvas, frame_landmarks)
                cv2.putText(
                    canvas,
                    f"Frame {frame_index}/{len(landmarks)} | {playback_fps:.1f} FPS",
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    canvas,
                    "Press ESC to close",
                    (16, args.height - 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                )
                cv2.imshow(window_name, canvas)
                key = cv2.waitKey(frame_delay_ms) & 0xFF
                if key == 27:
                    return 0

            if not args.loop:
                return 0
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
