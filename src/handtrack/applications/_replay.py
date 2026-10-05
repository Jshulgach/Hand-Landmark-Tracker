"""Replay validated landmark sessions using their recorded timing."""

import argparse
import time
import cv2
import numpy as np
from mavis_track import load_session
from handtrack.tracker import get_hand_connections


def _draw_landmarks(canvas, landmarks, valid=None):
    batch = np.asarray(landmarks)
    if batch.ndim == 2:
        batch = batch[None]
    if batch.ndim != 3 or batch.shape[1:] != (21, 3):
        raise ValueError("Expected (hands, 21, 3) landmarks")
    if valid is None:
        valid = np.isfinite(batch).all(axis=(1, 2)) & np.any(batch != 0, axis=(1, 2))
    height, width = canvas.shape[:2]
    for hand, present in zip(batch, valid):
        if not present:
            continue
        points = [tuple((np.clip(p[:2], 0, 1) * [width - 1, height - 1]).astype(int))
                  if np.isfinite(p).all() else None for p in hand]
        for a, b in get_hand_connections():
            if points[a] is not None and points[b] is not None:
                cv2.line(canvas, points[a], points[b], (0, 200, 255), 2)
        for point in points:
            if point is not None:
                cv2.circle(canvas, point, 4, (255, 255, 255), -1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--no-display", action="store_true", help="validate and render without a window")
    args = parser.parse_args(argv)
    if min(args.width, args.height) <= 0 or (args.fps is not None and (not np.isfinite(args.fps) or args.fps <= 0)):
        parser.error("dimensions and playback rate must be positive")
    try:
        session = load_session(args.session)
        if session.coordinate_space != "image_normalized":
            raise ValueError("2D replay requires image_normalized coordinates")
        if not len(session.landmarks):
            raise ValueError("Session has no frames")
    except (OSError, ValueError) as exc:
        print(f"[mavis] Cannot replay session: {exc}")
        return 1
    try:
        while True:
            started = time.perf_counter()
            for index, (points, valid) in enumerate(zip(session.landmarks, session.valid)):
                canvas = np.full((args.height, args.width, 3), 24, np.uint8)
                _draw_landmarks(canvas, points, valid)
                if args.no_display:
                    continue
                cv2.imshow("MAVIS Replay", canvas)
                target = (index + 1) / args.fps if args.fps else (
                    session.time_vector[index + 1] - session.time_vector[0]
                    if index + 1 < len(session.time_vector) else
                    session.time_vector[index] - session.time_vector[0] + 1 / session.sampling_rate)
                delay = max(1, int((target - (time.perf_counter() - started)) * 1000))
                if cv2.waitKey(delay) & 0xFF == 27:
                    return 0
            if not args.loop or args.no_display:
                return 0
    finally:
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
