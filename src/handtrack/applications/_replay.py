"""Replay validated landmark sessions using their recorded timing."""

import argparse
import time

import cv2
import numpy as np

from handtrack.tracker import get_hand_connections
from mavis_track import iter_session_chunks


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



def _draw_body(canvas, landmarks, visibility, threshold=0.5):
    from mediapipe.python.solutions.pose import POSE_CONNECTIONS
    height, width = canvas.shape[:2]
    points = [tuple((np.clip(point[:2], 0, 1) * [width - 1, height - 1]).astype(int))
              if np.isfinite(point).all() and np.isfinite(score) and score >= threshold else None
              for point, score in zip(landmarks, visibility)]
    for a, b in POSE_CONNECTIONS:
        if points[a] is not None and points[b] is not None:
            cv2.line(canvas, points[a], points[b], (255, 180, 40), 2)
    for point in points:
        if point is not None:
            cv2.circle(canvas, point, 3, (255, 180, 40), -1)

def _frames(path):
    for session in iter_session_chunks(path):
        if session.coordinate_space != "image_normalized":
            raise ValueError("2D replay requires image_normalized coordinates")
        for index, stamp in enumerate(session.time_vector):
            body = session.body_landmarks[index] if session.body_valid is not None and session.body_valid[index] else None
            visibility = session.body_visibility[index] if body is not None else None
            yield (session.landmarks[index], session.valid[index], stamp, body, visibility,
                   session.quality_visibility_threshold, session.sampling_rate)


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
        while True:
            frames = _frames(args.session)
            current = next(frames, None)
            if current is None:
                raise ValueError("Session has no frames")
            origin = current[2]
            started = time.perf_counter()
            index = 0
            while current is not None:
                following = next(frames, None)
                points, valid, stamp, body, visibility, threshold, rate = current
                canvas = np.full((args.height, args.width, 3), 24, np.uint8)
                _draw_landmarks(canvas, points, valid)
                if body is not None:
                    _draw_body(canvas, body, visibility, threshold)
                if not args.no_display:
                    cv2.imshow("MAVIS Replay", canvas)
                    target = (index + 1) / args.fps if args.fps else (
                        following[2] - origin if following is not None else stamp - origin + 1 / rate)
                    delay = max(1, int((target - (time.perf_counter() - started)) * 1000))
                    if cv2.waitKey(delay) & 0xFF == 27:
                        return 0
                index += 1
                current = following
            if not args.loop or args.no_display:
                return 0
    except (OSError, ValueError) as exc:
        print(f"[mavis] Cannot replay session: {exc}")
        return 1
    finally:
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
