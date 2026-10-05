"""Record caller-owned capture through the public MAVIS frame API."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Sequence

import cv2
import numpy as np

from mavis_track import HandTracker
from handtrack.processing._joint_angles import ANGLE_TRIPLES


def _normalize_source(value: str):
    return int(value) if value.isdigit() else value


def _draw_overlay(frame, result) -> None:
    from handtrack.tracker import get_hand_connections
    for hand in result.hands:
        points = [(int(x * (frame.shape[1] - 1)), int(y * (frame.shape[0] - 1)))
                  for x, y, _ in hand.landmarks]
        for a, b in get_hand_connections():
            cv2.line(frame, points[a], points[b], (0, 200, 255), 2)
        for point in points:
            cv2.circle(frame, point, 3, (255, 255, 255), -1)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="camera index or video path")
    parser.add_argument("--output-dir", default="recordings")
    parser.add_argument("--session-name", default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--flip-frame", action="store_true")
    parser.add_argument("--max-hands", type=int, choices=(1, 2), default=1)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--no-kalman", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.frames is not None and args.frames <= 0:
        parser.error("--frames must be positive")
    if not np.isfinite(args.confidence) or not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    name = args.session_name or datetime.now().strftime("session_%Y%m%d_%H%M%S_%f")
    if name in (".", "..") or Path(name).name != name or "/" in name or chr(92) in name:
        parser.error("--session-name must be a folder name, without a path")
    source = _normalize_source(str(args.source))
    video = isinstance(source, str)
    limit = args.frames if args.frames is not None else (None if video else 300)
    session_dir = Path(args.output_dir) / name
    if session_dir.exists():
        print(f"[mavis] Session already exists: {session_dir}; choose a new name")
        return 1
    cap = cv2.VideoCapture(source)
    tracker = None
    writer = None
    try:
        if not cap.isOpened():
            print(f"[mavis] Cannot open source {source!r}")
            return 1
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        tracker = HandTracker(max_hands=args.max_hands, confidence=args.confidence,
                              mirrored=args.flip_frame, source_id=str(source),
                              smoothing=not args.no_kalman)
        session_dir.mkdir(parents=True, exist_ok=False)
        raw_log, filtered_log, valid_log, id_log, labels_log, world_log = [], [], [], [], [], []
        times, angle_log, angle_spaces = [], [], []
        image_sizes_log = []
        names = tuple(ANGLE_TRIPLES)
        slots: dict[int, int] = {}
        started = time.perf_counter()
        while limit is None or len(times) < limit:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            captured = time.perf_counter()
            if args.flip_frame:
                frame = cv2.flip(frame, 1)
            if video:
                pts = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                stamp = pts if np.isfinite(pts) and pts >= 0 and (not times or pts > times[-1]) else len(times) / fps
                if times and stamp <= times[-1]:
                    stamp = times[-1] + 1 / fps
            else:
                stamp = captured - started
            result = tracker.process(frame, timestamp=stamp)
            shape = (args.max_hands, 21, 3)
            raw, filtered, world = (np.full(shape, np.nan, np.float32) for _ in range(3))
            valid = np.zeros(args.max_hands, bool)
            ids = np.full(args.max_hands, -1, np.int64)
            labels = np.full(args.max_hands, "Unknown", dtype="U7")
            angles = np.full((args.max_hands, len(names)), np.nan, np.float32)
            angle_space = np.full(args.max_hands, "absent", dtype="U20")
            present = {hand.track_id for hand in result.hands}
            slots = {key: slot for key, slot in slots.items() if key in present}
            for hand in result.hands:
                if hand.track_id not in slots:
                    slots[hand.track_id] = next(i for i in range(args.max_hands) if i not in slots.values())
                slot = slots[hand.track_id]
                raw[slot], filtered[slot] = hand.raw_landmarks, hand.landmarks
                if hand.world_landmarks is not None:
                    world[slot] = hand.world_landmarks
                valid[slot], ids[slot], labels[slot] = True, hand.track_id, hand.handedness
                angles[slot] = [hand.angles[name] for name in names]
                angle_space[slot] = hand.angle_space
            raw_log.append(raw)
            filtered_log.append(filtered)
            world_log.append(world)
            valid_log.append(valid)
            id_log.append(ids)
            labels_log.append(labels)
            times.append(stamp)
            angle_log.append(angles)
            angle_spaces.append(angle_space)
            image_sizes_log.append(result.image_size)
            if args.verbose:
                print(f"Frame {len(times)}: {len(result.hands)} hand(s), {stamp:.3f} s")
            if args.save_video:
                if writer is None:
                    writer = cv2.VideoWriter(str(session_dir / "annotated.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                             (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError("Cannot create annotated video with this codec")
                annotated = frame.copy()
                _draw_overlay(annotated, result)
                writer.write(annotated)
        count = len(times)
        if not count:
            print("[mavis] Source supplied no readable frames; no session bundle was saved")
            return 1
        rate = fps if video or count < 2 else (count - 1) / (times[-1] - times[0])
        np.savez_compressed(session_dir / "landmarks.npz", schema_version=1,
                            landmarks=np.asarray(filtered_log), raw_landmarks=np.asarray(raw_log),
                            world_landmarks=np.asarray(world_log), valid=np.asarray(valid_log),
                            image_sizes=np.asarray(image_sizes_log, dtype=np.int64),
                            track_ids=np.asarray(id_log), handedness=np.asarray(labels_log),
                            angle_names=np.asarray(names), angle_values=np.asarray(angle_log), angle_space=np.asarray(angle_spaces),
                            sampling_rate=rate, time_vector=np.asarray(times), total_frames=count,
                            coordinate_space="image_normalized", world_coordinate_space="hand_world_meters",
                            timestamp_clock="video_pts" if video else "capture_relative",
                            source=str(source), apply_kalman=not args.no_kalman)
        manifest = {"schema_version": 1, "created_at": datetime.now().isoformat(timespec="seconds"),
                    "source": str(source), "frame_count": count, "max_hands": args.max_hands,
                    "sampling_rate": rate, "coordinate_space": "image_normalized",
                    "timestamp_clock": "video_pts" if video else "capture_relative",
                    "apply_kalman": not args.no_kalman, "save_video": args.save_video}
        (session_dir / "session.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[mavis] Recorded {count} frames: {session_dir}")
        return 0
    except (ValueError, RuntimeError, OSError, cv2.error) as exc:
        print(f"[mavis] Recording failed: {exc}")
        return 1
    finally:
        cap.release()
        if tracker is not None:
            tracker.close()
        if writer is not None:
            writer.release()


if __name__ == "__main__":
    raise SystemExit(main())
