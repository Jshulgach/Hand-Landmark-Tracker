"""Record caller-owned capture through the public MAVIS frame API."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from handtrack.processing._joint_angles import ANGLE_TRIPLES
from mavis_track import (
    BODY_LANDMARK_NAMES,
    QUALITY_REGION_NAMES,
    HandBodyTracker,
    HandTracker,
)
from mavis_track._recording import SessionWriter


def _normalize_source(value: str):
    return int(value) if value.isdigit() else value


def _draw_overlay(frame, result) -> None:
    from handtrack.tracker import get_hand_connections
    for hand in result.hands:
        points = [tuple((np.clip(point[:2], 0, 1) *
                         [frame.shape[1] - 1, frame.shape[0] - 1]).astype(int))
                  for point in hand.landmarks]
        for a, b in get_hand_connections():
            cv2.line(frame, points[a], points[b], (0, 200, 255), 2)
        for point in points:
            cv2.circle(frame, point, 3, (255, 255, 255), -1)


def _observation(result, hands, names, slots, combined):
    shape = (hands, 21, 3)
    raw, filtered, world = (np.full(shape, np.nan, np.float32) for _ in range(3))
    valid = np.zeros(hands, bool)
    ids = np.full(hands, -1, np.int64)
    labels = np.full(hands, "Unknown", dtype="U7")
    angles = np.full((hands, len(names)), np.nan, np.float32)
    spaces = np.full(hands, "absent", dtype="U20")
    present = {hand.track_id for hand in result.hands}
    slots = {key: slot for key, slot in slots.items() if key in present}
    for hand in result.hands:
        if hand.track_id not in slots:
            slots[hand.track_id] = next(i for i in range(hands) if i not in slots.values())
        slot = slots[hand.track_id]
        raw[slot], filtered[slot] = hand.raw_landmarks, hand.landmarks
        if hand.world_landmarks is not None:
            world[slot] = hand.world_landmarks
        valid[slot], ids[slot], labels[slot] = True, hand.track_id, hand.handedness
        angles[slot] = [hand.angles[name] for name in names]
        spaces[slot] = hand.angle_space
    observation = {"raw_landmarks": raw, "landmarks": filtered, "world_landmarks": world,
                       "valid": valid, "track_ids": ids, "handedness": labels, "angle_values": angles,
                       "angle_space": spaces, "time_vector": result.timestamp,
                       "image_sizes": np.asarray(result.image_size, dtype=np.int64)}
    if combined:
        body = result.body
        observation.update(
            body_landmarks=body.landmarks if body is not None else np.full((33, 3), np.nan, np.float32),
            body_world_landmarks=(body.world_landmarks if body is not None and body.world_landmarks is not None
                                  else np.full((33, 3), np.nan, np.float32)),
            body_visibility=body.visibility if body is not None else np.full(33, np.nan, np.float32),
            body_valid=body is not None,
            quality_flags=np.array([int(getattr(result.quality, region)) for region in QUALITY_REGION_NAMES],
                                   dtype=np.uint16))
    return observation, slots


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("hands", "hand-body"), default="hands")
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    parser.add_argument("--max-gap", type=float, default=0.5, help="tracking gap threshold in seconds")
    parser.add_argument("--chunk-frames", type=int, default=4096,
                        help="maximum buffered frames, 1-16384 (default 4096)")
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
    if not 1 <= args.chunk_frames <= 16384:
        parser.error("--chunk-frames must be between 1 and 16384")
    for name, value in (("confidence", args.confidence), ("visibility-threshold", args.visibility_threshold)):
        if not np.isfinite(value) or not 0 <= value <= 1:
            parser.error(f"--{name} must be between 0 and 1")
    if not np.isfinite(args.max_gap) or args.max_gap <= 0:
        parser.error("--max-gap must be positive finite seconds")
    combined = args.mode == "hand-body"
    if combined:
        args.max_hands = 2
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
    recorder = None
    code, error = 0, None
    try:
        with ExitStack() as resources:
            cap = cv2.VideoCapture(source)
            resources.callback(cap.release)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open source {source!r}")
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if not np.isfinite(fps) or fps <= 0:
                fps = 30.0
            options = {"confidence": args.confidence, "mirrored": args.flip_frame,
                           "source_id": str(source), "smoothing": not args.no_kalman}
            tracker = (HandBodyTracker(**options, visibility_threshold=args.visibility_threshold,
                                       max_gap=args.max_gap) if combined else
                       HandTracker(**options, max_hands=args.max_hands))
            resources.callback(tracker.close)
            session_dir.mkdir(parents=True, exist_ok=False)
            names = tuple(ANGLE_TRIPLES)
            schema = 2 if combined else 1
            clock = "video_pts" if video else "capture_relative"
            metadata = {"schema_version": schema, "sampling_rate": fps, "angle_names": np.asarray(names),
                            "coordinate_space": "image_normalized", "world_coordinate_space": "hand_world_meters",
                            "timestamp_clock": clock, "source": str(source), "apply_kalman": not args.no_kalman}
            manifest = {"schema_version": schema, "created_at": datetime.now().isoformat(timespec="seconds"),
                            "source": str(source), "max_hands": args.max_hands, "coordinate_space": "image_normalized",
                            "timestamp_clock": clock, "apply_kalman": not args.no_kalman, "save_video": args.save_video}
            if combined:
                from mavis_track import QualityFlag
                metadata.update(tracking_mode="hand-body", body_landmark_names=np.asarray(BODY_LANDMARK_NAMES),
                                body_world_coordinate_space="body_world_meters",
                                quality_region_names=np.asarray(QUALITY_REGION_NAMES),
                                quality_visibility_threshold=args.visibility_threshold, quality_max_gap=args.max_gap)
                manifest.update(tracking_mode="hand-body", mirrored=args.flip_frame,
                                quality_region_names=list(QUALITY_REGION_NAMES),
                                quality_flag_definitions={"OK": 0, **{flag.name: int(flag) for flag in QualityFlag}},
                                quality_visibility_threshold=args.visibility_threshold, quality_max_gap=args.max_gap,
                                body_world_coordinate_space="body_world_meters", smoothing_scope="hands_only")
            recorder = SessionWriter(session_dir, metadata, manifest, chunk_frames=args.chunk_frames, video=video)
            slots = {}
            writer = None
            started = time.perf_counter()
            while limit is None or recorder.count < limit:
                ok, frame = cap.read()
                if not ok or frame is None:
                    if not video:
                        raise RuntimeError("Camera disconnected or failed to supply a frame")
                    break
                captured = time.perf_counter()
                if args.flip_frame:
                    frame = cv2.flip(frame, 1)
                if video:
                    pts = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                    previous = recorder.last_time
                    stamp = pts if np.isfinite(pts) and pts >= 0 and (previous is None or pts > previous) else recorder.count / fps
                    if previous is not None and stamp <= previous:
                        stamp = previous + 1 / fps
                else:
                    stamp = captured - started
                result = tracker.process(frame, timestamp=stamp)
                observation, slots = _observation(result, args.max_hands, names, slots, combined)
                recorder.append(observation)
                if args.verbose:
                    print(f"Frame {recorder.count}: {len(result.hands)} hand(s), {stamp:.3f} s")
                if args.save_video:
                    if writer is None:
                        writer = cv2.VideoWriter(str(session_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 fps, (frame.shape[1], frame.shape[0]))
                        resources.callback(writer.release)
                        if not writer.isOpened():
                            raise RuntimeError("Cannot create annotated video with this codec")
                    annotated = frame.copy()
                    _draw_overlay(annotated, result)
                    if combined and result.body is not None:
                        from ._replay import _draw_body
                        _draw_body(annotated, result.body.landmarks, result.body.visibility, args.visibility_threshold)
                    writer.write(annotated)
            if not recorder.count:
                raise RuntimeError("Source supplied no readable frames; no session bundle was saved")
    except KeyboardInterrupt:
        code, error = 130, "Recording interrupted by operator"
    except (ValueError, RuntimeError, OSError, cv2.error) as exc:
        code, error = 1, str(exc)
    finally:
        if recorder is not None:
            try:
                recorder.finish(complete=code == 0, error=error)
            except (ValueError, OSError) as exc:
                code, error = 1, f"Cannot finalize recording: {exc}"
    if error:
        print(f"[mavis] {error}")
    if recorder is not None and recorder.count:
        print(f"[mavis] Recording directory: {session_dir} ({recorder.count} frames processed)")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
