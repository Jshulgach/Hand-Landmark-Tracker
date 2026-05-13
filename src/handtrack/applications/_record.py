"""Record landmarks and angles from a webcam or video source into a session bundle."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

import cv2
import mediapipe as mp
import numpy as np

from handtrack.tracker import HandTracker


def _normalize_source(value: str):
    return int(value) if value.isdigit() else value


def _default_session_name() -> str:
    return datetime.now().strftime("session_%Y%m%d_%H%M%S")


def _draw_overlay(frame, landmarks, filtered_landmarks) -> None:
    if landmarks is not None:
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
        )

    if filtered_landmarks is not None:
        for x, y, _ in filtered_landmarks:
            cv2.circle(
                frame,
                (int(x * frame.shape[1]), int(y * frame.shape[0])),
                4,
                (255, 255, 255),
                -1,
            )


def _write_angles_csv(
    path: Path, angle_names: list[str], timestamps: list[float], values: np.ndarray
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp"] + angle_names)
        for timestamp, row in zip(timestamps, values, strict=True):
            writer.writerow([f"{timestamp:.6f}"] + [f"{value:.6f}" for value in row])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record landmarks and joint angles from a webcam or video source.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="camera index or video path (default: webcam 0)",
    )
    parser.add_argument(
        "--output-dir",
        default="recordings",
        help="parent directory where the session folder should be created",
    )
    parser.add_argument(
        "--session-name",
        default=None,
        help="optional session folder name; defaults to a timestamped name",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="maximum number of frames to record; defaults to all frames for video or 300 for webcam",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="also save an annotated MP4 alongside the data bundle",
    )
    parser.add_argument(
        "--flip-frame",
        action="store_true",
        help="flip incoming frames horizontally before processing",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=1,
        help="maximum number of hands to detect",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.8,
        help="minimum MediaPipe detection confidence",
    )
    parser.add_argument(
        "--no-kalman",
        action="store_true",
        help="disable Kalman smoothing and save raw landmarks only",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print additional tracker information while recording",
    )
    args = parser.parse_args(argv)

    source = _normalize_source(str(args.source))
    session_name = args.session_name or _default_session_name()
    session_dir = Path(args.output_dir) / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    tracker = HandTracker(
        source=source,
        max_hands=args.max_hands,
        confidence=args.confidence,
        apply_kalman=not args.no_kalman,
        verbose=args.verbose,
    )

    fps = tracker.cap.get(cv2.CAP_PROP_FPS) or 30.0
    default_frame_limit = None if tracker.mode == "video" else 300
    frame_limit = args.frames if args.frames is not None else default_frame_limit

    writer = None
    raw_landmarks_log: list[np.ndarray] = []
    filtered_landmarks_log: list[np.ndarray] = []
    timestamps: list[float] = []
    angle_rows: list[list[float]] = []
    angle_names: list[str] | None = None

    try:
        frame_index = 0
        while tracker.cap.isOpened():
            frame = tracker.get_image(flip_frame=args.flip_frame)
            if frame is None:
                break

            landmarks, filtered_landmarks, angles, _results = tracker._process_frame(
                frame
            )

            if landmarks is not None:
                raw_landmarks = np.array(
                    [[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
                    dtype=np.float32,
                )
            else:
                raw_landmarks = np.zeros((21, 3), dtype=np.float32)

            raw_landmarks_log.append(raw_landmarks)
            filtered_landmarks_log.append(
                np.array(filtered_landmarks, dtype=np.float32)
            )

            timestamp = frame_index / fps
            timestamps.append(timestamp)

            if angle_names is None:
                angle_names = list(angles.keys())
            angle_rows.append([float(angles[name]) for name in angle_names])

            if args.save_video:
                if writer is None:
                    height, width = frame.shape[:2]
                    video_path = session_dir / "annotated.mp4"
                    writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                annotated = frame.copy()
                _draw_overlay(annotated, landmarks, filtered_landmarks)
                writer.write(annotated)

            frame_index += 1
            if frame_limit is not None and frame_index >= frame_limit:
                break

        raw_landmarks_np = np.asarray(raw_landmarks_log, dtype=np.float32)
        filtered_landmarks_np = np.asarray(filtered_landmarks_log, dtype=np.float32)
        angle_values_np = np.asarray(angle_rows, dtype=np.float32)
        time_vector_np = np.asarray(timestamps, dtype=np.float64)

        np.savez(
            session_dir / "landmarks.npz",
            raw_landmarks=raw_landmarks_np,
            landmarks=filtered_landmarks_np,
            angle_names=np.asarray(angle_names or [], dtype=str),
            angle_values=angle_values_np,
            sampling_rate=fps,
            total_frames=filtered_landmarks_np.shape[0],
            time_vector=time_vector_np,
            source=str(source),
            mode=tracker.mode,
            apply_kalman=not args.no_kalman,
        )

        if angle_names:
            _write_angles_csv(
                session_dir / "angles.csv",
                angle_names,
                timestamps,
                angle_values_np,
            )

        manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source": str(source),
            "mode": tracker.mode,
            "sampling_rate": fps,
            "frame_count": int(filtered_landmarks_np.shape[0]),
            "apply_kalman": not args.no_kalman,
            "save_video": args.save_video,
            "session_dir": str(session_dir),
        }
        (session_dir / "session.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        print(f"[handtracker] Recorded session: {session_dir}")
        print(f"[handtracker] Frames saved: {filtered_landmarks_np.shape[0]}")
        return 0
    finally:
        tracker.cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
