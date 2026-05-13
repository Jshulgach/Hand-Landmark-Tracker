"""Benchmark backend tracking performance from the command line."""

from __future__ import annotations

import argparse
import importlib
import statistics
import time
from typing import Sequence

_TRACKER_MODULES = {
    "optitrack": "unity_hand_tracking.optitrack_cam_py.mocap_tracker",
    "webcam": "unity_hand_tracking.webcam.mocap_tracker",
}


def _load_tracker_class(selected_backend: str):
    module = importlib.import_module(_TRACKER_MODULES[selected_backend])
    return getattr(module, "MultiCameraTracker")


def _summarize(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}

    return {
        "mean": statistics.fmean(samples),
        "min": min(samples),
        "max": max(samples),
    }


def _print_metric(name: str, values: dict[str, float], unit: str) -> None:
    print(
        f"{name:<16} mean={values['mean']:.2f}{unit} "
        f"min={values['min']:.2f}{unit} max={values['max']:.2f}{unit}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the selected HandTrack backend over a fixed number of frames.",
    )
    parser.add_argument(
        "--backend",
        choices=("optitrack", "webcam"),
        required=True,
        help="backend to benchmark",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=60,
        help="number of measured frames to process after warmup",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="number of warmup frames to discard before measurement",
    )
    parser.add_argument(
        "--min-det-conf",
        type=float,
        default=0.6,
        help="minimum MediaPipe detection confidence",
    )
    parser.add_argument(
        "--min-track-conf",
        type=float,
        default=0.6,
        help="minimum MediaPipe tracking confidence",
    )
    args = parser.parse_args(argv)

    tracker_cls = _load_tracker_class(args.backend)
    tracker = tracker_cls()

    try:
        if not tracker.initialize_cameras(
            min_det_conf=args.min_det_conf,
            min_track_conf=args.min_track_conf,
        ):
            print("[handtracker] Failed to initialize cameras for benchmark.")
            return 1

        total_frames = args.warmup + args.frames
        capture_ms: list[float] = []
        detection_ms: list[float] = []
        triangulation_ms: list[float] = []
        fps_values: list[float] = []
        hands_seen: list[float] = []
        end_to_end_ms: list[float] = []

        for frame_index in range(total_frames):
            t0 = time.perf_counter()
            _frames, triangulated_hands, _all_results, _valid = tracker.process_frame()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            stats = tracker.get_fps_stats()

            if frame_index < args.warmup:
                continue

            capture_ms.append(stats["capture_ms"])
            detection_ms.append(stats["detection_ms"])
            triangulation_ms.append(stats["triangulation_ms"])
            fps_values.append(stats["fps"])
            hands_seen.append(float(len(triangulated_hands)))
            end_to_end_ms.append(elapsed_ms)

        print(f"[handtracker] Benchmark backend: {args.backend}")
        print(f"[handtracker] Measured frames: {args.frames}")
        print(f"[handtracker] Connected cameras: {tracker.num_cameras}")
        _print_metric("fps", _summarize(fps_values), "")
        _print_metric("capture", _summarize(capture_ms), "ms")
        _print_metric("detection", _summarize(detection_ms), "ms")
        _print_metric("triangulation", _summarize(triangulation_ms), "ms")
        _print_metric("end_to_end", _summarize(end_to_end_ms), "ms")
        _print_metric("hands_seen", _summarize(hands_seen), "")
        return 0
    finally:
        tracker.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
