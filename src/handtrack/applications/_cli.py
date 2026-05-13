"""Unified command-line interface for HandTrack applications."""

from __future__ import annotations

import argparse
import importlib
from typing import Sequence

_BACKEND_ENTRYPOINTS = {
    "optitrack": {
        "gui": "unity_hand_tracking.optitrack_cam_py.mocap_handtrack_gui",
        "calibrate": "unity_hand_tracking.optitrack_cam_py.calibration",
        "cameras": "unity_hand_tracking.optitrack_cam_py.multi_mjpeg",
        "board": "unity_hand_tracking.optitrack_cam_py.generate_charuco_board",
        "test-sender": "unity_hand_tracking.optitrack_cam_py.test_sender",
    },
    "webcam": {
        "gui": "unity_hand_tracking.webcam.mocap_handracker_gui",
        "calibrate": "unity_hand_tracking.webcam.calibration",
        "cameras": "unity_hand_tracking.webcam.multi_webcam",
        "board": "unity_hand_tracking.webcam.generate_charuco_board",
        "test-sender": "unity_hand_tracking.webcam.test_sender",
    },
}


def _optitrack_sdk_available() -> bool:
    try:
        config = importlib.import_module("unity_hand_tracking.optitrack_cam_py.config")
    except Exception:
        return False

    return getattr(config, "optitrack_cam", None) is not None


def _resolve_backend(requested_backend: str) -> str:
    if requested_backend != "auto":
        return requested_backend

    if _optitrack_sdk_available():
        return "optitrack"

    return "webcam"


def _run_module_entrypoint(module_name: str) -> int:
    module = importlib.import_module(module_name)
    main = getattr(module, "main", None)
    if main is None:
        raise RuntimeError(
            f"Module '{module_name}' does not expose a main() entrypoint."
        )

    result = main()
    return 0 if result is None else int(result)


def _run_doctor(selected_backend: str) -> int:
    from ._doctor import main as doctor_main

    argv = ["--backend", selected_backend]
    return int(doctor_main(argv))


def _run_inspect_calibration(selected_backend: str) -> int:
    from ._inspect_calibration import main as inspect_main

    argv = ["--backend", selected_backend]
    return int(inspect_main(argv))


def _run_benchmark(selected_backend: str, args: argparse.Namespace) -> int:
    from ._benchmark import main as benchmark_main

    argv = [
        "--backend",
        selected_backend,
        "--frames",
        str(args.frames),
        "--warmup",
        str(args.warmup),
        "--min-det-conf",
        str(args.min_det_conf),
        "--min-track-conf",
        str(args.min_track_conf),
    ]
    return int(benchmark_main(argv))


def _run_record(args: argparse.Namespace) -> int:
    from ._record import main as record_main

    argv = [
        "--source",
        str(args.source),
        "--output-dir",
        args.output_dir,
        "--max-hands",
        str(args.max_hands),
        "--confidence",
        str(args.confidence),
    ]
    if args.session_name is not None:
        argv.extend(["--session-name", args.session_name])
    if args.frames is not None:
        argv.extend(["--frames", str(args.frames)])
    if args.save_video:
        argv.append("--save-video")
    if args.flip_frame:
        argv.append("--flip-frame")
    if args.no_kalman:
        argv.append("--no-kalman")
    if args.verbose:
        argv.append("--verbose")
    return int(record_main(argv))


def _run_replay(args: argparse.Namespace) -> int:
    from ._replay import main as replay_main

    argv = [
        args.session,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
    ]
    if args.fps is not None:
        argv.extend(["--fps", str(args.fps)])
    if args.loop:
        argv.append("--loop")
    return int(replay_main(argv))


def _run_export(args: argparse.Namespace) -> int:
    from ._export import main as export_main

    argv = [args.session, "--variant", args.variant]
    if args.output_dir is not None:
        argv.extend(["--output-dir", args.output_dir])
    if args.skip_angles:
        argv.append("--skip-angles")
    return int(export_main(argv))


def _add_backend_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=("auto", "optitrack", "webcam"),
        default="auto",
        help=(
            "camera backend to use; defaults to auto, which prefers OptiTrack "
            "when its SDK is available and otherwise falls back to webcam"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="handtracker",
        description="Launch HandTrack applications without remembering nested module paths.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    gui_parser = subparsers.add_parser("gui", help="launch the live hand-tracking GUI")
    _add_backend_argument(gui_parser)

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="launch the multi-camera ChArUco calibration workflow",
    )
    _add_backend_argument(calibrate_parser)

    cameras_parser = subparsers.add_parser(
        "cameras",
        help="preview live feeds from the selected camera backend",
    )
    _add_backend_argument(cameras_parser)

    board_parser = subparsers.add_parser(
        "board",
        help="generate the ChArUco board used for camera calibration",
    )
    _add_backend_argument(board_parser)

    sender_parser = subparsers.add_parser(
        "test-sender",
        help="send sample landmark and joint-angle packets for downstream testing",
    )
    _add_backend_argument(sender_parser)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="measure backend timing and throughput over a fixed number of frames",
    )
    _add_backend_argument(benchmark_parser)
    benchmark_parser.add_argument(
        "--frames",
        type=int,
        default=60,
        help="number of measured frames to process after warmup",
    )
    benchmark_parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="number of warmup frames to discard before measurement",
    )
    benchmark_parser.add_argument(
        "--min-det-conf",
        type=float,
        default=0.6,
        help="minimum MediaPipe detection confidence",
    )
    benchmark_parser.add_argument(
        "--min-track-conf",
        type=float,
        default=0.6,
        help="minimum MediaPipe tracking confidence",
    )

    inspect_parser = subparsers.add_parser(
        "inspect-calibration",
        help="summarize the saved calibration file for the selected backend",
    )
    _add_backend_argument(inspect_parser)

    record_parser = subparsers.add_parser(
        "record",
        help="record a webcam or video source into a session bundle",
    )
    record_parser.add_argument(
        "--source",
        default="0",
        help="camera index or video path (default: webcam 0)",
    )
    record_parser.add_argument(
        "--output-dir",
        default="recordings",
        help="parent directory where the session folder should be created",
    )
    record_parser.add_argument(
        "--session-name",
        default=None,
        help="optional session folder name; defaults to a timestamped name",
    )
    record_parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="maximum number of frames to record; defaults to all frames for video or 300 for webcam",
    )
    record_parser.add_argument(
        "--save-video",
        action="store_true",
        help="also save an annotated MP4 alongside the data bundle",
    )
    record_parser.add_argument(
        "--flip-frame",
        action="store_true",
        help="flip incoming frames horizontally before processing",
    )
    record_parser.add_argument(
        "--max-hands",
        type=int,
        default=1,
        help="maximum number of hands to detect",
    )
    record_parser.add_argument(
        "--confidence",
        type=float,
        default=0.8,
        help="minimum MediaPipe detection confidence",
    )
    record_parser.add_argument(
        "--no-kalman",
        action="store_true",
        help="disable Kalman smoothing and save raw landmarks only",
    )
    record_parser.add_argument(
        "--verbose",
        action="store_true",
        help="print additional tracker information while recording",
    )

    replay_parser = subparsers.add_parser(
        "replay",
        help="replay a recorded landmarks session",
    )
    replay_parser.add_argument(
        "session",
        help="path to a session folder containing landmarks.npz",
    )
    replay_parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="override playback rate; defaults to the session sampling rate",
    )
    replay_parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="window width in pixels",
    )
    replay_parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="window height in pixels",
    )
    replay_parser.add_argument(
        "--loop",
        action="store_true",
        help="loop playback until Escape is pressed",
    )

    export_parser = subparsers.add_parser(
        "export",
        help="export a recorded session bundle into flat CSV files",
    )
    export_parser.add_argument(
        "session",
        help="path to a session folder containing landmarks.npz",
    )
    export_parser.add_argument(
        "--output-dir",
        default=None,
        help="directory for exported files; defaults to <session>/exports",
    )
    export_parser.add_argument(
        "--variant",
        choices=("landmarks", "raw_landmarks"),
        default="landmarks",
        help="which landmark array to export",
    )
    export_parser.add_argument(
        "--skip-angles",
        action="store_true",
        help="skip exporting angle_values to CSV",
    )

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="check runtime dependencies and backend availability",
    )
    _add_backend_argument(doctor_parser)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    selected_backend = None
    if hasattr(args, "backend"):
        selected_backend = _resolve_backend(args.backend)
    if getattr(args, "backend", None) == "auto":
        print(f"[handtracker] Selected backend: {selected_backend}")

    if args.command == "doctor":
        return _run_doctor(selected_backend)

    if args.command == "inspect-calibration":
        return _run_inspect_calibration(selected_backend)

    if args.command == "benchmark":
        return _run_benchmark(selected_backend, args)

    if args.command == "record":
        return _run_record(args)

    if args.command == "replay":
        return _run_replay(args)

    if args.command == "export":
        return _run_export(args)

    module_name = _BACKEND_ENTRYPOINTS[selected_backend][args.command]
    return _run_module_entrypoint(module_name)


if __name__ == "__main__":
    raise SystemExit(main())
