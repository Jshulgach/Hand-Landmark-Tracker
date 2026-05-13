"""Inspect saved calibration artifacts for a selected camera backend."""

from __future__ import annotations

import argparse
import importlib
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

_CONFIG_MODULES = {
    "optitrack": "unity_hand_tracking.optitrack_cam_py.config",
    "webcam": "unity_hand_tracking.webcam.config",
}


def _load_backend_config(selected_backend: str):
    module_name = _CONFIG_MODULES[selected_backend]
    return importlib.import_module(module_name)


def _format_scalar(value) -> str:
    if value is None:
        return "<missing>"
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return str(value.item())
        return np.array2string(value, separator=", ")
    if isinstance(value, (list, tuple)):
        return str(list(value))
    return str(value)


def _print_row(label: str, value: str) -> None:
    print(f"{label:<18} {value}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect the saved calibration file for a HandTrack backend.",
    )
    parser.add_argument(
        "--backend",
        choices=("optitrack", "webcam"),
        required=True,
        help="camera backend whose calibration artifact should be inspected",
    )
    args = parser.parse_args(argv)

    cfg = _load_backend_config(args.backend)
    calibration_path = Path(cfg.CALIBRATION_FILE)

    print(f"[handtracker] Calibration backend: {args.backend}")
    _print_row("file", str(calibration_path))

    if not calibration_path.exists():
        _print_row("status", "missing")
        return 1

    modified_at = datetime.fromtimestamp(calibration_path.stat().st_mtime)
    _print_row("status", "found")
    _print_row("modified", modified_at.isoformat(timespec="seconds"))

    try:
        data = np.load(calibration_path, allow_pickle=True)
    except Exception as exc:
        _print_row("status", f"unreadable ({exc})")
        return 1

    keys = set(data.files)
    num_cameras = data["num_cameras"].item() if "num_cameras" in keys else None
    img_size = data["img_size"] if "img_size" in keys else None
    num_captures = data["num_captures"] if "num_captures" in keys else None
    camera_ids = data["camera_ids"] if "camera_ids" in keys else None
    checkerboard_size = (
        data["checkerboard_size"] if "checkerboard_size" in keys else None
    )
    square_size = data["square_size"] if "square_size" in keys else None

    matrix_count = len([key for key in keys if key.startswith("camera_matrix_")])
    distortion_count = len([key for key in keys if key.startswith("dist_coeffs_")])
    rotation_count = len([key for key in keys if key.startswith("R_")])
    translation_count = len([key for key in keys if key.startswith("T_")])

    _print_row("num_cameras", _format_scalar(num_cameras))
    _print_row("img_size", _format_scalar(img_size))
    _print_row("captures", _format_scalar(num_captures))
    _print_row("camera_ids", _format_scalar(camera_ids))
    _print_row("board_size", _format_scalar(checkerboard_size))
    _print_row("square_size", _format_scalar(square_size))
    _print_row("intrinsics", str(matrix_count))
    _print_row("distortion", str(distortion_count))
    _print_row("rotations", str(rotation_count))
    _print_row("translations", str(translation_count))

    expected = int(num_cameras) if num_cameras is not None else None
    if expected is not None:
        complete = all(
            count == expected
            for count in (
                matrix_count,
                distortion_count,
                rotation_count,
                translation_count,
            )
        )
        _print_row("complete", "yes" if complete else "no")
        return 0 if complete else 1

    _print_row("complete", "unknown")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
