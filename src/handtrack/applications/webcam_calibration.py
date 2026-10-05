"""Webcam entrypoint for shared ChArUco calibration."""

from ._calibration_runtime import main as _run_calibration


def main() -> int:
    return _run_calibration("webcam")


if __name__ == "__main__":
    raise SystemExit(main())
