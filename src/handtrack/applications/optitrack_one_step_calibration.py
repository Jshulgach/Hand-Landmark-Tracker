"""OptiTrack entrypoint for one-step synchronized ChArUco calibration."""

from .one_step_calibration_runtime import main as _run_calibration


def main() -> int:
    _run_calibration("optitrack")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
