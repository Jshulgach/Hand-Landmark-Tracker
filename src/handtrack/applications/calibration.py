"""Generic calibration entrypoint that dispatches to the selected backend."""

from __future__ import annotations

import argparse
import importlib
from typing import Sequence

from ._backend_selection import resolve_backend

_BACKEND_MODULES = {
    "webcam": "handtrack.applications.webcam_calibration",
    "optitrack": "handtrack.applications.optitrack_calibration",
}


def _run_module_entrypoint(module_name: str) -> int:
    module = importlib.import_module(module_name)
    main = getattr(module, "main", None)
    if main is None:
        raise RuntimeError(
            f"Module '{module_name}' does not expose a main() entrypoint."
        )

    result = main()
    return 0 if result is None else int(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run ChArUco calibration for the selected backend."
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "optitrack", "webcam"),
        default=None,
        help=(
            "camera backend to use; defaults to HANDTRACK_BACKEND if set, "
            "otherwise auto"
        ),
    )
    args = parser.parse_args(argv)

    selected_backend = resolve_backend(args.backend)
    module_name = _BACKEND_MODULES[selected_backend]
    return _run_module_entrypoint(module_name)


if __name__ == "__main__":
    raise SystemExit(main())
