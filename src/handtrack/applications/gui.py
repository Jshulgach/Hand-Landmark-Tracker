"""Generic GUI entrypoint that dispatches to the selected backend."""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Sequence

from ._backend_selection import resolve_backend

_BACKEND_MODULES = {
    "webcam": "handtrack.applications.mediapipe_gui",
    "optitrack": "handtrack.applications.optitrack_gui",
}


def _run_module_entrypoint(module_name: str, passthrough_args: list[str]) -> int:
    module = importlib.import_module(module_name)
    main = getattr(module, "main", None)
    if main is None:
        raise RuntimeError(
            f"Module '{module_name}' does not expose a main() entrypoint."
        )

    previous_argv = sys.argv
    sys.argv = [previous_argv[0], *passthrough_args]
    try:
        result = main()
    finally:
        sys.argv = previous_argv

    return 0 if result is None else int(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch the tracking GUI for the selected backend."
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
    args, passthrough = parser.parse_known_args(argv)

    selected_backend = resolve_backend(args.backend)
    advanced_hands = "--advanced-hands" in passthrough
    if advanced_hands:
        passthrough.remove("--advanced-hands")
    module_name = (
        "handtrack.applications.webcam_gui"
        if selected_backend == "webcam" and advanced_hands
        else _BACKEND_MODULES[selected_backend]
    )
    return _run_module_entrypoint(module_name, passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
