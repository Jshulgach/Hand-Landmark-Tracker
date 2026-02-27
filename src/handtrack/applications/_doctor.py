"""Runtime diagnostics for HandTrack deployments."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import Iterable


def _check_import(module_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        return True, f"ok (version={version})"
    except Exception as exc:
        return False, f"missing ({exc})"


def _print_table(rows: Iterable[tuple[str, str]]) -> None:
    rows = list(rows)
    width = max((len(k) for k, _ in rows), default=0)
    for key, value in rows:
        print(f"{key.ljust(width)} : {value}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check HandTrack runtime dependencies and optional OptiTrack SDK wiring.",
    )
    parser.add_argument(
        "--optitrack",
        action="store_true",
        help="also validate OptiTrack camera SDK module availability",
    )
    args = parser.parse_args()

    required_modules = [
        "numpy",
        "cv2",
        "mediapipe",
        "scipy",
    ]
    optional_modules = [
        "pylsl",
        "PyQt5",
        "pyqtgraph",
    ]

    rows: list[tuple[str, str]] = [
        ("python", sys.version.split()[0]),
        ("executable", sys.executable),
    ]

    ok = True

    for name in required_modules:
        passed, detail = _check_import(name)
        rows.append((f"required:{name}", detail))
        ok = ok and passed

    for name in optional_modules:
        passed, detail = _check_import(name)
        rows.append((f"optional:{name}", detail))
        if not passed:
            rows.append(
                (
                    f"hint:{name}",
                    "install optional dependency if this feature is needed",
                )
            )

    if args.optitrack:
        sdk_env = os.environ.get("OPTITRACK_CAM_PY_PATH", "")
        rows.append(("env:OPTITRACK_CAM_PY_PATH", sdk_env or "<not-set>"))
        passed, detail = _check_import("unity_hand_tracking.optitrack_cam_py.config")
        rows.append(("optitrack:config", detail))

        try:
            cfg = importlib.import_module("unity_hand_tracking.optitrack_cam_py.config")
            sdk_available = getattr(cfg, "optitrack_cam", None) is not None
            rows.append(("optitrack:sdk", "ok" if sdk_available else "missing"))
            ok = ok and sdk_available
            if not sdk_available:
                rows.append(
                    (
                        "hint:optitrack",
                        "place SDK in optitrack_cam_py/Release or set OPTITRACK_CAM_PY_PATH",
                    )
                )
        except Exception as exc:
            rows.append(("optitrack:error", str(exc)))
            ok = False

    _print_table(rows)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
