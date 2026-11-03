"""Locate optional local OptiTrack bindings without initializing the camera SDK."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType


def sdk_search_paths(src_root: Path, configured: str = "") -> list[Path]:
    paths = [Path(configured)] if configured.strip() else []
    checkout = src_root.parent
    if (checkout / "pyproject.toml").is_file():
        paths.append(checkout / "external" / "optitrack-sdk-1.10" / "runtime")
    # Wheel installs in a project-local venv can keep the SDK beside it.
    if src_root.resolve().is_relative_to(Path(sys.prefix).resolve()):
        adjacent = Path(sys.prefix).parent / "external" / "optitrack-sdk-1.10" / "runtime"
        if adjacent.is_dir() and adjacent not in paths:
            paths.append(adjacent)
    paths.extend((src_root / "handtrack" / "cameras" / "optitrack" / "Release",
                  src_root / "unity_hand_tracking" / "optitrack_cam_py" / "Release"))
    return paths


def load_sdk(src_root: Path) -> tuple[ModuleType | None, str]:
    paths = sdk_search_paths(src_root, os.environ.get("OPTITRACK_CAM_PY_PATH", "").strip())
    for path in reversed(paths):
        if path.is_dir():
            # Explicit configuration wins, including when a fallback was already on sys.path.
            entry = str(path)
            if entry in sys.path:
                sys.path.remove(entry)
            sys.path.insert(0, entry)
    try:
        return importlib.import_module("optitrack_cam"), ""
    except Exception as exc:  # noqa: BLE001 - Optional native imports must not break webcam applications.
        return None, f"{type(exc).__name__}: {exc}"
