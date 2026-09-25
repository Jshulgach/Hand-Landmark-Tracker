"""Backend resolution helpers for application entry modules."""

from __future__ import annotations

import importlib
import os


def optitrack_sdk_available() -> bool:
    try:
        config = importlib.import_module("handtrack.cameras.optitrack.config")
    except Exception:
        return False

    return getattr(config, "optitrack_cam", None) is not None


def resolve_backend(requested_backend: str | None = None) -> str:
    if requested_backend is None:
        requested_backend = os.environ.get("HANDTRACK_BACKEND", "auto")

    requested_backend = str(requested_backend).strip().lower()
    if requested_backend not in {"auto", "webcam", "optitrack"}:
        raise ValueError("backend must be one of: auto, webcam, optitrack")

    if requested_backend != "auto":
        return requested_backend

    return "optitrack" if optitrack_sdk_available() else "webcam"
