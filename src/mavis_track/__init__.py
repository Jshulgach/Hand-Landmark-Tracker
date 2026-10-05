"""MAVIS frame processing API. Importing this module never opens a camera."""

__version__ = "0.1.0"
__all__ = ["FrameResult", "HandResult", "HandTracker", "Session", "load_session"]


def __getattr__(name):
    if name in {"HandTracker", "HandResult", "FrameResult"}:
        from .tracker import FrameResult, HandResult, HandTracker
        return {"HandTracker": HandTracker, "HandResult": HandResult,
                "FrameResult": FrameResult}[name]
    if name in {"Session", "load_session"}:
        from .session import Session, load_session
        return {"Session": Session, "load_session": load_session}[name]
    raise AttributeError(f"module 'mavis_track' has no attribute {name!r}")
