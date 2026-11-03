"""MAVIS frame processing API. Importing this module never opens a camera."""

__version__ = "0.1.0"
__all__ = [
    "BODY_LANDMARK_NAMES",
    "QUALITY_REGION_NAMES",
    "BodyResult",
    "FrameResult",
    "HandBodyFrameResult",
    "HandBodyTracker",
    "HandResult",
    "HandTracker",
    "QualityFlag",
    "Session",
    "TrackingQuality",
    "WiLoRFrameResult",
    "WiLoRHandResult",
    "WiLoRMultiViewTracker",
    "WiLoRTracker",
    "iter_session_chunks",
    "load_session",
]


def __getattr__(name):
    if name in {"WiLoRTracker", "WiLoRMultiViewTracker", "WiLoRHandResult", "WiLoRFrameResult"}:
        from . import wilor
        return getattr(wilor, name)
    if name in {"HandBodyTracker", "HandBodyFrameResult", "BodyResult", "TrackingQuality",
                "QualityFlag", "BODY_LANDMARK_NAMES", "QUALITY_REGION_NAMES"}:
        from . import body
        return getattr(body, name)
    if name in {"HandTracker", "HandResult", "FrameResult"}:
        from .tracker import FrameResult, HandResult, HandTracker
        return {"HandTracker": HandTracker, "HandResult": HandResult,
                "FrameResult": FrameResult}[name]
    if name in {"Session", "load_session", "iter_session_chunks"}:
        from . import session
        return getattr(session, name)
    raise AttributeError(f"module 'mavis_track' has no attribute {name!r}")
