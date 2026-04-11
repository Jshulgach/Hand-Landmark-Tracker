from ._video_files import check_video_path, load_video
from ._sync_utils import interpolate_landmarks_to_emg, interpolate_array_to_timebase

__all__ = [
    "check_video_path",
    "load_video",
    "load_txt_config",
    "load_yaml_config",
    "update_yaml_config",
    "get_sync_offset",
    "parse_events_file",
    "extract_event_indices",
    "SessionLoader",
    "interpolate_landmarks_to_emg",
    "interpolate_array_to_timebase",
]


def __getattr__(name):
    if name in {
        "load_txt_config",
        "load_yaml_config",
        "update_yaml_config",
        "get_sync_offset",
        "parse_events_file",
        "extract_event_indices",
    }:
        from . import _file_utils

        return getattr(_file_utils, name)
    if name == "SessionLoader":
        from ._session_loader import SessionLoader

        return SessionLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

