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
    "DataBroadcaster",
    "UDPBroadcaster",
    "LSLBroadcaster",
    "LSL_AVAILABLE",
]


def __getattr__(name):
    if name in {"check_video_path", "load_video"}:
        from ._video_files import check_video_path, load_video

        return {"check_video_path": check_video_path, "load_video": load_video}[name]

    if name in {
        "load_txt_config",
        "load_yaml_config",
        "update_yaml_config",
        "get_sync_offset",
        "parse_events_file",
        "extract_event_indices",
    }:
        from ._file_utils import (
            extract_event_indices,
            get_sync_offset,
            load_txt_config,
            load_yaml_config,
            parse_events_file,
            update_yaml_config,
        )

        return {
            "load_txt_config": load_txt_config,
            "load_yaml_config": load_yaml_config,
            "update_yaml_config": update_yaml_config,
            "get_sync_offset": get_sync_offset,
            "parse_events_file": parse_events_file,
            "extract_event_indices": extract_event_indices,
        }[name]

    if name == "SessionLoader":
        from ._session_loader import SessionLoader

        return SessionLoader

    if name in {"interpolate_landmarks_to_emg", "interpolate_array_to_timebase"}:
        from ._sync_utils import (
            interpolate_array_to_timebase,
            interpolate_landmarks_to_emg,
        )

        return {
            "interpolate_landmarks_to_emg": interpolate_landmarks_to_emg,
            "interpolate_array_to_timebase": interpolate_array_to_timebase,
        }[name]

    if name in {"DataBroadcaster", "UDPBroadcaster", "LSLBroadcaster", "LSL_AVAILABLE"}:
        from .broadcast import (
            LSL_AVAILABLE,
            DataBroadcaster,
            LSLBroadcaster,
            UDPBroadcaster,
        )

        return {
            "DataBroadcaster": DataBroadcaster,
            "UDPBroadcaster": UDPBroadcaster,
            "LSLBroadcaster": LSLBroadcaster,
            "LSL_AVAILABLE": LSL_AVAILABLE,
        }[name]

    raise AttributeError(f"module 'handtrack.io' has no attribute '{name}'")
