"""Validated landmark sessions, including numeric legacy single-hand files."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._files import load_npz


@dataclass(frozen=True, slots=True)
class Session:
    landmarks: np.ndarray
    raw_landmarks: np.ndarray
    world_landmarks: np.ndarray | None
    image_sizes: np.ndarray | None
    valid: np.ndarray
    track_ids: np.ndarray
    handedness: np.ndarray
    time_vector: np.ndarray
    sampling_rate: float
    angle_names: tuple[str, ...]
    angle_values: np.ndarray
    coordinate_space: str
    timestamp_clock: str
    legacy_single_hand: bool
    angle_space: np.ndarray


def _landmarks(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be numeric")
    if array.ndim == 3:
        array = array[:, None]
    if array.ndim != 4 or array.shape[1] not in (1, 2) or array.shape[2:] != (21, 3):
        raise ValueError(f"{name} must have shape (frames, [hands,] 21, 3)")
    return array.astype(np.float32)


def load_session(path) -> Session:
    """Load a numeric NPZ, rejecting executable objects and invalid shapes.

    The canonical shape is (frames, hands, 21, 3). Invalid hands use NaNs;
    numeric legacy (frames, 21, 3) files remain readable.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "landmarks.npz"
    data = load_npz(path)
    if "landmarks" not in data:
        raise ValueError("Session is missing landmarks")
    version = int(data.get("schema_version", 0))
    if version not in (0, 1):
        raise ValueError(f"Unsupported session schema version: {version}")
    if version == 1 and not {"valid", "track_ids", "handedness", "time_vector", "sampling_rate", "coordinate_space", "timestamp_clock"} <= data.keys():
        raise ValueError("Session schema 1 is missing required metadata")
    legacy = data["landmarks"].ndim == 3
    points = _landmarks(data["landmarks"], "landmarks")
    raw = _landmarks(data.get("raw_landmarks", data["landmarks"]), "raw_landmarks")
    if raw.shape != points.shape:
        raise ValueError("Raw and processed landmark shapes must match")
    world = _landmarks(data["world_landmarks"], "world_landmarks") if "world_landmarks" in data else None
    if world is not None and (world.shape != points.shape or np.isinf(world).any()):
        raise ValueError("World landmarks must match the session shape and cannot contain infinity")
    if world is not None:
        available = np.isfinite(world).all(axis=(2, 3))
        missing = np.isnan(world).all(axis=(2, 3))
        if not np.all(available | missing):
            raise ValueError("World landmarks must be wholly available or absent for each hand")
    count, hands = points.shape[:2]
    image_sizes = data.get("image_sizes")
    if image_sizes is not None:
        if (image_sizes.shape != (count, 2) or image_sizes.dtype.kind not in "iu"
                or np.any(image_sizes <= 0)):
            raise ValueError("image_sizes must contain positive integer width/height per frame")
        image_sizes.flags.writeable = False
    rate = float(data.get("sampling_rate", 30))
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("sampling_rate must be positive finite Hz")
    times = np.asarray(data.get("time_vector", np.arange(count) / rate), dtype=np.float64)
    if (times.shape != (count,) or not np.isfinite(times).all()
            or np.any(times < 0) or np.any(np.diff(times) <= 0)):
        raise ValueError("time_vector must match frames and contain increasing nonnegative seconds")
    inferred = np.isfinite(points).all(axis=(2, 3)) & np.any(points != 0, axis=(2, 3))
    valid_input = np.asarray(data.get("valid", inferred))
    if valid_input.dtype.kind not in "biu" or not np.isin(valid_input, (0, 1)).all():
        raise ValueError("valid must contain only boolean values")
    valid = valid_input.astype(bool)
    if valid.shape != (count, hands):
        raise ValueError("valid must have shape (frames, hands)")
    if not np.isfinite(points[valid]).all() or not np.isfinite(raw[valid]).all():
        raise ValueError("Valid hands must have finite landmarks")
    if np.isinf(points).any() or np.isinf(raw).any():
        raise ValueError("Landmarks cannot contain infinity")
    ids_raw = data.get("track_ids", np.where(valid, np.arange(hands), -1))
    if np.asarray(ids_raw).dtype.kind not in "iu":
        raise ValueError("track_ids must be integers")
    ids = np.asarray(ids_raw, dtype=np.int64)
    labels = np.asarray(data.get("handedness", np.full((count, hands), "Unknown")), dtype=str)
    if ids.shape != valid.shape or labels.shape != valid.shape:
        raise ValueError("Hand metadata must match (frames, hands)")
    if np.any(ids[valid] < 0) or np.any(ids[~valid] != -1):
        raise ValueError("Present hands need nonnegative IDs; absent hands must use -1")
    if np.any(~np.isin(labels, ("Left", "Right", "Unknown"))):
        raise ValueError("handedness must be Left, Right, or Unknown")
    for row, mask in zip(ids, valid):
        if len(np.unique(row[mask])) != int(mask.sum()):
            raise ValueError("Hand track IDs must be unique within a frame")
    names_array = np.asarray(data.get("angle_names", np.array([], dtype=str)))
    if names_array.ndim != 1 or names_array.dtype.kind not in "US":
        raise ValueError("angle_names must be a string vector")
    names = tuple(str(name) for name in names_array)
    if len(set(names)) != len(names):
        raise ValueError("angle_names must be unique")
    angles = np.asarray(data.get("angle_values", np.empty((count, hands, 0))), dtype=np.float32)
    if angles.ndim == 2 and hands == 1:
        angles = angles[:, None]
    if angles.shape != (count, hands, len(names)) or np.isinf(angles).any():
        raise ValueError("angle_values must match frames, hands, and angle_names")
    space = str(data.get("coordinate_space", "image_normalized"))
    if space not in ("image_normalized", "calibrated_world_meters"):
        raise ValueError("Unsupported session coordinate_space")
    clock = str(data.get("timestamp_clock", "video_relative"))
    angle_space = np.asarray(data.get("angle_space", np.full((count, hands), "legacy_unspecified")), dtype=str)
    if angle_space.shape != valid.shape or not np.isin(angle_space, ("image_scaled", "hand_world_meters", "legacy_unspecified", "absent")).all():
        raise ValueError("angle_space must describe each hand's geometric angle coordinate system")
    if world is not None:
        world.flags.writeable = False
    for array in (points, raw, valid, ids, labels, times, angles, angle_space):
        array.flags.writeable = False
    return Session(points, raw, world, image_sizes, valid, ids, labels, times, rate, names, angles,
                   space, clock, legacy, angle_space)
