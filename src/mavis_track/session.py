"""Validated landmark sessions, including numeric legacy single-hand files."""

import json
from dataclasses import dataclass, fields, replace
from itertools import chain
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
    tracking_mode: str = "hands"
    body_landmarks: np.ndarray | None = None
    body_world_landmarks: np.ndarray | None = None
    body_visibility: np.ndarray | None = None
    body_valid: np.ndarray | None = None
    quality_flags: np.ndarray | None = None
    quality_region_names: tuple[str, ...] = ()
    body_world_coordinate_space: str | None = None
    quality_visibility_threshold: float | None = None
    quality_max_gap: float | None = None


def _landmarks(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be numeric")
    if array.ndim == 3:
        array = array[:, None]
    if array.ndim != 4 or array.shape[1] not in (1, 2) or array.shape[2:] != (21, 3):
        raise ValueError(f"{name} must have shape (frames, [hands,] 21, 3)")
    return array.astype(np.float32)


def _load_flat_session(path) -> Session:
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
    if version not in (0, 1, 2):
        raise ValueError(f"Unsupported session schema version: {version}")
    if version in (1, 2) and not {"valid", "track_ids", "handedness", "time_vector", "sampling_rate", "coordinate_space", "timestamp_clock"} <= data.keys():
        raise ValueError(f"Session schema {version} is missing required metadata")
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
    combined = _load_body(data, count, labels, valid, space) if version == 2 else {}
    return Session(points, raw, world, image_sizes, valid, ids, labels, times, rate, names, angles,
                   space, clock, legacy, angle_space, **combined)


def _load_body(data, count, labels, valid, space):
    from .body import BODY_LANDMARK_NAMES, QUALITY_REGION_NAMES, QualityFlag
    required = {"tracking_mode", "body_landmarks", "body_world_landmarks", "body_valid",
                "body_visibility", "body_landmark_names", "body_world_coordinate_space",
                "quality_flags", "quality_region_names", "quality_visibility_threshold",
                "quality_max_gap"}
    if not required <= data.keys():
        raise ValueError("Session schema 2 is missing combined tracking metadata")
    if (str(data["tracking_mode"]) != "hand-body" or space != "image_normalized"
            or valid.shape[1] != 2):
        raise ValueError("Combined sessions require hand-body mode, image coordinates and two hand slots")
    if (data["body_landmark_names"].shape != (33,)
            or tuple(data["body_landmark_names"]) != BODY_LANDMARK_NAMES
            or data["quality_region_names"].shape != (3,)
            or tuple(data["quality_region_names"]) != QUALITY_REGION_NAMES):
        raise ValueError("Combined landmark/quality names do not match the schema")
    world_space = str(data["body_world_coordinate_space"])
    if world_space != "body_world_meters":
        raise ValueError("Body world coordinates must be labeled body_world_meters")
    visibility_threshold = float(data["quality_visibility_threshold"])
    max_gap = float(data["quality_max_gap"])
    if (not np.isfinite(visibility_threshold) or not 0 <= visibility_threshold <= 1
            or not np.isfinite(max_gap) or max_gap <= 0):
        raise ValueError("Combined quality thresholds must be finite and in range")
    body_mask = data["body_valid"]
    if (body_mask.shape != (count,) or body_mask.dtype.kind not in "biu"
            or not np.isin(body_mask, (0, 1)).all()):
        raise ValueError("body_valid must contain one boolean per frame")
    body_mask = body_mask.astype(bool)
    arrays = {}
    for name, shape in (("body_landmarks", (count, 33, 3)),
                        ("body_world_landmarks", (count, 33, 3)),
                        ("body_visibility", (count, 33))):
        array = data[name]
        if array.shape != shape or array.dtype.kind not in "fiu" or np.isinf(array).any():
            raise ValueError(f"{name} has invalid shape, dtype or infinite coordinates")
        array = array.astype(np.float32)
        if not np.isnan(array[~body_mask]).all():
            raise ValueError(f"Absent bodies must have NaN {name}")
        arrays[name] = array
    if not np.isfinite(arrays["body_landmarks"][body_mask]).all():
        raise ValueError("Present bodies must have finite landmarks")
    world = arrays["body_world_landmarks"]
    if not np.all(np.isfinite(world).all(axis=(1, 2)) | np.isnan(world).all(axis=(1, 2))):
        raise ValueError("Body world landmarks must be wholly available or absent per frame")
    visibility = arrays["body_visibility"]
    if np.any(np.isfinite(visibility) & ((visibility < 0) | (visibility > 1))):
        raise ValueError("Body visibility must be between zero and one or NaN")
    flags = data["quality_flags"]
    if (flags.shape != (count, 3) or flags.dtype.kind not in "iu"
            or np.any(flags < 0) or np.any(flags > 63)):
        raise ValueError("quality_flags must contain supported integer masks per region/frame")
    flags = flags.astype(np.uint16)
    left = np.any(valid & (labels == "Left"), axis=1)
    right = np.any(valid & (labels == "Right"), axis=1)
    if (np.any(valid & (labels == "Unknown"))
            or np.any(np.sum(valid & (labels == "Left"), axis=1) > 1)
            or np.any(np.sum(valid & (labels == "Right"), axis=1) > 1)):
        raise ValueError("Combined hands must have unique anatomical Left/Right labels")
    for column, present in enumerate((left, right, body_mask)):
        if np.any(((flags[:, column] & int(QualityFlag.MISSING)) == 0) != present):
            raise ValueError("Quality MISSING flags must agree with region presence")
    for array in (*arrays.values(), body_mask, flags):
        array.flags.writeable = False
    return dict(**arrays, body_valid=body_mask, quality_flags=flags,
                tracking_mode="hand-body", quality_region_names=QUALITY_REGION_NAMES,
                body_world_coordinate_space=world_space,
                quality_visibility_threshold=visibility_threshold, quality_max_gap=max_gap)


def _chunk_manifest(path):
    path = Path(path)
    manifest_path = path / "session.json" if path.is_dir() else path
    if manifest_path.name != "session.json" or not manifest_path.is_file():
        return None
    if manifest_path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Session manifest exceeds 4 MiB")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Invalid session manifest") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Session manifest must be an object")  # noqa: TRY004 - Invalid file content.
    if manifest.get("schema_version") != 3:
        return None
    chunks = manifest.get("chunks")
    count = manifest.get("frame_count")
    rate = manifest.get("sampling_rate")
    if (manifest.get("storage") != "mavis-chunks" or not isinstance(chunks, list)
            or not chunks or type(count) is not int or count <= 0
            or type(rate) not in (int, float) or not np.isfinite(rate) or rate <= 0
            or manifest.get("chunk_schema_version") not in (1, 2)):
        raise ValueError("Invalid chunked session metadata")
    total = 0
    for index, chunk in enumerate(chunks):
        if (not isinstance(chunk, dict) or chunk.get("file") != f"chunks/{index:06d}.npz"
                or type(chunk.get("frame_count")) is not int or not 0 < chunk["frame_count"] <= 16384):
            raise ValueError("Invalid chunk path or frame count")
        total += chunk["frame_count"]
    if count != total:
        raise ValueError("Chunk frame counts disagree with the session")
    return manifest_path.parent, manifest


def _same_layout(first, current):
    for field in fields(Session):
        a, b = getattr(first, field.name), getattr(current, field.name)
        if isinstance(a, np.ndarray):
            if not isinstance(b, np.ndarray) or a.shape[1:] != b.shape[1:] or a.dtype != b.dtype:
                raise ValueError(f"Chunk layout differs for {field.name}")
        elif isinstance(b, np.ndarray) or a != b:
            raise ValueError(f"Chunk metadata differs for {field.name}")


def iter_session_chunks(path):
    """Yield validated read-only sessions with bounded memory.

    Schema 0/1/2 files yield once. Schema 3 directories yield one chunk at a
    time, retaining the source's absolute timeline, IDs, units and quality.
    Incomplete recordings can be read through their last committed chunk.
    """
    info = _chunk_manifest(path)
    if info is None:
        yield _load_flat_session(path)
        return
    root, manifest = info
    first = None
    last_time = None
    for entry in manifest["chunks"]:
        file = root / entry["file"]
        if file.resolve().parent != (root / "chunks").resolve():
            raise ValueError("Chunk resolves outside its session directory")
        if (root / "chunks").resolve().parent != root.resolve():
            raise ValueError("Chunk directory resolves outside its session")
        # Keep the per-archive size/dtype limits; never load the whole run here.
        session = _load_flat_session(file)
        with np.load(file, allow_pickle=False) as archive:
            if int(archive["schema_version"]) != manifest["chunk_schema_version"]:
                raise ValueError("Chunk schema differs from its manifest")
        session = replace(session, sampling_rate=float(manifest["sampling_rate"]))
        if len(session.time_vector) != entry["frame_count"]:
            raise ValueError("Chunk length differs from its manifest")
        if last_time is not None and session.time_vector[0] <= last_time:
            raise ValueError("Chunk timestamps must increase across boundaries")
        if first is not None:
            _same_layout(first, session)
        else:
            first = session
        last_time = session.time_vector[-1]
        yield session


def load_session(path, *, max_bytes=256 * 1024 * 1024) -> Session:
    """Materialize a complete session; use iter_session_chunks for bounded RAM.

    Schema-3 arrays are allocated once and filled chunk by chunk. The default
    materialization limit is 256 MiB; use max_bytes=None only for trusted runs
    that fit in RAM. Replay/export use the bounded streaming iterator instead.
    """
    info = _chunk_manifest(path)
    if info is None:
        return _load_flat_session(path)
    iterator = iter_session_chunks(path)
    first = next(iterator)
    count = info[1]["frame_count"]
    required = sum(count * value[0:1].nbytes for field in fields(Session)
                   if isinstance((value := getattr(first, field.name)), np.ndarray))
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes <= 0):
        raise ValueError("max_bytes must be positive bytes or None")
    if max_bytes is not None and required > max_bytes:
        raise ValueError("Session exceeds the materialization limit; use iter_session_chunks or raise max_bytes explicitly")
    arrays = {field.name: np.empty((count, *getattr(first, field.name).shape[1:]),
                                   dtype=getattr(first, field.name).dtype)
              for field in fields(Session) if isinstance(getattr(first, field.name), np.ndarray)}
    offset = 0
    for chunk in chain((first,), iterator):
        end = offset + len(chunk.time_vector)
        for name, array in arrays.items():
            array[offset:end] = getattr(chunk, name)
        offset = end
    for array in arrays.values():
        array.flags.writeable = False
    return replace(first, **arrays)
