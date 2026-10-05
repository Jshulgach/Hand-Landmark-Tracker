"""Safe calibrated-camera state shared by the advanced camera backends."""

import os
import warnings
from pathlib import Path

import numpy as np

from ._files import load_npz


def default_calibration_dir(backend):
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "mavis-track" / "calibration" / backend


def load_camera_calibration(path, *, count, image_sizes=None, camera_ids=None, diagnostics=None):
    data = load_npz(path)
    n = int(data.get("num_cameras", 0))
    if n <= 0 or n != count:
        raise ValueError(f"Calibration has {n} cameras but {count} are connected")
    if int(data.get("schema_version", 0)) not in (0, 1, 2):
        raise ValueError("Unsupported calibration schema version")
    if "quality_passed" in data and not bool(data["quality_passed"]):
        raise ValueError("Calibration candidate did not pass its quality checks")
    if str(data.get("coordinate_units", "meters")) != "meters":
        raise ValueError("Calibration coordinates must be in meters")
    order = list(range(n))
    verified = False
    uid_fields = {"camera_uid_high", "camera_uid_low"} & data.keys()
    if uid_fields and (len(uid_fields) != 2 or not diagnostics):
        raise ValueError("Saved camera UIDs require current UID diagnostics; source indices cannot verify them")
    if diagnostics and "camera_uid_high" in data and "camera_uid_low" in data:
        stored = list(zip(data["camera_uid_high"].tolist(), data["camera_uid_low"].tolist()))
        current = [tuple(item.get("uid") or ()) for item in diagnostics]
        if len(set(stored)) != n or len(current) != n or set(current) != set(stored):
            raise ValueError("Connected camera UIDs do not match this calibration")
        order = [stored.index(uid) for uid in current]
        verified = True
    elif camera_ids is not None and "camera_ids" in data:
        stored = data["camera_ids"].tolist()
        if len(set(stored)) != n or len(camera_ids) != n or set(stored) != set(camera_ids):
            raise ValueError("Connected camera source IDs do not match this calibration")
        order = [stored.index(source) for source in camera_ids]
        verified = True
    else:
        warnings.warn("Legacy calibration has no verifiable camera identity; verify camera order before use", RuntimeWarning)
    saved_sizes = data.get("image_sizes")
    if saved_sizes is None and "img_size" in data:
        saved_sizes = np.tile(data["img_size"], (n, 1))
    if saved_sizes is not None:
        if saved_sizes.shape != (n, 2) or np.any(saved_sizes <= 0):
            raise ValueError("Invalid calibration image sizes")
        if image_sizes is not None and not np.array_equal(saved_sizes[order], image_sizes):
            raise ValueError("Camera resolution differs from calibration; recalibrate at the current resolution")
    elif image_sizes is not None:
        warnings.warn("Legacy calibration has no saved image sizes", RuntimeWarning)
    Ks, dists, Rs, Ts = [], [], [], []
    for index in order:
        try:
            K, dist, R, T = (np.asarray(data[f"{prefix}_{index}"], dtype=np.float64)
                             for prefix in ("camera_matrix", "dist_coeffs", "R", "T"))
        except KeyError as exc:
            raise ValueError(f"Calibration missing camera field {exc}") from exc
        if (K.shape != (3, 3) or R.shape != (3, 3) or T.size != 3
                or dist.size not in (4, 5, 8, 12, 14)
                or not all(np.isfinite(array).all() for array in (K, dist, R, T))):
            raise ValueError("Invalid camera matrix, distortion, rotation, or translation")
        if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
            raise ValueError("Invalid camera intrinsics")
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(R), 1, atol=1e-4):
            raise ValueError("Camera rotation must be a proper orthonormal matrix")
        Ks.append(K); dists.append(dist); Rs.append(R); Ts.append(T.reshape(3, 1))
    return Ks, dists, Rs, Ts, verified
