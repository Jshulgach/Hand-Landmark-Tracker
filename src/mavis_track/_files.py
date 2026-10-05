"""Bounded, pickle-free archive loading shared by sessions and calibration."""

import math
import zipfile
from pathlib import Path

import numpy as np


def load_npz(path, *, max_bytes=256 * 1024 * 1024) -> dict[str, np.ndarray]:
    path = Path(path)
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > 256 or sum(item.file_size for item in entries) > max_bytes:
                raise ValueError("NPZ exceeds the allowed uncompressed size or field count")
            if len({item.filename for item in entries}) != len(entries):
                raise ValueError("NPZ contains duplicate fields")
            if any(not item.filename.endswith(".npy") for item in entries):
                raise ValueError("NPZ must contain only NumPy array fields")
            for item in entries:
                with archive.open(item) as field:
                    version = np.lib.format.read_magic(field)
                    reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                              else np.lib.format.read_array_header_2_0)
                    shape, _, dtype = reader(field)
                    expected = math.prod(shape) * dtype.itemsize
                    if dtype.hasobject or dtype.kind not in "biufUS":
                        raise ValueError("Executable objects or unsupported array dtype")
                    if expected > max_bytes or expected != item.file_size - field.tell():
                        raise ValueError("NPZ array size differs from its declared shape")
        with np.load(path, allow_pickle=False) as archive:
            data = {name: archive[name].copy() for name in archive.files}
        if any(value.dtype.kind not in "biufUS" for value in data.values()):
            raise ValueError("NPZ fields must be numeric, boolean, or plain strings")
        return data
    except (zipfile.BadZipFile, ValueError, EOFError) as exc:
        raise ValueError(f"Invalid or unsafe NPZ file {path.name}: {exc}") from exc
