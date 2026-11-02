"""Bounded, incremental numeric recording with atomic manifest checkpoints."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class SessionWriter:
    """Internal writer. At most chunk_frames observations are buffered."""

    def __init__(self, folder: str | Path, metadata: dict, manifest: dict, *,
                 chunk_frames: int = 4096, video: bool = False) -> None:
        if type(chunk_frames) is not int or not 1 <= chunk_frames <= 16384:
            raise ValueError("chunk_frames must be between 1 and 16384")
        self.folder = Path(folder)
        self.metadata = dict(metadata)
        self.manifest = dict(manifest)
        self.chunk_frames = chunk_frames
        self.video = video
        self.count = 0
        self.first_time: float | None = None
        self.last_time: float | None = None
        self.buffer: dict[str, list] = {}
        self.chunks: list[dict] = []
        self.closed = False

    def append(self, observation: dict[str, Any]) -> None:
        if self.closed:
            raise RuntimeError("Recording is closed")
        stamp = float(observation["time_vector"])
        if not np.isfinite(stamp) or stamp < 0 or (self.last_time is not None and stamp <= self.last_time):
            raise ValueError("Recorded timestamps must increase")
        if self.buffer and observation.keys() != self.buffer.keys():
            raise ValueError("Observation fields changed during recording")
        for name, value in observation.items():
            self.buffer.setdefault(name, []).append(value)
        self.count += 1
        if self.first_time is None:
            self.first_time = stamp
        self.last_time = stamp
        if len(self.buffer["time_vector"]) >= self.chunk_frames:
            self._flush(chunked=True)
            self._checkpoint(complete=False)

    def _rate(self) -> float:
        if self.video or self.count < 2:
            return float(self.metadata["sampling_rate"])
        assert self.last_time is not None and self.first_time is not None
        return (self.count - 1) / (self.last_time - self.first_time)

    def _flush(self, *, chunked: bool) -> None:
        if not self.buffer or not self.buffer["time_vector"]:
            return
        arrays = {name: np.asarray(values) for name, values in self.buffer.items()}
        count = len(arrays["time_vector"])
        extra = dict(self.metadata, total_frames=count, sampling_rate=self._rate())
        if chunked:
            relative = f"chunks/{len(self.chunks):06d}.npz"
        else:
            relative = "landmarks.npz"
        target = self.folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **arrays, **extra)
        temporary.replace(target)
        if chunked:
            self.chunks.append({"file": relative, "frame_count": count})
        self.buffer.clear()

    def _checkpoint(self, *, complete: bool, error: str | None = None) -> None:
        document = dict(self.manifest, frame_count=self.count,
                        sampling_rate=self._rate(), complete=complete)
        if self.chunks:
            document.update(schema_version=3, storage="mavis-chunks",
                            chunk_schema_version=self.metadata["schema_version"], chunks=self.chunks)
        if error:
            document["error"] = str(error)
        temporary = self.folder / "session.tmp.json"
        temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
        temporary.replace(self.folder / "session.json")

    def finish(self, *, complete: bool = True, error: str | None = None) -> None:
        if self.closed:
            return
        if self.count:
            self._flush(chunked=bool(self.chunks))
            self._checkpoint(complete=complete, error=error)
        self.closed = True
