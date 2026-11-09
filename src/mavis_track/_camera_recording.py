"""Independent, bounded multi-camera video writer and timing sidecars."""
from __future__ import annotations

import hashlib
import json
import queue
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

from ._frame_diagnostics import FrameDiagnostics
from ._trial import trial_slug, utc_stamp


class CameraRecording:
    """Capture never waits for encoding. Overload is counted, never hidden.

    AVI playback uses nominal_fps; frames.jsonl is authoritative for timing.
    Hardware timestamps retain their SDK units and are not host timestamps.
    """

    def __init__(self, directory, *, nominal_fps=100., camera_ids=(), camera_settings=(), queue_size=32,
                 trial_name="trial", session_metadata=None, calibration_path=None, codec="MJPG"):
        if not 0 < nominal_fps <= 1000:
            raise ValueError("Video playback FPS must be between 0 and 1000")
        if type(queue_size) is not int or queue_size < 1:
            raise ValueError("Recording queue size must be a positive integer")
        if codec not in ('MJPG', 'FFV1'):
            raise ValueError("Video codec must be MJPG or FFV1")
        self.codec = codec
        slug = trial_slug(trial_name)
        self.trial_name = str(trial_name).strip()
        self.file_prefix = f"{slug}_{utc_stamp()}"
        self.video_files = {}
        self.session_metadata = dict(session_metadata or {})
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.calibration = {"available": False}
        if calibration_path is not None:
            source = Path(calibration_path)
            self.calibration['source'] = str(source)
            if source.is_file():
                data = source.read_bytes()
                (self.directory / 'calibration.npz').write_bytes(data)
                self.calibration.update(available=True, file='calibration.npz',
                                        sha256=hashlib.sha256(data).hexdigest())
        self.fps = float(nominal_fps)
        self.camera_ids = list(camera_ids)
        self.camera_settings = list(camera_settings)
        self._queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._control_lock = threading.Lock()
        self.written = 0
        self.frames_per_camera = {}
        self.diagnostics = FrameDiagnostics()
        self.dropped_by_kind = {}
        self.dropped = 0
        self.error = None
        self.started = time.perf_counter()
        self.started_utc = datetime.now(timezone.utc).isoformat()
        self._write_manifest(False)
        self._thread = threading.Thread(target=self._run, name="camera-recorder", daemon=True)
        self._thread.start()

    def status(self):
        with self._lock:
            return {"written": self.written, "dropped": self.dropped, "error": self.error,
                        "pending": self._queue.qsize(), "active": not self._stop.is_set(),
                        "sdk_missing": sum(s['sdk_missing'] for s in self.diagnostics.cameras.values()),
                        "sdk_discontinuities": sum(s['sdk_resets_or_duplicates'] for s in self.diagnostics.cameras.values())}

    def _submit(self, item):
        with self._lock:
            if self._stop.is_set() or self.error:
                return
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                self.dropped += 1
                kind = item[0] if item[0] != 'frame' else f'camera_{item[1]}'
                self.dropped_by_kind[kind] = self.dropped_by_kind.get(kind, 0) + 1

    def frame(self, camera_index, image, metadata):
        # Capture creates a new array each time; retain it until encoded.
        self._submit(("frame", camera_index, image, dict(metadata)))

    def sync(self, status):
        self._submit(("sync", dict(status)))

    def event(self, event, **details):
        self._submit(("event", dict(event=event, host_monotonic=time.perf_counter(), **details)))

    def control_event(self, event, **details):
        """Low-rate trigger audit bypasses the frame queue, including after writer errors."""
        with self._control_lock, (self.directory/'control_events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(event=event, host_monotonic=time.perf_counter(), **details),
                                    allow_nan=False)+'\n')

    def _write_manifest(self, complete):
        data = {"schema": "mavis-camera-video-v1", "complete": complete,
                    "started_utc": self.started_utc,
                    "trial_name": self.trial_name, "session_metadata": self.session_metadata,
                    "calibration": self.calibration, "video_files": dict(self.video_files),
                    "software": {"python": sys.version.split()[0], "opencv": cv2.__version__},
                    "host_started_monotonic": self.started, "timestamp_clock": "time.perf_counter seconds",
                    "nominal_video_fps": self.fps, "camera_ids": self.camera_ids,
                    "video_codec": self.codec,
                    "camera_settings": self.camera_settings,
                    "video": "Timestamped per-camera AVI (no tracking overlays)",
                    "timing": "frames.jsonl; nominal video FPS does not preserve variable timing",
                    "sync": "sync.jsonl contains sampled configuration, not electrical input edges",
                    "input_edge_recording_supported": False,
                    "frames_written": self.written, "queue_items_dropped": self.dropped, "error": self.error}
        data["frames_per_camera"] = dict(self.frames_per_camera)
        data['frame_diagnostics'] = self.diagnostics.summary()
        data['queue_drops_by_kind'] = dict(self.dropped_by_kind)
        temporary = self.directory / "manifest.json.tmp"
        temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.directory / "manifest.json")

    def _run(self):
        writers, sizes, counts = {}, {}, {}
        try:
            with (self.directory / "frames.jsonl").open("w", encoding="utf-8") as frames, \
                    (self.directory / "sync.jsonl").open("w", encoding="utf-8") as sync, \
                    (self.directory / "events.jsonl").open("w", encoding="utf-8") as events:
                events.write(json.dumps({"event": 'recording_started', "host_monotonic": self.started,
                                             "utc": self.started_utc, "trial_name": self.trial_name}) + '\n')
                events.flush()
                while not self._stop.is_set() or not self._queue.empty():
                    try:
                        item = self._queue.get(timeout=.1)
                    except queue.Empty:
                        continue
                    if item[0] == "sync":
                        sync.write(json.dumps(item[1], allow_nan=False) + "\n")
                        sync.flush()
                        continue
                    if item[0] == 'event':
                        events.write(json.dumps(item[1], allow_nan=False) + '\n')
                        events.flush()
                        continue
                    _, index, image, metadata = item
                    size = (image.shape[1], image.shape[0])
                    if index not in writers:
                        filename = f"{self.file_prefix}_camera_{index}.avi"
                        self.video_files[str(index)] = filename
                        writer = cv2.VideoWriter(str(self.directory / filename),
                                                 cv2.VideoWriter_fourcc(*self.codec), self.fps, size)
                        writers[index] = writer
                        sizes[index] = size
                        counts[index] = 0
                        if not writer.isOpened():
                            raise RuntimeError(f"Cannot open video writer for camera {index}")
                    if sizes[index] != size:
                        raise RuntimeError(f"Camera {index} changed resolution during recording")
                    writers[index].write(image)
                    with self._lock:
                        self.diagnostics.observe(index, metadata)
                    metadata.update(camera_index=index, video_frame_index=counts[index])
                    frames.write(json.dumps(metadata, allow_nan=False) + "\n")
                    frames.flush()
                    counts[index] += 1
                    self.frames_per_camera[index] = counts[index]
                    with self._lock:
                        self.written += 1
                    if self.written % 100 == 0:
                        if shutil.disk_usage(self.directory).free < 64 * 1024 * 1024:
                            raise RuntimeError("Recording stopped: less than 64 MiB of free disk space")
                        self._write_manifest(False)
                events.write(json.dumps({"event": 'recording_stopped', "host_monotonic": time.perf_counter(),
                                             "frames_written": self.written}) + '\n')
        except Exception as exc:  # noqa: BLE001 - Recorder boundary must report native and finalization failures.
            with self._lock:
                self.error = str(exc)
            self._stop.set()
        finally:
            for writer in writers.values():
                try:
                    writer.release()
                except Exception as exc:  # noqa: BLE001 - Recorder boundary must report native and finalization failures.
                    with self._lock:
                        self.error = f"Cannot finalize video: {exc}"
            if not self.written and not self.error:
                with self._lock:
                    self.error = "No camera frames were recorded"
            try:
                (self.directory / 'diagnostics.json').write_text(json.dumps({
                    "cameras": self.diagnostics.summary(), "queue_drops_by_kind": self.dropped_by_kind,
                    "error": self.error, "note": 'SDK gaps do not count electrical input pulses.'}, indent=2) + '\n',
                    encoding='utf-8')
            except Exception as exc:  # noqa: BLE001 - Recorder boundary must report native and finalization failures.
                with self._lock:
                    self.error = f"Cannot finalize recording: {exc}"
            try:
                self._write_manifest(self._stop.is_set() and not self.error)
            except Exception as exc:  # noqa: BLE001 - Recorder boundary must report native and finalization failures.
                with self._lock:
                    self.error = f"Cannot finalize manifest: {exc}"

    def close(self):
        # Serialize stop with producers so the writer cannot exit before enqueue.
        with self._lock:
            self._stop.set()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise RuntimeError("Video writer is still flushing; retry stopping before closing cameras")
        if self.error:
            raise RuntimeError(self.error)
