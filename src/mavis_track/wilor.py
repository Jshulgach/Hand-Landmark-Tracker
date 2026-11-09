"""Optional official WiLoR adapter. Importing this module does not import Torch.

WiLoR outputs are model estimates, not calibrated multi-camera measurements.
No model weights, licensed MANO assets, or third-party source are bundled.
"""
from __future__ import annotations

import copy
import importlib
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True, slots=True)
class WiLoRHandResult:
    track_id: int
    handedness: str
    detection_score: float
    image_landmarks: np.ndarray  # (21, 2), full-image pixels
    image_vertices: np.ndarray  # (778, 2), full-image pixels
    joints: np.ndarray  # (21, 3), wrist-relative camera axes, model meters
    vertices: np.ndarray  # (778, 3), same origin/axes as joints
    wrist_camera_translation: np.ndarray  # (3,), model-estimated meters
    bbox: np.ndarray  # xyxy pixels
    angles: dict[str, float]


@dataclass(frozen=True, slots=True)
class WiLoRFrameResult:
    frame_id: int
    timestamp: float
    source_id: str
    image_size: tuple[int, int]
    hands: tuple[WiLoRHandResult, ...]
    focal_length: float
    timestamp_clock: str
    joint_space: str = "wrist_relative_camera_meters"
    translation_space: str = "model_estimated_camera_meters"
    image_space: str = "pixels"


def wilor_root(root: str | Path | None = None) -> Path:
    value = root if root is not None else os.environ.get("MAVIS_WILOR_ROOT")
    if value is None:
        # Source-checkout convenience; installed wheels should use the argument/env.
        checkout = Path(__file__).resolve().parents[2] / "external" / "WiLoR"
        adjacent = Path(sys.prefix).parent / "external" / "WiLoR"
        value = checkout if (checkout / "wilor/models/wilor.py").is_file() else adjacent
        if not (value / "wilor/models/wilor.py").is_file():
            value = checkout
    return Path(value).expanduser().resolve()


def missing_assets(root: str | Path | None = None) -> list[str]:
    folder = wilor_root(root)
    names = ("wilor/models/wilor.py", "pretrained_models/model_config.yaml",
             "pretrained_models/wilor_final.ckpt", "pretrained_models/detector.pt",
             "mano_data/mano_mean_params.npz", "mano_data/MANO_RIGHT.pkl")
    return [name for name in names if not (folder / name).is_file()]


def _owned(value: Any) -> np.ndarray:
    array = np.array(value, dtype=np.float32, copy=True)
    array.flags.writeable = False
    return array


def _project_hand(joints, vertices, camera, center, box_size, image_size, focal, right):
    """Undo the crop and left-hand reflection before full-image projection."""
    joints = np.array(joints, dtype=np.float64, copy=True)
    vertices = np.array(vertices, dtype=np.float64, copy=True)
    camera = np.array(camera, dtype=np.float64, copy=True)
    center = np.asarray(center, dtype=np.float64)
    if (joints.shape != (21, 3) or vertices.shape != (778, 3)
            or camera.shape != (3,) or center.shape != (2,)
            or right not in (0, 1) or not np.isfinite(box_size) or box_size <= 0
            or not np.isfinite(focal) or focal <= 0
            or not all(np.isfinite(a).all() for a in (joints, vertices, camera, center))
            or camera[0] <= 0):
        raise ValueError("WiLoR returned invalid geometry or crop parameters")
    sign = 2 * right - 1
    joints[:, 0] *= sign
    vertices[:, 0] *= sign
    width, height = image_size
    denominator = box_size * camera[0] + 1e-9
    translation = np.array([
        2 * (center[0] - width / 2) / denominator + sign * camera[1],
        2 * (center[1] - height / 2) / denominator + camera[2],
        2 * focal / denominator])
    projected = []
    for points in (joints, vertices):
        camera_points = points + translation
        if np.any(camera_points[:, 2] <= 1e-8):
            raise ValueError("WiLoR returned points behind the estimated camera")
        projected.append(camera_points[:, :2] / camera_points[:, 2:] * focal
                         + np.array([width / 2, height / 2]))
    wrist = joints[0].copy()
    return (*projected, joints - wrist, vertices - wrist, wrist + translation)


def _detector_class(root: Path):
    # Old detector checkpoints need undeclared dependencies. Never let their
    # loader invoke an unscoped package manager against another environment.
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / ".mavis-runtime"))
    ultralytics = importlib.import_module("ultralytics")
    utils: Any = importlib.import_module("ultralytics.utils")
    utils.ONLINE = False  # Also cover imports made by the host before MAVIS.
    checks: Any = importlib.import_module("ultralytics.utils.checks")
    checks.AUTOINSTALL = False  # Also cover a host that imported it first.
    ultralytics.settings.update({"sync": False})
    return ultralytics.YOLO


class _OfficialWiLoR:
    """Own heavyweight dependencies and models only when explicitly selected."""

    def __init__(self, root: Path, device: str, fast: bool,
                 progress: Callable[[str], None] | None = None):
        report = progress if progress is not None else (lambda stage: None)
        # The upstream renderer defaults to Linux EGL even on Windows. We do
        # not create its renderers; the live overlay uses OpenCV instead.
        if sys.platform == "win32":
            os.environ.setdefault("PYOPENGL_PLATFORM", "win32")
        existing = sys.modules.get("wilor")
        if existing is not None:
            location = Path(next(iter(existing.__path__))).resolve()
            if location != root / "wilor":
                raise RuntimeError(f"Another WiLoR checkout is already imported: {location}")
        source = str(root)
        inserted = source not in sys.path
        if inserted:
            sys.path.insert(0, source)
        try:
            report("Importing PyTorch")
            torch = importlib.import_module("torch")
            report("Importing WiLoR dependencies")
            YOLO = _detector_class(root)
            WiLoR = importlib.import_module("wilor.models").WiLoR
            get_config = importlib.import_module("wilor.configs").get_config
            self.dataset = importlib.import_module("wilor.datasets.vitdet_dataset").ViTDetDataset
        except ImportError as exc:
            raise RuntimeError(
                f"WiLoR dependency unavailable: {exc}. Use the separate WiLoR environment; "
                "see docs/wilor.md. The MediaPipe environment need not change.") from exc
        finally:
            if inserted:
                sys.path.remove(source)
        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but this PyTorch installation cannot use an NVIDIA GPU")
        if fast and self.device != "cuda":
            raise ValueError("WiLoR fast mode requires CUDA; omit --wilor-fast on CPU")
        report(f"Preparing WiLoR on {self.device.upper()}")
        cfg = get_config(str(root / "pretrained_models/model_config.yaml"))
        cfg.defrost()
        if "vit" in cfg.MODEL.BACKBONE.TYPE and "BBOX_SHAPE" not in cfg.MODEL:
            cfg.MODEL.BBOX_SHAPE = [192, 256]
        cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS", None)
        cfg.MANO.DATA_DIR = str(root / "mano_data")
        cfg.MANO.MODEL_PATH = str(root / "mano_data")
        cfg.MANO.MEAN_PARAMS = str(root / "mano_data/mano_mean_params.npz")
        cfg.freeze()
        self.cfg = cfg
        report("Loading reconstruction checkpoint")
        self.model = WiLoR.load_from_checkpoint(
            str(root / "pretrained_models/wilor_final.ckpt"),
            map_location="cpu", strict=False, cfg=cfg, init_renderer=False)
        report(f"Moving reconstruction model to {self.device.upper()}")
        self.model = self.model.to(self.device).eval()
        report(f"Loading hand detector on {self.device.upper()}")
        self.detector = YOLO(str(root / "pretrained_models/detector.pt")).to(self.device)
        self.fast = fast
        if fast:
            report("Enabling CUDA fast mode")
            self.model = self.model.half()
            self.model.backbone.skip_blocks = True
        self.faces = np.array(self.model.mano.faces, dtype=np.int32, copy=True)
        self.faces.flags.writeable = False

    def infer(self, bgr: np.ndarray, confidence: float, max_hands: int):
        return self.infer_many([bgr], confidence, max_hands, batch_size=max_hands)[0]

    def infer_many(self, images, confidence: float, max_hands: int, *, batch_size=2):
        """Detect each view, then batch crops across views through one model."""
        torch = self.torch
        datasets: list[Any] = []
        rows: list[tuple[int, int]] = []
        per_view: list[Any] = []
        results: list[list[dict[str, Any]]] = [[] for _ in images]
        for view, bgr in enumerate(images):
            boxes = self.detector.predict(
                bgr, conf=confidence, max_det=max_hands, verbose=False)[0].boxes
            if boxes is None or len(boxes) == 0:
                per_view.append(None)
                continue
            detections = boxes.data.detach().float().cpu().numpy()
            valid = (np.isfinite(detections).all(axis=1)
                     & np.isin(detections[:, 5], [0, 1])
                     & (detections[:, 2] > detections[:, 0])
                     & (detections[:, 3] > detections[:, 1])
                     & (detections[:, 4] >= confidence)
                     & (detections[:, 4] <= 1))
            detections = detections[valid]
            detections = detections[np.argsort(-detections[:, 4])][:max_hands]
            per_view.append(detections)
            if not len(detections):
                continue
            datasets.append(self.dataset(
                self.cfg, bgr, detections[:, :4], detections[:, 5],
                rescale_factor=2.0, fp16=self.fast))
            rows.extend((view, person) for person in range(len(detections)))
        if not datasets:
            return results
        loader = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset(datasets), batch_size=batch_size,
            shuffle=False, num_workers=0)
        offset = 0
        with torch.inference_mode():
            for batch in loader:
                batch = {key: value.to(self.device) if hasattr(value, "to") else value
                         for key, value in batch.items()}
                output = self.model(batch)
                def cpu(value):
                    return value.detach().float().cpu().numpy()
                joints = cpu(output["pred_keypoints_3d"])
                vertices = cpu(output["pred_vertices"])
                cameras = cpu(output["pred_cam"])
                centers = cpu(batch["box_center"])
                sizes = cpu(batch["box_size"])
                people = cpu(batch["personid"]).astype(int)
                for index, person in enumerate(people):
                    view, expected_person = rows[offset + index]
                    if person != expected_person:
                        raise RuntimeError("WiLoR crop order changed; refusing cross-camera association")
                    detections = per_view[view]
                    results[view].append({
                        "joints": joints[index], "vertices": vertices[index],
                        "camera": cameras[index], "center": centers[index],
                        "box_size": float(sizes[index]), "bbox": detections[person, :4],
                        "score": float(detections[person, 4]),
                        "right": int(detections[person, 5])})
                offset += len(people)
        return results

    def close(self):
        self.model = self.detector = None
        # Release our references; never empty another model's global CUDA cache.


class WiLoRTracker:
    """Process caller-owned BGR/RGB frames using a local official WiLoR checkout.

    Arrays are owned/read-only. Detector scores are not joint confidence.
    Image coordinates are pixels. 3D uses camera axes (x right, y down, z away),
    a wrist-relative origin, and MANO model scale; absolute position/scale are
    monocular estimates using WiLoR's assumed focal length, not calibration.
    Track IDs associate adjacent frames only and do not identify a person.
    An optional progress callback receives startup stages on the calling thread;
    it should return promptly and must not directly update GUI widgets.
    """

    def __init__(self, *, root: str | Path | None = None, device: str = "auto",
                 fast: bool = False, max_hands: int = 2, confidence: float = 0.3,
                 mirrored: bool = False, source_id: str = "frames",
                 progress: Callable[[str], None] | None = None):
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")
        if type(fast) is not bool or type(mirrored) is not bool:
            raise ValueError("fast and mirrored must be bool")
        if type(max_hands) is not int or max_hands not in (1, 2):
            raise ValueError("max_hands must be 1 or 2")
        if not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable or None")
        folder = wilor_root(root)
        missing = missing_assets(folder)
        if missing:
            raise FileNotFoundError(
                f"WiLoR assets missing in {folder}: {', '.join(missing)}. "
                "See docs/wilor.md; obtain MANO_RIGHT.pkl from the MANO website.")
        self._runtime: _OfficialWiLoR | None = (
            _OfficialWiLoR(folder, device, fast, progress=progress) if progress is not None
            else _OfficialWiLoR(folder, device, fast))
        self.faces = self._runtime.faces
        self.device = self._runtime.device
        self.max_hands, self.confidence = max_hands, confidence
        self.mirrored, self.source_id = mirrored, str(source_id)
        self._last_timestamp: float | None = None
        self._frame_id = self._next_id = 0
        self._tracks: dict[int, tuple[np.ndarray, str]] = {}

    def _prepare_frame(self, frame: np.ndarray, *, color: Literal["bgr", "rgb"] = "bgr",
                timestamp: float | None = None) -> tuple[np.ndarray, float]:
        if self._runtime is None:
            raise RuntimeError("WiLoR tracker is closed")
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1):
            raise ValueError("frame must be a nonempty uint8 image with three channels")
        if color not in ("bgr", "rgb"):
            raise ValueError("color must be bgr or rgb")
        stamp = time.perf_counter() if timestamp is None else float(timestamp)
        if (not np.isfinite(stamp) or stamp < 0
                or (self._last_timestamp is not None and stamp <= self._last_timestamp)):
            raise ValueError("timestamps must be finite, nonnegative and increasing")
        bgr = np.ascontiguousarray(frame[:, :, ::-1] if color == "rgb" else frame).copy()
        return bgr, stamp

    def process(self, frame: np.ndarray, *, color: Literal["bgr", "rgb"] = "bgr",
                timestamp: float | None = None) -> WiLoRFrameResult:
        bgr, stamp = self._prepare_frame(frame, color=color, timestamp=timestamp)
        assert self._runtime is not None
        items = self._runtime.infer(bgr, self.confidence, self.max_hands)
        return self._build_result(bgr, items, stamp, timestamp is not None)

    def _build_result(self, frame: np.ndarray, items: list[dict[str, Any]],
                      stamp: float, source_clock: bool) -> WiLoRFrameResult:
        assert self._runtime is not None
        height, width = frame.shape[:2]
        focal = float(self._runtime.cfg.EXTRA.FOCAL_LENGTH) / self._runtime.cfg.MODEL.IMAGE_SIZE * max(width, height)
        old = self._tracks if self._last_timestamp is not None and stamp - self._last_timestamp <= .5 else {}
        available = set(old)
        tracks, hands = {}, []
        from handtrack.processing._joint_angles import compute_all_joint_angles
        for item in items:
            image, image_mesh, joints, vertices, translation = _project_hand(
                item["joints"], item["vertices"], item["camera"], item["center"],
                item["box_size"], (width, height), focal, item["right"])
            right = bool(item["right"]) != self.mirrored
            label = "Right" if right else "Left"
            wrist = image[0] / [width, height]
            candidates = [(float(np.linalg.norm(wrist - old[key][0])), key)
                          for key in available if old[key][1] == label]
            distance, key = min(candidates, default=(float("inf"), -1))
            if distance > .25:
                key = self._next_id
                self._next_id += 1
            else:
                available.remove(key)
            tracks[key] = (wrist.copy(), label)
            hands.append(WiLoRHandResult(
                key, label, item["score"], _owned(image), _owned(image_mesh),
                _owned(joints), _owned(vertices), _owned(translation), _owned(item["bbox"]),
                compute_all_joint_angles(joints)))
        result = WiLoRFrameResult(
            self._frame_id, stamp, self.source_id, (width, height), tuple(hands), focal,
            "source" if source_clock else "monotonic")
        self._tracks = tracks
        self._frame_id += 1
        self._last_timestamp = stamp
        return result

    def close(self):
        if self._runtime is not None:
            runtime, self._runtime = self._runtime, None
            runtime.close()
        self._tracks.clear()

    def __enter__(self):
        if self._runtime is None:
            raise RuntimeError("WiLoR tracker is closed")
        return self

    def __exit__(self, *_: object):
        self.close()



class WiLoRMultiViewTracker:
    """One inference runtime shared across independent, ordered camera views.

    Track IDs belong to each view's source_id; no cross-camera hand association
    or hardware synchronization is inferred. None marks a missing camera frame.
    3D outputs retain each camera's separate monocular model coordinate system.
    """

    def __init__(self, *, batch_size: int = 2, **kwargs: Any) -> None:
        if type(batch_size) is not int or not 1 <= batch_size <= 16:
            raise ValueError("batch_size must be an integer between 1 and 16")
        self._owner = WiLoRTracker(**kwargs)
        self.batch_size = batch_size
        self.faces = self._owner.faces
        self.device = self._owner.device
        self._views: dict[int, WiLoRTracker] = {}

    @property
    def confidence(self):
        return self._owner.confidence

    @confidence.setter
    def confidence(self, value):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("confidence must be between 0 and 1")
        self._owner.confidence = float(value)

    def process_views(self, frames: Sequence[np.ndarray | None], *,
                      color: Literal["bgr", "rgb"] = "bgr",
                      timestamp: float | None = None) -> tuple[WiLoRFrameResult | None, ...]:
        runtime = self._owner._runtime
        if runtime is None:
            raise RuntimeError("WiLoR multi-view tracker is closed")
        frames = list(frames)
        stamp = time.perf_counter() if timestamp is None else float(timestamp)
        if not np.isfinite(stamp) or stamp < 0:
            raise ValueError("timestamps must be finite and nonnegative")
        if color not in ("bgr", "rgb"):
            raise ValueError("color must be bgr or rgb")
        images, indices = [], []
        for index, frame in enumerate(frames):
            if frame is None:
                continue
            if index not in self._views:
                # Only immutable settings and the runtime are shared. Each view
                # owns its timestamp, counters and adjacent-frame association.
                view = copy.copy(self._owner)
                view.source_id = f"{self._owner.source_id}:camera:{index}"
                view._tracks = {}
                view._frame_id = view._next_id = 0
                view._last_timestamp = None
                self._views[index] = view
            image, _ = self._views[index]._prepare_frame(frame, color=color, timestamp=stamp)
            images.append(image)
            indices.append(index)
        predictions = (runtime.infer_many(
            images, self.confidence, self._owner.max_hands, batch_size=self.batch_size)
            if images else [])
        if len(predictions) != len(images):
            raise RuntimeError("WiLoR did not return one prediction list per supplied view")
        results: list[WiLoRFrameResult | None] = [None] * len(frames)
        for index, image, items in zip(indices, images, predictions):
            results[index] = self._views[index]._build_result(
                image, items, stamp, timestamp is not None)
        for index, view in list(self._views.items()):
            if index >= len(frames):
                del self._views[index]
            elif frames[index] is None:
                view._tracks.clear()
        return tuple(results)

    def reset_tracking(self):
        """Discard temporal associations after inference has been suspended."""
        self._views.clear()

    def close(self):
        self._views.clear()
        self._owner.close()

    def __enter__(self):
        if self._owner._runtime is None:
            raise RuntimeError("WiLoR multi-view tracker is closed")
        return self

    def __exit__(self, *_):
        self.close()
