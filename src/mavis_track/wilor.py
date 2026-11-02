"""Optional official WiLoR adapter. Importing this module does not import Torch.

WiLoR outputs are model estimates, not calibrated multi-camera measurements.
No model weights, licensed MANO assets, or third-party source are bundled.
"""
from __future__ import annotations

import importlib
import os
import sys
import time
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
        value = Path(__file__).resolve().parents[2] / "external" / "WiLoR"
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
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / ".mavis-runtime"))
    ultralytics = importlib.import_module("ultralytics")
    checks: Any = importlib.import_module("ultralytics.utils.checks")
    checks.AUTOINSTALL = False  # Also cover a host that imported it first.
    ultralytics.settings.update({"sync": False})
    return ultralytics.YOLO


class _OfficialWiLoR:
    """Own heavyweight dependencies and models only when explicitly selected."""

    def __init__(self, root: Path, device: str, fast: bool):
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
            torch = importlib.import_module("torch")
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
        self.model = WiLoR.load_from_checkpoint(
            str(root / "pretrained_models/wilor_final.ckpt"),
            map_location="cpu", strict=False, cfg=cfg, init_renderer=False)
        self.model = self.model.to(self.device).eval()
        self.detector = YOLO(str(root / "pretrained_models/detector.pt")).to(self.device)
        self.fast = fast
        if fast:
            self.model = self.model.half()
            self.model.backbone.skip_blocks = True
        self.faces = np.array(self.model.mano.faces, dtype=np.int32, copy=True)
        self.faces.flags.writeable = False

    def infer(self, bgr: np.ndarray, confidence: float, max_hands: int):
        torch = self.torch
        boxes = self.detector.predict(
            bgr, conf=confidence, max_det=max_hands, verbose=False)[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        detections = boxes.data.detach().float().cpu().numpy()
        valid = (np.isfinite(detections).all(axis=1)
                 & np.isin(detections[:, 5], [0, 1])
                 & (detections[:, 2] > detections[:, 0])
                 & (detections[:, 3] > detections[:, 1])
                 & (detections[:, 4] >= confidence)
                 & (detections[:, 4] <= 1))
        detections = detections[valid]
        detections = detections[np.argsort(-detections[:, 4])][:max_hands]
        if not len(detections):
            return []
        dataset = self.dataset(self.cfg, bgr, detections[:, :4], detections[:, 5],
                               rescale_factor=2.0, fp16=self.fast)
        loader = torch.utils.data.DataLoader(dataset, batch_size=max_hands,
                                             shuffle=False, num_workers=0)
        results = []
        with torch.inference_mode():
            for batch in loader:
                batch = {key: value.to(self.device) if hasattr(value, "to") else value
                         for key, value in batch.items()}
                output = self.model(batch)
                def cpu(value):
                    return value.detach().float().cpu().numpy()
                for index, person in enumerate(cpu(batch["personid"]).astype(int)):
                    results.append({
                        "joints": cpu(output["pred_keypoints_3d"][index]),
                        "vertices": cpu(output["pred_vertices"][index]),
                        "camera": cpu(output["pred_cam"][index]),
                        "center": cpu(batch["box_center"][index]),
                        "box_size": float(batch["box_size"][index].item()),
                        "bbox": detections[person, :4],
                        "score": float(detections[person, 4]),
                        "right": int(detections[person, 5])})
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
    """

    def __init__(self, *, root: str | Path | None = None, device: str = "auto",
                 fast: bool = False, max_hands: int = 2, confidence: float = 0.3,
                 mirrored: bool = False, source_id: str = "frames"):
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")
        if type(fast) is not bool or type(mirrored) is not bool:
            raise ValueError("fast and mirrored must be bool")
        if type(max_hands) is not int or max_hands not in (1, 2):
            raise ValueError("max_hands must be 1 or 2")
        if not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        folder = wilor_root(root)
        missing = missing_assets(folder)
        if missing:
            raise FileNotFoundError(
                f"WiLoR assets missing in {folder}: {', '.join(missing)}. "
                "See docs/wilor.md; obtain MANO_RIGHT.pkl from the MANO website.")
        self._runtime: _OfficialWiLoR | None = _OfficialWiLoR(folder, device, fast)
        self.faces = self._runtime.faces
        self.device = self._runtime.device
        self.max_hands, self.confidence = max_hands, confidence
        self.mirrored, self.source_id = mirrored, str(source_id)
        self._last_timestamp: float | None = None
        self._frame_id = self._next_id = 0
        self._tracks: dict[int, tuple[np.ndarray, str]] = {}

    def process(self, frame: np.ndarray, *, color: Literal["bgr", "rgb"] = "bgr",
                timestamp: float | None = None) -> WiLoRFrameResult:
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
        height, width = frame.shape[:2]
        focal = float(self._runtime.cfg.EXTRA.FOCAL_LENGTH) / self._runtime.cfg.MODEL.IMAGE_SIZE * max(width, height)
        old = self._tracks if self._last_timestamp is not None and stamp - self._last_timestamp <= .5 else {}
        available = set(old)
        tracks, hands = {}, []
        from handtrack.processing._joint_angles import compute_all_joint_angles
        for item in self._runtime.infer(bgr, self.confidence, self.max_hands):
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
            "source" if timestamp is not None else "monotonic")
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
