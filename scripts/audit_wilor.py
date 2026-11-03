"""Check an installed WiLoR runtime, optionally running real inference without network."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def offline_environment():
    env = os.environ.copy()
    env.update(YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    if sys.platform == "win32":
        env.setdefault("PYOPENGL_PLATFORM", "win32")
    return env


def deny_network(attempts):
    # This guards Python socket calls, not arbitrary native DLL networking.
    def hook(event, args):
        if event == "socket.getaddrinfo":
            host = args[0]
        elif event == "socket.connect":
            address = args[1]
            host = address[0] if isinstance(address, tuple) else address
        else:
            return
        if host in ("localhost", "127.0.0.1", "::1", None):
            return
        attempts.append({"event": event, "host": str(host)})
        raise OSError("External Python networking blocked by WiLoR audit")
    sys.addaudithook(hook)


def pinned_packages(lock):
    pins = {}
    for line in lock.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "--")):
            continue
        if line.startswith("chumpy @"):
            pins["chumpy"] = "0.71"
        else:
            name, version = line.split("==")
            pins[name] = version
    return pins


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "external/WiLoR")
    parser.add_argument("--lock", type=Path, default=REPO / "requirements/wilor-windows-cu118.txt")
    parser.add_argument("--report", type=Path, default=Path("wilor-audit.json"))
    parser.add_argument("--video", type=Path, help="trusted reference video for real four-slot inference")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--optitrack", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--block-network", action="store_true")
    args = parser.parse_args(argv)
    os.environ.update(offline_environment())
    attempts = []
    if args.block_network:
        deny_network(attempts)
    checks = []
    report = {"python": sys.version, "executable": sys.executable, "root": str(args.root.resolve()),
              "network_guard": "Python socket calls" if args.block_network else "not enabled",
              "network_attempts": attempts, "checks": checks}
    def check(name, action):
        started = time.perf_counter()
        try:
            detail = action()
            checks.append({"name": name, "ok": True, "detail": detail,
                           "seconds": time.perf_counter() - started})
            print(f"PASS {name}: {detail}", flush=True)
        except Exception as exc:  # noqa: BLE001 - Collect independent native/runtime failures.
            checks.append({"name": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAIL {name}: {exc}", flush=True)

    def versions():
        bad = []
        for name, expected in pinned_packages(args.lock).items():
            try:
                found = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                found = "missing"
            if found != expected:
                bad.append(f"{name}: expected {expected}, got {found}")
        if bad:
            raise RuntimeError("; ".join(bad))
        return "All locked versions match"
    check("locked packages", versions)

    def checkpoint_hashes():
        from setup_wilor import CHECKPOINTS, sha256
        for name, (size, expected) in CHECKPOINTS.items():
            path = args.root / "pretrained_models" / name
            if not path.is_file() or path.stat().st_size != size or sha256(path) != expected:
                raise RuntimeError(f"Different/incomplete checkpoint: {path}")
        return "Both official checkpoint sizes/SHA-256 match; nothing deserialized"
    check("checkpoint integrity", checkpoint_hashes)

    def pip_check():
        run = subprocess.run([sys.executable, "-m", "pip", "check"],
                             capture_output=True, text=True, env=offline_environment(), check=False)
        if run.returncode:
            raise RuntimeError(run.stdout + run.stderr)
        return run.stdout.strip()
    check("package dependencies", pip_check)

    def doctor():
        from handtrack.applications._wilor_doctor import run
        if run(args.root, backend="optitrack" if args.optitrack else "webcam", gui=args.gui):
            raise RuntimeError("Dependency/asset/backend doctor failed; see output above")
        return "Imports, assets and requested backend available; no cameras opened"
    check("doctor", doctor)

    def gpu():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: check NVIDIA driver and CUDA-enabled PyTorch")
        value = (torch.ones((16, 16), device="cuda") @ torch.ones((16, 16), device="cuda"))[0, 0].item()
        torch.cuda.synchronize()
        if value != 16:
            raise RuntimeError("CUDA tensor operation failed")
        return {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                "vram_gib": torch.cuda.get_device_properties(0).total_memory / 2**30}
    if args.require_cuda:
        check("CUDA operation", gpu)

    if args.gui:
        def qt():
            env = offline_environment()
            env["QT_QPA_PLATFORM"] = "offscreen"
            code = ("from PyQt5.QtWidgets import QApplication; "
                    "from unity_hand_tracking.optitrack_cam_py.mocap_handtrack_gui import StereoHandTrackerGUI; "
                    "app=QApplication([]); w=StereoHandTrackerGUI(model='wilor'); w.close()")
            result = subprocess.run([sys.executable, "-c", code], env=env,
                                    capture_output=True, text=True, timeout=90, check=False)
            if result.returncode:
                raise RuntimeError(result.stderr[-3000:])
            return "Offscreen multi-camera window constructed/closed without starting cameras"
        check("GUI startup", qt)

    if args.video:
        def inference():
            import cv2
            import numpy as np
            import torch

            from handtrack.applications._wilor_preview import draw_wilor
            from mavis_track import WiLoRMultiViewTracker
            torch.set_num_threads(4)
            cap = cv2.VideoCapture(str(args.video.resolve()))
            try:
                valid, frame = cap.read()
            finally:
                cap.release()
            if not valid:
                raise RuntimeError("Cannot decode reference video")
            images = [frame, cv2.resize(frame, (960, 540)), frame.copy(),
                      cv2.flip(cv2.resize(frame, (640, 360)), 1)]
            with WiLoRMultiViewTracker(root=args.root, device="cuda" if args.require_cuda else "cpu",
                                       fast=args.fast, max_hands=1, batch_size=2) as model:
                outputs = model.process_views(images, timestamp=1)
                if len(outputs) != 4 or any(item is None for item in outputs):
                    raise RuntimeError("Camera slots changed")
                counts = []
                for image, result in zip(images, outputs):
                    if result is None:
                        counts.append(None)
                        continue
                    counts.append(len(result.hands))
                    for hand in result.hands:
                        if (hand.joints.shape != (21, 3) or hand.vertices.shape != (778, 3)
                                or not np.isfinite(hand.joints).all() or not np.isfinite(hand.vertices).all()):
                            raise RuntimeError("Invalid model geometry")
                    draw_wilor(image.copy(), result, model.faces)
                if not any(count for count in counts if count is not None):
                    raise RuntimeError("No hands found in reference video; use a known positive sample")
                missing = model.process_views([images[0], images[1], None, images[3]], timestamp=1.1)
                if missing[2] is not None or len(missing) != 4:
                    raise RuntimeError("Missing-camera slot changed")
                return {"hands_per_slot": counts, "device": model.device, "batch_size": model.batch_size,
                        "simulated_views": True, "missing_slot_preserved": True}
        check("real multi-view inference and overlay", inference)
    report["ok"] = all(item["ok"] for item in checks) and not attempts
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {args.report.resolve()}", flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
