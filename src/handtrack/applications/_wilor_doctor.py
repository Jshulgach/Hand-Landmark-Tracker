"""Inspect WiLoR assets/dependencies without loading checkpoints or cameras."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import sys

from mavis_track.wilor import missing_assets, wilor_root


def run(root=None, *, backend="webcam", gui=False):
    folder = wilor_root(root)
    print(f"python          : {sys.version.split()[0]}")
    print(f"executable      : {sys.executable}")
    print(f"wilor:root      : {folder}")
    missing = missing_assets(folder)
    print(f"wilor:assets    : {'missing: ' + ', '.join(missing) if missing else 'ok'}")
    ready = not missing
    packages = {
        "torch": "torch", "torchvision": "torchvision", "pytorch_lightning": "pytorch-lightning",
        "ultralytics": "ultralytics", "smplx": "smplx", "yacs": "yacs",
        "timm": "timm", "einops": "einops", "skimage": "scikit-image",
        "pyrender": "pyrender", "chumpy": "chumpy", "dill": "dill",
    }
    for module, distribution in packages.items():
        try:
            present = importlib.util.find_spec(module) is not None
            version = importlib.metadata.version(distribution) if present else "missing"
        except (ImportError, ValueError, importlib.metadata.PackageNotFoundError):
            present, version = False, "missing"
        ready &= present
        print(f"wilor:{module:<18}: {version}")
    try:
        import torch
    except (ImportError, OSError, RuntimeError) as exc:
        print(f"wilor:device    : unavailable ({exc})")
        ready = False
    else:
        try:
            available = torch.cuda.is_available()
            print(f"wilor:device    : {'cuda' if available else 'cpu'}")
            if available:
                print(f"wilor:gpu       : {torch.cuda.get_device_name(0)}")
                print(f"wilor:vram      : {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
        except (OSError, RuntimeError) as exc:
            print(f"wilor:device    : probe failed ({exc})")
            ready = False
    if gui or backend in ("optitrack", "auto"):
        from ._doctor import main as base_doctor
        base_args = ["--backend", backend]
        if gui:
            base_args.append("--gui")
        ready &= base_doctor(base_args) == 0
    print("wilor:inference : not tested (this check does not load model weights)")
    return 0 if ready else 1
