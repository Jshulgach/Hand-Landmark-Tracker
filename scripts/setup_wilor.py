"""Prepare the optional WiLoR source, verified checkpoints and isolated runtime.

Examples (Windows, Python 3.11):
    python scripts/setup_wilor.py --prepare-source --download-checkpoints --install
    python scripts/setup_wilor.py --check --require-cuda

MANO_RIGHT.pkl must be obtained separately by the user under the MANO license.
No camera is opened and no checkpoint is deserialized by the downloader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://github.com/rolpotamias/WiLoR.git"
SOURCE_REVISION = "fcb911312a38fa8badd30d9656a167485d61b8f9"
BOOTSTRAP = ("pip==26.2.2", "setuptools==84.2.1", "wheel==0.48.0")
HF_REVISION = "99fe3d7acff8104ecca1055df7467709506c2fa6"
CHECKPOINTS = {
    "detector.pt": (53582271, "5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033"),
    "wilor_final.ckpt": (2564989533, "3e97aafc7dd08d883a4cc5a027df61fdb6fda6136dbd1319405413862ada6bb2"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command, *, env=None):
    subprocess.run([str(value) for value in command], check=True, cwd=REPO, env=env)


def prepare_source(root: Path):
    print("Preparing WiLoR source with Git (no Python packages are installed).", flush=True)
    if root.exists():
        if not (root / ".git").is_dir():
            raise RuntimeError(f"Existing directory preserved: {root}; expected a WiLoR Git checkout")
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if revision != SOURCE_REVISION:
            raise RuntimeError(f"Different WiLoR revision preserved: {revision}; expected {SOURCE_REVISION}")
        print(f"WiLoR source already at {revision}", flush=True)
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", SOURCE_URL, root])
    _run(["git", "-C", root, "sparse-checkout", "set", "wilor", "mano_data", "pretrained_models"])
    _run(["git", "-C", root, "fetch", "--depth", "1", "origin", SOURCE_REVISION])
    _run(["git", "-C", root, "checkout", "--detach", SOURCE_REVISION])


def download_checkpoint(folder: Path, name: str, size: int, digest: str, curl: str):
    destination = folder / name
    if destination.is_symlink() or folder.is_symlink():
        raise RuntimeError("Refusing redirected checkpoint path")
    if destination.exists():
        if destination.stat().st_size != size or sha256(destination) != digest:
            raise RuntimeError(f"Different existing checkpoint preserved: {destination}")
        print(f"Verified existing {name}", flush=True)
        return
    partial = destination.with_name(name + ".download")
    if partial.is_symlink():
        raise RuntimeError("Refusing redirected partial-download path")
    print(f"Downloading {name}: {size / 2**30:.2f} GiB", flush=True)
    url = f"https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/{HF_REVISION}/pretrained_models/{name}"
    _run([curl, "--silent", "--show-error", "--fail", "--location",
          "--proto", "=https", "--proto-redir", "=https",
          "--retry", "3", "--retry-delay", "2", "--connect-timeout", "20",
          "--max-time", "1800", "--max-filesize", str(size),
          "--continue-at", "-", "--output", partial, url])
    if partial.stat().st_size != size or sha256(partial) != digest:
        raise RuntimeError(f"Download verification failed; partial preserved: {partial}")
    os.replace(partial, destination)
    print(f"SHA256 verified: {name}", flush=True)


def download_checkpoints(root: Path):
    if not (root / "wilor/models/wilor.py").is_file():
        raise RuntimeError("Prepare the upstream source first with --prepare-source")
    folder = root / "pretrained_models"
    folder.mkdir(exist_ok=True)
    if sys.platform == "win32":
        curl_path = Path(os.environ["SystemRoot"]) / "System32/curl.exe"
        curl = str(curl_path) if curl_path.is_file() else None
    else:
        curl = shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is required for verified HTTPS downloads")
    for name, (size, digest) in CHECKPOINTS.items():
        download_checkpoint(folder, name, size, digest, curl)
    manifest = {
        "huggingface_revision": HF_REVISION,
        "files": {name: {"size": size, "sha256": digest}
                  for name, (size, digest) in CHECKPOINTS.items()},
    }
    (folder / "mavis-checkpoints.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def environment_paths():
    folder = REPO / ".venv-wilor"
    if folder.is_symlink() or folder.resolve().parent != REPO.resolve():
        raise RuntimeError("The runtime environment must remain inside this checkout")
    return folder, folder / "Scripts/python.exe"


def install_environment(installer="auto"):
    if sys.platform != "win32" or sys.version_info[:2] != (3, 11):
        raise RuntimeError("This runtime lock requires 64-bit Windows CPython 3.11")
    if sys.maxsize <= 2**32:
        raise RuntimeError("64-bit Python is required")
    folder, python = environment_paths()
    if folder.exists():
        config = folder / "pyvenv.cfg"
        if (not config.is_file() or not python.is_file()
                or "include-system-site-packages = false" not in config.read_text(encoding="utf-8").lower()):
            raise RuntimeError(f"Existing directory is not an isolated venv; preserved: {folder}")
        version = subprocess.check_output(
            [str(python), "-c", "import sys; print('%s.%s' % sys.version_info[:2])"], text=True).strip()
        if version != "3.11":
            raise RuntimeError(f"Existing runtime uses Python {version}; preserved")
    else:
        venv.EnvBuilder(with_pip=True).create(folder)
    print(f"Installing WiLoR packages into {python}", flush=True)
    runtime_env = os.environ.copy()
    runtime_env["VIRTUAL_ENV"] = str(folder)
    runtime_env["PATH"] = str(python.parent) + os.pathsep + runtime_env.get("PATH", "")
    runtime_env["YOLO_AUTOINSTALL"] = "false"
    lock = REPO / "requirements/wilor-windows-cu118.txt"
    uv = shutil.which("uv") if installer != "pip" else None
    if installer == "uv" and not uv:
        raise RuntimeError("uv was explicitly selected but is not installed")
    if uv:
        # Chumpy's legacy build script imports pip without declaring it.
        _run([uv, "--native-tls", "pip", "install", "--python", python,
              *BOOTSTRAP], env=runtime_env)
        _run([uv, "--native-tls", "pip", "install", "--python", python,
              "--index-strategy", "unsafe-best-match",
              "--no-build-isolation-package", "chumpy", "-r", lock], env=runtime_env)
        _run([uv, "--native-tls", "pip", "install", "--python", python,
              "--no-deps", "-e", REPO], env=runtime_env)
    else:
        # Fresh Python 3.11 environments can contain pip 23.1, which cannot
        # enable truststore before that package is installed. Bootstrap with
        # pip's normal certificate validation; the pinned modern pip includes
        # truststore and uses system certificates for the remaining installs.
        _run([python, "-m", "pip", "install",
              "--upgrade", *BOOTSTRAP], env=runtime_env)
        _run([python, "-m", "pip", "install", "--no-build-isolation",
              "-r", lock], env=runtime_env)
        _run([python, "-m", "pip", "install", "--no-deps", "-e", REPO], env=runtime_env)
    _run([python, "-m", "pip", "check"], env=runtime_env)
    print(f"Isolated runtime installed: {folder}", flush=True)


def check_environment(root: Path, *, require_cuda=False):
    _, python = environment_paths()
    if not python.is_file():
        raise RuntimeError("The isolated environment is missing; run with --install")
    _run([python, "-m", "mavis_track", "doctor", "--model", "wilor", "--wilor-root", root])
    if require_cuda:
        probe = """import sys
import torch
print("CUDA build:", torch.version.cuda)
if not torch.cuda.is_available():
    sys.exit("CUDA test requires the NVIDIA computer")
print("GPU:", torch.cuda.get_device_name(0))
tensor = torch.ones((16, 16), device="cuda")
value = (tensor @ tensor)[0, 0].item()
torch.cuda.synchronize()
if value != 16:
    sys.exit("CUDA tensor verification failed")
print("CUDA tensor operation passed")
"""
        _run([python, "-c", probe])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "external/WiLoR")
    parser.add_argument("--prepare-source", action="store_true")
    parser.add_argument("--download-checkpoints", action="store_true")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--installer", choices=("auto", "uv", "pip"), default="auto")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    try:
        if args.prepare_source:
            prepare_source(root)
        if args.download_checkpoints:
            download_checkpoints(root)
        if args.install:
            install_environment(args.installer)
        if args.check or args.install or args.require_cuda or not (
                args.prepare_source or args.download_checkpoints):
            check_environment(root, require_cuda=args.require_cuda)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"WiLoR setup failed: {exc}", file=sys.stderr)
        return 1
    print("Setup checks finished. Camera/inference verification is a separate step.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
