"""Prepare, verify and install a private Windows/Python 3.11 WiLoR transfer kit.

Prepare on an online configured PC; verify/install use only local files.
Licensed assets in the generated kit must stay out of public repositories.
"""
from __future__ import annotations

import argparse
import email
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import venv
import zipfile
from pathlib import Path, PurePosixPath

import setup_wilor as setup

REPO = Path(__file__).resolve().parents[1]
TARGET = "windows-cpython311-amd64"


def digest(path):
    return setup.sha256(path)


def validate_platform():
    if (sys.platform != "win32" or platform.python_implementation() != "CPython"
            or sys.version_info[:2] != (3, 11)
            or sys.maxsize <= 2**32 or platform.machine().lower() not in ("amd64", "x86_64")):
        raise RuntimeError("Use 64-bit Windows CPython 3.11 on an x64 computer")


def clean_environment():
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("PIP_")}
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        env.pop(key, None)
    env.update(PIP_CONFIG_FILE=os.devnull, PIP_NO_INDEX="1",
               PIP_DISABLE_PIP_VERSION_CHECK="1", YOLO_OFFLINE="true",
               YOLO_AUTOINSTALL="false", HF_HUB_OFFLINE="1")
    return env


def run(command, *, env=None):
    print("RUN " + subprocess.list2cmdline([str(item) for item in command]), flush=True)
    subprocess.run([str(item) for item in command], check=True, env=env)


def safe_path(root, relative):
    if (not isinstance(relative, str) or "\\" in relative or ":" in relative
            or relative.startswith("/") or any(part in ("", ".", "..") for part in relative.split("/"))):
        raise RuntimeError(f"Unsafe kit path: {relative}")
    path = root / PurePosixPath(relative)
    if not path.resolve().is_relative_to(root.resolve()):
        raise RuntimeError(f"Redirected kit path: {relative}")
    for item in (path, *path.parents):
        if item == root.parent:
            break
        if item.is_symlink():
            raise RuntimeError(f"Symlink refused: {item}")
    return path


def copy_preserving(source, destination):
    if destination.exists():
        if not destination.is_file() or digest(source) != digest(destination):
            raise RuntimeError(f"Different existing file preserved: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Same-volume hard links avoid another copy of immutable large checkpoints.
    if source.suffix in (".ckpt", ".pt"):
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    with destination.open("xb") as out, source.open("rb") as incoming:
        shutil.copyfileobj(incoming, out, 4 * 1024**2)


def wheel_pins(folder):
    pins = {}
    for wheel in sorted(folder.glob("*.whl")):
        with zipfile.ZipFile(wheel) as archive:
            names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA") and len(PurePosixPath(name).parts) == 2]
            if len(names) != 1:
                raise RuntimeError(f"Invalid wheel metadata: {wheel.name}")
            metadata = email.message_from_bytes(archive.read(names[0]))
        name = re.sub(r"[-_.]+", "-", metadata["Name"]).lower()
        if name in pins:
            raise RuntimeError(f"Duplicate wheel package: {name}")
        pins[name] = (metadata["Version"], digest(wheel))
    return pins


def prepare(output, root, sdk=None):
    validate_platform()
    from audit_wilor import pinned_packages
    python = REPO / ".venv-wilor/Scripts/python.exe"
    if not python.is_file():
        raise RuntimeError("First set up .venv-wilor on this online PC")
    output.mkdir(parents=True, exist_ok=True)
    wheels = output / "wheels"
    wheels.mkdir(exist_ok=True)
    lock = REPO / "requirements/wilor-windows-cu118.txt"
    run([python, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
         "-r", lock, "--wheel-dir", wheels])
    run([python, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
         *setup.BOOTSTRAP, REPO, "--wheel-dir", wheels])
    expected = pinned_packages(lock)
    expected.update(item.split("==") for item in setup.BOOTSTRAP)
    metadata = email.message_from_bytes(subprocess.check_output(
        [python, "-m", "pip", "show", "mavis-track"]))
    expected["mavis-track"] = metadata["Version"]
    actual = wheel_pins(wheels)
    if set(actual) != set(expected) or any(actual[name][0] != value for name, value in expected.items()):
        raise RuntimeError("Wheelhouse does not exactly match the locked runtime and bootstrap")
    lines = [f"{name}=={actual[name][0]} --hash=sha256:{actual[name][1]}" for name in sorted(actual)]
    (output / "offline-requirements.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != setup.SOURCE_REVISION:
        raise RuntimeError("Upstream source revision differs from the validated version")
    run(["git", "-C", root, "diff", "--exit-code", "HEAD", "--", "wilor",
         "pretrained_models/model_config.yaml", "mano_data/mano_mean_params.npz"])
    for name, (size, expected_hash) in setup.CHECKPOINTS.items():
        path = root / "pretrained_models" / name
        if path.stat().st_size != size or digest(path) != expected_hash:
            raise RuntimeError(f"Checkpoint verification failed: {name}")
    for folder in ("wilor", "mano_data", "pretrained_models"):
        for source in sorted((root / folder).rglob("*")):
            if (not source.is_file() or source.is_symlink() or "__pycache__" in source.parts
                    or source.suffix in (".pyc", ".download") or ".mavis-runtime" in source.parts):
                continue
            copy_preserving(source, output / "assets/WiLoR" / source.relative_to(root))
    for name in ("license.txt", "README.md"):
        copy_preserving(root / name, output / "assets/WiLoR" / name)
    if not (output / "assets/WiLoR/mano_data/MANO_RIGHT.pkl").is_file():
        raise RuntimeError("Your licensed MANO_RIGHT.pkl is missing")
    if sdk:
        for name in ("optitrack_cam.cp311-win_amd64.pyd", "CameraLibrary2015x64S.dll", "binding-build.json"):
            copy_preserving(sdk / name, output / "assets/optitrack-runtime" / name)
    tools = output / "tools"
    tools.mkdir(exist_ok=True)
    for name in ("wilor_offline.py", "setup_wilor.py", "audit_wilor.py"):
        shutil.copyfile(REPO / "scripts" / name, tools / name)
    shutil.copyfile(lock, tools / lock.name)
    shutil.copyfile(REPO / "docs/wilor-install-audit.md", output / "GUIDE.md")
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        shutil.copyfile(REPO / name, output / name)
    copy_preserving(REPO / "docs/source/_static/HandDynamic.mp4", output / "reference.mp4")
    (output / "README.txt").write_text(
        "PRIVATE transfer kit: includes licensed MANO assets and possibly a vendor SDK runtime.\n"
        "Do not publish this folder. Transfer the WHOLE folder by USB/Drive.\n"
        "Prerequisites: Windows x64, Python 3.11 x64, NVIDIA display driver and OptiTrack USB drivers.\n"
        "Those OS components are not installed by this kit. No compiler or Git is needed on the target.\n"
        "From the target MAVIS repository folder (or an empty installation directory):\n"
        "  py -3.11 PATH-TO-KIT/tools/wilor_offline.py install PATH-TO-KIT --target . --require-cuda\n"
        "Then run external/launch_wilor_multicam.cmd and click Start Tracking.\n"
        "The installer verifies SHA256 for all files, refuses conflicting assets and uses local wheels only.\n"
        "It preserves your regular .venv. A virtual environment must not be copied between PCs.\n",
        encoding="utf-8")
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "kit-manifest.json":
            files.append({"path": path.relative_to(output).as_posix(),
                          "size": path.stat().st_size, "sha256": digest(path)})
    (output / "kit-manifest.json").write_text(json.dumps(
        {"schema": 1, "target": TARGET, "source_revision": revision,
         "mavis_source_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
         "files": files}, indent=2),
        encoding="utf-8")
    verify(output)
    print(f"PRIVATE KIT READY: {output.resolve()}", flush=True)


def verify(bundle):
    bundle = bundle.resolve()
    manifest = json.loads((bundle / "kit-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != 1 or manifest.get("target") != TARGET:
        raise RuntimeError("Unsupported kit schema/target")
    records = manifest["files"]
    names = [row["path"] for row in records]
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate manifest paths")
    for row in records:
        path = safe_path(bundle, row["path"])
        if not path.is_file() or path.stat().st_size != row["size"] or digest(path) != row["sha256"]:
            raise RuntimeError(f"Missing/corrupt kit file: {row['path']}")
    actual = {path.relative_to(bundle).as_posix() for path in (bundle / "wheels").glob("*")}
    listed = {name for name in names if name.startswith("wheels/")}
    if actual != listed or not all(name.endswith(".whl") for name in actual):
        raise RuntimeError("Unexpected or missing wheelhouse files")
    if "offline-requirements.txt" not in names or not listed:
        raise RuntimeError("Incomplete wheelhouse")
    print(f"Verified {len(records)} local files ({sum(row['size'] for row in records) / 2**30:.2f} GiB)", flush=True)
    return manifest


def asset_targets(bundle, target, manifest):
    pending = []
    for row in manifest["files"]:
        name = row["path"]
        if name.startswith("assets/WiLoR/"):
            relative = "external/WiLoR/" + name[len("assets/WiLoR/"):]
        elif name.startswith("assets/optitrack-runtime/"):
            relative = "external/optitrack-sdk-1.10/runtime/" + name[len("assets/optitrack-runtime/"):]
        else:
            continue
        destination = safe_path(target, relative)
        if destination.exists() and (not destination.is_file() or digest(destination) != row["sha256"]):
            raise RuntimeError(f"Different existing asset preserved: {destination}")
        pending.append((safe_path(bundle, name), destination))
    return pending


def install(bundle, target, require_cuda=False):
    validate_platform()
    manifest = verify(bundle)
    target = target.resolve()
    pending = asset_targets(bundle, target, manifest)  # Fail before changing the environment.
    folder = target / ".venv-wilor"
    if folder.is_symlink() or folder.resolve().parent != target:
        raise RuntimeError("Redirected environment refused")
    python = folder / "Scripts/python.exe"
    if folder.exists():
        config = folder / "pyvenv.cfg"
        if not (python.is_file() and config.is_file()
                and "include-system-site-packages = false" in config.read_text(encoding="utf-8").lower()):
            raise RuntimeError("Existing non-isolated environment preserved")
        version = subprocess.check_output([str(python), "-c",
             "import sys,struct,platform; print(sys.version_info[:2], struct.calcsize('P'), platform.python_implementation())"],
            text=True).strip()
        if version != "(3, 11) 8 CPython":
            raise RuntimeError(f"Existing Python environment preserved: {version}")
    else:
        target.mkdir(parents=True, exist_ok=True)
        venv.EnvBuilder(with_pip=True).create(folder)
    env = clean_environment()
    env["VIRTUAL_ENV"] = str(folder)
    env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
    wheels = bundle.resolve() / "wheels"
    run([python, "-m", "pip", "install", "--no-index", "--find-links", wheels,
         "--no-cache-dir", "--no-compile", "--upgrade", *setup.BOOTSTRAP], env=env)
    run([python, "-m", "pip", "install", "--no-index", "--find-links", wheels,
         "--only-binary=:all:", "--no-cache-dir", "--no-compile", "--require-hashes",
         "-r", bundle.resolve() / "offline-requirements.txt"], env=env)
    # MAVIS keeps a development version number across commits. Replace its
    # wheel even when an older editable checkout reports the same version.
    mavis = list(wheels.glob("mavis_track-*.whl"))
    if len(mavis) != 1:
        raise RuntimeError("Expected exactly one MAVIS wheel")
    run([python, "-m", "pip", "install", "--no-index", "--no-deps",
         "--force-reinstall", mavis[0]], env=env)
    run([python, "-m", "pip", "check"], env=env)
    for source, destination in pending:
        copy_preserving(source, destination)
    launcher = target / "external/launch_wilor_multicam.cmd"
    launcher.parent.mkdir(parents=True, exist_ok=True)
    text = ("@echo off\nsetlocal\ncd /d \"%~dp0..\"\n"
            "set \"MAVIS_WILOR_ROOT=%CD%\\external\\WiLoR\"\n"
            "if exist \"%CD%\\external\\optitrack-sdk-1.10\\runtime\\optitrack_cam.cp311-win_amd64.pyd\" "
            "set \"OPTITRACK_CAM_PY_PATH=%CD%\\external\\optitrack-sdk-1.10\\runtime\"\n"
            "\".venv-wilor\\Scripts\\python.exe\" -m mavis_track gui --backend optitrack "
            "--model wilor --wilor-device cuda --wilor-fast %*\nexit /b %ERRORLEVEL%\n")
    if launcher.exists() and launcher.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"Different launcher preserved: {launcher}")
    launcher.write_text(text, encoding="utf-8")
    env["MAVIS_WILOR_ROOT"] = str(target / "external/WiLoR")
    runtime = target / "external/optitrack-sdk-1.10/runtime"
    if runtime.is_dir():
        env["OPTITRACK_CAM_PY_PATH"] = str(runtime)
    audit = [python, bundle.resolve() / "tools/audit_wilor.py",
             "--root", target / "external/WiLoR", "--lock", bundle.resolve() / "tools/wilor-windows-cu118.txt",
             "--report", target / "external/wilor-install-audit.json", "--gui", "--block-network",
             "--video", bundle.resolve() / "reference.mp4"]
    if runtime.is_dir():
        audit.append("--optitrack")
    if require_cuda:
        audit.extend(["--require-cuda", "--fast"])
    run(audit, env=env)
    print(f"Installed without package indexes. Launch: {launcher}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "install"))
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--root", type=Path, default=REPO / "external/WiLoR")
    parser.add_argument("--optitrack-runtime", type=Path)
    parser.add_argument("--target", type=Path, default=Path.cwd())
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args.bundle.resolve(), args.root.resolve(), args.optitrack_runtime)
        elif args.command == "verify":
            verify(args.bundle)
        else:
            install(args.bundle, args.target, args.require_cuda)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
        print(f"WiLoR offline setup failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
