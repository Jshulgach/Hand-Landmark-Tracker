"""
Launcher for the OptiTrack multi-camera ChArUco calibration workflow.

Keeps calibration easy to run from examples/ while the implementation
remains under src/unity_hand_tracking/optitrack_cam_py.
"""

import importlib
import os
import subprocess
import sys
import traceback
from pathlib import Path


def _preflight(repo_root: Path) -> bool:
    """Check critical runtime dependencies before launching calibration."""
    ok = True

    print("[Preflight] Python executable:", sys.executable)
    print("[Preflight] Python version:", sys.version.split()[0])

    try:
        cv2 = importlib.import_module("cv2")
        print(f"[Preflight] cv2: ok (version={getattr(cv2, '__version__', 'unknown')})")
    except Exception as exc:
        ok = False
        print(f"[Preflight] cv2: FAILED ({exc})")

    # Ensure src/ imports are resolvable for config preflight when not installed as package.
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    try:
        cfg = importlib.import_module("unity_hand_tracking.optitrack_cam_py.config")
        sdk = getattr(cfg, "optitrack_cam", None)
        if sdk is None:
            ok = False
            print("[Preflight] optitrack_cam: FAILED (module not found)")
            print(
                "[Preflight] Hint: place the SDK in "
                "src/unity_hand_tracking/optitrack_cam_py/Release "
                "or set OPTITRACK_CAM_PY_PATH to the folder containing optitrack_cam.*.pyd"
            )
        else:
            print("[Preflight] optitrack_cam: ok")
    except Exception:
        ok = False
        print("[Preflight] optitrack_cam: FAILED (could not import config)")
        traceback.print_exc()

    return ok


def _is_msys_or_mingw_shell() -> bool:
    return bool(
        os.environ.get("MSYSTEM")
        or os.environ.get("MINGW_PREFIX")
        or os.environ.get("MSYS")
    )


def _build_child_env() -> dict[str, str]:
    """Build a cleaner env for native Windows extension loading."""
    env = os.environ.copy()

    if not _is_msys_or_mingw_shell():
        return env

    path_entries = env.get("PATH", "").split(";")
    blocked_markers = (
        "\\usr\\bin",
        "\\mingw64\\bin",
        "\\mingw32\\bin",
        "\\msys64\\",
        "\\git\\usr\\bin",
    )

    cleaned = []
    for entry in path_entries:
        entry_norm = entry.strip().lower().replace("/", "\\")
        if not entry_norm:
            continue
        if any(marker in entry_norm for marker in blocked_markers):
            continue
        cleaned.append(entry)

    env["PATH"] = ";".join(cleaned)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    print(
        "[Launcher] Detected MINGW/MSYS shell; using sanitized PATH for child process."
    )
    return env


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    target = (
        repo_root
        / "src"
        / "unity_hand_tracking"
        / "optitrack_cam_py"
        / "calibration.py"
    )

    if not target.exists():
        print(f"Error: could not find OptiTrack calibration script at: {target}")
        return 1

    if not _preflight(repo_root):
        print("[Preflight] Dependency check failed. Fix the issues above, then retry.")
        return 1

    cmd = [sys.executable, str(target), *sys.argv[1:]]
    child_env = _build_child_env()
    return subprocess.call(cmd, env=child_env, cwd=str(repo_root))


if __name__ == "__main__":
    raise SystemExit(main())
