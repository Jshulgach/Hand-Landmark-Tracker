"""Prepare the historical MAVIS OptiTrack runtime locally; never initialize cameras."""
from __future__ import annotations

import argparse
import hashlib
import platform
import struct
import subprocess
import sys
from pathlib import Path

REVISION = 'd0981e46b7f1838f7ac3af2ebc4f935ba763be6d'
RUNTIME_FILES = {
    'optitrack_cam.cp311-win_amd64.pyd': {
        'sha256': '5a8ae9ab85cf515947d3c4ee24a50f4ed63b2f51f1db88c580b2cec49e83043b',
        'source': 'src/unity_hand_tracking/optitrack_cam_py/Release/optitrack_cam.cp311-win_amd64.pyd',
    },
    'CameraLibrary2015x64S.dll': {
        'sha256': '428b412009be1efc9600d11c61c46af84b6ff03c2c1f166401ea35feda2a1ee8',
        'source': 'src/unity_hand_tracking/optitrack_cam_py/Release/CameraLibrary2015x64S.dll',
    },
}


def validate_interpreter():
    if (sys.platform != 'win32' or platform.python_implementation() != 'CPython'
            or sys.version_info[:2] != (3, 11) or struct.calcsize('P') != 8
            or platform.machine().lower() not in ('amd64', 'x86_64')):
        raise RuntimeError(
            'This historical binding requires 64-bit CPython 3.11 on Windows x64. '
            f'Current interpreter: {sys.executable} ({platform.python_version()}, '
            f'{sys.platform}, {platform.machine()}). Use a Python 3.11 x64 venv, '
            'or build a binding matching your interpreter and SDK.')


def runtime_folder(repo):
    repo = Path(repo).resolve()
    target = (repo / 'external' / 'optitrack-sdk-1.10' / 'runtime').resolve()
    if not target.is_relative_to(repo):
        raise RuntimeError('Runtime destination resolves outside this checkout.')
    return target


def verify_data(name, data):
    if hashlib.sha256(data).hexdigest() != RUNTIME_FILES[name]['sha256']:
        raise RuntimeError(f'Checksum mismatch for {name}; no runtime files were replaced.')


def historical_file(repo, name):
    try:
        result = subprocess.run(
            ['git', 'show', f"{REVISION}:{RUNTIME_FILES[name]['source']}"],
            cwd=repo, capture_output=True, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError('Git is required for history recovery; install Git or use --runtime-dir.') from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            'Historical SDK objects are unavailable. Use a full Git clone '
            '(git fetch --unshallow for a shallow clone), or copy the runtime '
            'from your configured computer and pass --runtime-dir PATH. '
            'A GitHub source ZIP does not contain Git history.') from exc
    return result.stdout


def restore_runtime(repo, source=None):
    repo = Path(repo).resolve()
    target = runtime_folder(repo)
    pending = {}
    # Validate the complete pair and existing files before writing anything.
    for name in RUNTIME_FILES:
        data = (Path(source) / name).read_bytes() if source is not None else historical_file(repo, name)
        verify_data(name, data)
        output = target / name
        if output.is_symlink():
            raise RuntimeError(f'Refusing to replace a symlink: {output}')
        if output.exists() and output.read_bytes() != data:
            raise RuntimeError(f'Refusing to overwrite a different local runtime: {output}')
        pending[name] = data
    target.mkdir(parents=True, exist_ok=True)
    for name, data in pending.items():
        output = target / name
        if not output.exists():
            # Exclusive creation also refuses to overwrite files created concurrently.
            with output.open('xb') as stream:
                stream.write(data)
    check_runtime(repo)
    return target


def check_runtime(repo):
    target = runtime_folder(repo)
    for name in RUNTIME_FILES:
        verify_data(name, (target / name).read_bytes())
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--from-history', action='store_true',
                       help='restore only the known binding and DLL from existing local Git history')
    group.add_argument('--runtime-dir', type=Path,
                       help='copy the matching runtime pair from another configured computer')
    group.add_argument('--check', action='store_true', help='verify interpreter and existing runtime checksums')
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    try:
        validate_interpreter()
        target = check_runtime(repo) if args.check else restore_runtime(repo, args.runtime_dir)
    except (OSError, RuntimeError) as exc:
        print(f'OptiTrack setup failed: {exc}', file=sys.stderr)
        return 1
    print(f'Verified local Python 3.11 x64 runtime files: {target}')
    print('Vendor files remain local and ignored by Git; no headers or build libraries were restored.')
    print('Prerequisite on a new PC: complete the official OptiTrack Camera SDK installer USB driver step.')
    print('USB driver status: not checked or changed by this utility; see external/README.md for driver-only setup.')
    print('If USB drivers are already installed, test camera detection before changing them.')
    print('Local runtime: legacy Camera SDK 1.10.x DLL paired with this Python binding.')
    print('MAVIS source checkouts discover this runtime automatically after restarting the application.')
    print('Next: python -m mavis_track doctor --backend optitrack --gui')
    print('Then: python -m mavis_track gui --backend optitrack')
    print('No SDK initialization, camera access, driver installation or license acceptance was performed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
