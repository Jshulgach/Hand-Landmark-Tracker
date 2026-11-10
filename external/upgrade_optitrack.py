"""Install the pinned binding-only release without changing SDK DLLs or starting cameras."""
from __future__ import annotations

import hashlib
import io
import json
import os
import ssl
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile

BUNDLE_URL = 'https://github.com/Jshulgach/Hand-Landmark-Tracker/releases/download/optitrack-binding-sync-v1/optitrack-binding-cp311-win_amd64.zip'
BUNDLE_SHA256 = '8be350efa5b8f673c0227f249513e61d76184dfffd90d9d5e36b9d4dd5ac3068'
SOURCE_REVISION = '4c8a4e18b6f4dc8d5df803307a7760b77a351afe'
SDK_REVISION = 'd0981e46b7f1838f7ac3af2ebc4f935ba763be6d'
SDK_DLL_SHA256 = '428b412009be1efc9600d11c61c46af84b6ff03c2c1f166401ea35feda2a1ee8'
BINDING = 'optitrack_cam.cp311-win_amd64.pyd'
MANIFEST = 'binding-build.json'
LICENSES = {'LICENSE', 'THIRD_PARTY_NOTICES.md', 'PYBIND11_LICENSE.txt'}
ALLOWED_FILES = {BINDING, MANIFEST, *LICENSES}
MAX_DOWNLOAD = 2 * 1024 * 1024
MAX_UNCOMPRESSED = 4 * 1024 * 1024


def download_with_windows_tls():
    # Schannel uses the Windows certificate store. Verification remains enabled.
    curl = Path(os.environ.get('SystemRoot', 'C:/Windows')) / 'System32' / 'curl.exe'
    if not curl.is_file():
        raise RuntimeError('Python certificate verification failed and Windows curl is unavailable. '
                           'Download the release ZIP in a trusted browser and use --binding-zip PATH.')
    with tempfile.TemporaryDirectory(prefix='mavis-binding-download-') as folder:
        destination = Path(folder) / 'binding.zip'
        command = [str(curl), '--fail', '--location', '--silent', '--show-error',
                   '--proto', '=https', '--proto-redir', '=https', '--max-time', '30',
                   '--max-filesize', str(MAX_DOWNLOAD), '--output', str(destination), BUNDLE_URL]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=35,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if completed.returncode or not destination.is_file():
            raise RuntimeError('Windows HTTPS download failed: ' + completed.stderr.strip())
        if destination.stat().st_size > MAX_DOWNLOAD:
            raise RuntimeError('Binding download exceeds the allowed size')
        return destination.read_bytes()


def download_bundle():
    if not BUNDLE_SHA256:
        raise RuntimeError('No verified binding release is configured yet')
    request = urllib.request.Request(BUNDLE_URL, headers={'User-Agent': 'MAVIS-binding-installer'})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read(MAX_DOWNLOAD + 1)
    except urllib.error.URLError as exc:
        if sys.platform != 'win32' or not isinstance(exc.reason, ssl.SSLCertVerificationError):
            raise
        print('Using Windows certificate-store HTTPS verification for this download.')
        payload = download_with_windows_tls()
    if len(payload) > MAX_DOWNLOAD:
        raise RuntimeError('Binding download exceeds the allowed size')
    return payload


def read_bundle(payload):
    if len(payload) > MAX_DOWNLOAD or hashlib.sha256(payload).hexdigest() != BUNDLE_SHA256:
        raise RuntimeError('Binding ZIP checksum mismatch; installed files were not changed')
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist()
            if len(entries) != len(ALLOWED_FILES) or {entry.filename for entry in entries} != ALLOWED_FILES:
                raise RuntimeError('Binding ZIP contains unexpected paths or duplicate files')
            if sum(entry.file_size for entry in entries) > MAX_UNCOMPRESSED:
                raise RuntimeError('Binding ZIP expands beyond the allowed size')
            files = {entry.filename: archive.read(entry) for entry in entries}
        record = json.loads(files[MANIFEST].decode('utf-8'))
        if (not isinstance(record, dict) or record.get('binding_api') != 2
                or record.get('python_abi') != 'cp311-win_amd64'
                or record.get('sdk_revision') != SDK_REVISION
                or record.get('sdk_dll_sha256') != SDK_DLL_SHA256
                or record.get('source_revision') != SOURCE_REVISION
                or record.get('binding_sha256') != hashlib.sha256(files[BINDING]).hexdigest()):
            raise RuntimeError('Binding manifest does not match the approved ABI, SDK or build')
        return files
    except (zipfile.BadZipFile, KeyError, UnicodeError, ValueError) as exc:
        raise RuntimeError(f'Invalid binding ZIP: {exc}') from exc


def verify_native(staging, runtime):
    code = """
import importlib, os, sys
from pathlib import Path
with os.add_dll_directory(sys.argv[2]):
    sys.path.insert(0,sys.argv[1])
    sdk=importlib.import_module('optitrack_cam')
    assert Path(sdk.__file__).parent.resolve()==Path(sys.argv[1]).resolve()
    assert sdk.BINDING_API_VERSION==2
    assert sdk.CAMERA_RECORDING_API_VERSION==1
    assert sdk.OPTIHUB_SYNC_CONTROL_API_VERSION==1
    for name in ('get_sync_status','configure_sync_preset','configure_exposure_output'):
        assert callable(getattr(sdk,name)),name
    assert callable(sdk.Frame.timing)
    assert callable(sdk.Camera.capture_info)
    for name in ('get_intensity','min_intensity','max_intensity','set_intensity',
                 'is_high_power_supported','is_high_power_available','get_high_power','set_high_power'):
        assert callable(getattr(sdk.Camera,name)),name
print('Native illumination, frame-timing and sync-control APIs verified without initializing cameras.')
"""
    completed = subprocess.run([sys.executable, '-B', '-c', code, str(staging), str(runtime)],
                               capture_output=True, text=True, timeout=30)
    if completed.returncode:
        raise RuntimeError('Native binding verification failed; existing files preserved: '
                           + (completed.stderr.strip() or completed.stdout.strip()))
    print(completed.stdout.strip())


def install_bundle(runtime, payload):
    runtime = Path(runtime).resolve()
    dll = runtime / 'CameraLibrary2015x64S.dll'
    if hashlib.sha256(dll.read_bytes()).hexdigest() != SDK_DLL_SHA256:
        raise RuntimeError('The installed SDK DLL does not match this binding; it was not changed')
    files = read_bundle(payload)
    binding, manifest = runtime / BINDING, runtime / MANIFEST
    backup = runtime / (BINDING + '.legacy-backup')
    license_dir = runtime / 'binding-licenses'
    for target in (binding, manifest, backup, license_dir):
        if target.is_symlink() or not target.resolve().is_relative_to(runtime):
            raise RuntimeError(f'Refusing a redirected binding installation path: {target}')
    previous = binding.read_bytes()
    previous_manifest = manifest.read_bytes() if manifest.exists() else None
    if previous == files[BINDING] and previous_manifest == files[MANIFEST]:
        print('Sync-capable binding API v2 is already installed.')
        return runtime
    with tempfile.TemporaryDirectory(prefix='binding-upgrade-', dir=runtime) as folder:
        staging = Path(folder)
        for name, data in files.items():
            (staging / name).write_bytes(data)
        verify_native(staging, runtime)
        if not backup.exists():
            with backup.open('xb') as stream:
                stream.write(previous)
        # Prepare notices before switching binaries; the SDK DLL is never copied/replaced.
        license_dir.mkdir(exist_ok=True)
        for name in LICENSES:
            target = license_dir / name
            if target.is_symlink() or not target.resolve().is_relative_to(runtime):
                raise RuntimeError('Refusing a redirected license file')
            target.write_bytes(files[name])
        changed = False
        try:
            (staging / BINDING).replace(binding)
            changed = True
            (staging / MANIFEST).replace(manifest)
        except OSError as exc:
            if changed:
                rollback = staging / 'rollback.pyd'
                rollback.write_bytes(previous)
                try:
                    rollback.replace(binding)
                except OSError as rollback_error:
                    raise RuntimeError('Rollback was blocked by a file lock; close MAVIS/Python and '
                                       f'restore the original binding from {backup}') from rollback_error
                # A failed atomic manifest replacement leaves its original contents intact.
            raise RuntimeError('Binding installation failed; previous binding preserved. '
                               'Close MAVIS/Python processes using it, then retry.') from exc
    print('Installed sync-capable binding API v2; original binding backup:', backup)
    print('SDK DLL, USB drivers and camera settings were not changed.')
    return runtime


def upgrade_runtime(runtime, archive_path=None):
    if archive_path is not None:
        archive_path = Path(archive_path)
        if archive_path.stat().st_size > MAX_DOWNLOAD:
            raise RuntimeError('Binding ZIP exceeds the allowed size')
        payload = archive_path.read_bytes()
    else:
        payload = download_bundle()
    return install_bundle(runtime, payload)
