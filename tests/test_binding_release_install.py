"""Binding releases verify authenticity and preserve SDK/runtime state on failures."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'external/upgrade_optitrack.py'
spec=importlib.util.spec_from_file_location('binding_release_installer',SCRIPT)
installer=importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


@pytest.fixture
def release(monkeypatch,tmp_path):
    runtime=tmp_path/'runtime'
    runtime.mkdir()
    old=b'original binding'
    dll=b'matching SDK DLL'
    new=b'verified new binding'
    (runtime/installer.BINDING).write_bytes(old)
    (runtime/'CameraLibrary2015x64S.dll').write_bytes(dll)
    monkeypatch.setattr(installer,'SDK_DLL_SHA256',hashlib.sha256(dll).hexdigest())
    record={'binding_api':2,'python_abi':'cp311-win_amd64','sdk_revision':installer.SDK_REVISION,
            'source_revision':installer.SOURCE_REVISION,'sdk_dll_sha256':installer.SDK_DLL_SHA256,
            'binding_sha256':hashlib.sha256(new).hexdigest()}
    files={installer.BINDING:new,installer.MANIFEST:json.dumps(record).encode(),
           **{name:b'license notice' for name in installer.LICENSES}}
    def pack(data):
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
            for name,payload in data.items():
                archive.writestr(name,payload)
        payload=stream.getvalue()
        monkeypatch.setattr(installer,'BUNDLE_SHA256',hashlib.sha256(payload).hexdigest())
        return payload
    probes=[]
    def verify(staging,target):
        assert target==runtime
        assert (staging/installer.BINDING).read_bytes()==new
        probes.append(True)
    monkeypatch.setattr(installer,'verify_native',verify)
    return runtime,files,pack,old,dll,probes


def test_release_install_preserves_sdk_backs_up_original_and_is_repeatable(release):
    runtime,files,pack,old,dll,probes=release
    payload=pack(files)
    installer.install_bundle(runtime,payload)
    assert (runtime/installer.BINDING).read_bytes()==files[installer.BINDING]
    assert (runtime/(installer.BINDING+'.legacy-backup')).read_bytes()==old
    assert (runtime/'CameraLibrary2015x64S.dll').read_bytes()==dll
    assert set(path.name for path in (runtime/'binding-licenses').iterdir())==installer.LICENSES
    installer.install_bundle(runtime,payload)
    assert probes==[True]


@pytest.mark.parametrize('problem',['traversal','vendor-dll','wrong-abi','wrong-build','wrong-binding-hash'])
def test_invalid_bundle_never_changes_installed_binding(release,problem):
    runtime,files,pack,old,dll,probes=release
    modified=dict(files)
    if problem in ('traversal','vendor-dll'):
        modified['../escaped.pyd' if problem=='traversal' else 'CameraLibrary2015x64S.dll']=b'wrong'
    else:
        record=json.loads(modified[installer.MANIFEST])
        key={'wrong-abi':'python_abi','wrong-build':'source_revision','wrong-binding-hash':'binding_sha256'}[problem]
        record[key]='wrong'
        modified[installer.MANIFEST]=json.dumps(record).encode()
    with pytest.raises(RuntimeError):
        installer.install_bundle(runtime,pack(modified))
    assert (runtime/installer.BINDING).read_bytes()==old
    assert not (runtime/(installer.BINDING+'.legacy-backup')).exists()
    assert not probes


def test_corrupt_download_is_rejected_before_native_code_runs(release):
    runtime,files,pack,old,dll,probes=release
    payload=pack(files)
    with pytest.raises(RuntimeError,match='checksum'):
        installer.install_bundle(runtime,payload+b'changed')
    assert (runtime/installer.BINDING).read_bytes()==old and not probes


def test_native_import_failure_preserves_existing_installation(monkeypatch,release):
    runtime,files,pack,old,dll,probes=release
    def fail(*args):raise RuntimeError('Native DLL import failed')
    monkeypatch.setattr(installer,'verify_native',fail)
    with pytest.raises(RuntimeError,match='Native DLL'):
        installer.install_bundle(runtime,pack(files))
    assert (runtime/installer.BINDING).read_bytes()==old
    assert not (runtime/installer.MANIFEST).exists()


def test_manifest_failure_rolls_back_binary_and_keeps_old_manifest(monkeypatch,release):
    runtime,files,pack,old,dll,probes=release
    prior=b'previous manifest'
    (runtime/installer.MANIFEST).write_bytes(prior)
    replace=Path.replace
    def fail_manifest(self,target):
        if Path(target)==runtime/installer.MANIFEST:
            raise PermissionError('simulated metadata lock')
        return replace(self,target)
    monkeypatch.setattr(Path,'replace',fail_manifest)
    with pytest.raises(RuntimeError,match='previous binding preserved'):
        installer.install_bundle(runtime,pack(files))
    assert (runtime/installer.BINDING).read_bytes()==old
    assert (runtime/installer.MANIFEST).read_bytes()==prior
    assert (runtime/'CameraLibrary2015x64S.dll').read_bytes()==dll


def test_mismatched_sdk_dll_is_never_replaced(release):
    runtime,files,pack,old,dll,probes=release
    (runtime/'CameraLibrary2015x64S.dll').write_bytes(b'user SDK')
    with pytest.raises(RuntimeError,match='SDK DLL'):
        installer.install_bundle(runtime,pack(files))
    assert (runtime/'CameraLibrary2015x64S.dll').read_bytes()==b'user SDK'
    assert (runtime/installer.BINDING).read_bytes()==old and not probes



def test_certificate_store_fallback_keeps_verification_enabled(monkeypatch,release,tmp_path):
    from types import SimpleNamespace
    runtime,files,pack,old,dll,probes=release
    payload=pack(files)
    system=tmp_path/'Windows'
    curl=system/'System32/curl.exe'
    curl.parent.mkdir(parents=True)
    curl.touch()
    monkeypatch.setenv('SystemRoot',str(system))
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        assert command[0]==str(curl)
        assert '--insecure' not in command and '-k' not in command
        assert command[command.index('--proto')+1]=='=https'
        assert command[command.index('--proto-redir')+1]=='=https'
        Path(command[command.index('--output')+1]).write_bytes(payload)
        return SimpleNamespace(returncode=0,stderr='')
    monkeypatch.setattr(installer.subprocess,'run',run)
    assert installer.download_with_windows_tls()==payload
    assert len(calls)==1


def test_python_certificate_error_uses_windows_store_only_on_windows(monkeypatch,release):
    import ssl
    import urllib.error
    runtime,files,pack,old,dll,probes=release
    payload=pack(files)
    def failure(*args,**kwargs):
        raise urllib.error.URLError(ssl.SSLCertVerificationError('untrusted Python CA store'))
    monkeypatch.setattr(installer.urllib.request,'urlopen',failure)
    monkeypatch.setattr(installer.sys,'platform','win32')
    monkeypatch.setattr(installer,'download_with_windows_tls',lambda:payload)
    assert installer.download_bundle()==payload


def test_http_errors_do_not_switch_tls_clients(monkeypatch,release):
    import urllib.error
    runtime,files,pack,old,dll,probes=release
    pack(files)
    def failure(*args,**kwargs):
        raise urllib.error.URLError('HTTP connection error')
    def unexpected():raise AssertionError('TLS fallback was not needed')
    monkeypatch.setattr(installer.urllib.request,'urlopen',failure)
    monkeypatch.setattr(installer,'download_with_windows_tls',unexpected)
    with pytest.raises(urllib.error.URLError):
        installer.download_bundle()


@pytest.mark.parametrize("missing", [None, "recording_version", "sync_setter", "exposure_output", "frame_timing"])
def test_native_probe_requires_sync_and_frame_extensions(monkeypatch, tmp_path, missing):
    staging = tmp_path/"staging"
    staging.mkdir()
    source = """
BINDING_API_VERSION = 2
CAMERA_RECORDING_API_VERSION = 1
OPTIHUB_SYNC_CONTROL_API_VERSION = 1
def get_sync_status(): pass
def configure_sync_preset(): pass
def configure_exposure_output(): pass
class Frame:
    def timing(self): pass
class Camera:
    def capture_info(self): pass
"""
    for name in ("get_intensity", "min_intensity", "max_intensity", "set_intensity",
                 "is_high_power_supported", "is_high_power_available", "get_high_power", "set_high_power"):
        source += f"    def {name}(self): pass\n"
    changes = {"recording_version": "CAMERA_RECORDING_API_VERSION = 0",
               "sync_setter": "del configure_sync_preset",
               "exposure_output": "del configure_exposure_output",
               "frame_timing": "del Frame.timing"}
    if missing:
        source += changes[missing]+"\n"
    (staging/"optitrack_cam.py").write_text(source)
    real_run = installer.subprocess.run
    def run(command, **kwargs):
        command = list(command)
        # The test module is pure Python; only emulate Windows' DLL-directory
        # context so the actual subprocess capability checks run on every OS.
        command[3] = ("import os, contextlib; "
                      "os.add_dll_directory=lambda path: contextlib.nullcontext()\n") + command[3]
        return real_run(command, **kwargs)
    monkeypatch.setattr(installer.subprocess, "run", run)
    if missing:
        with pytest.raises(RuntimeError, match="Native binding verification failed"):
            installer.verify_native(staging, tmp_path)
    else:
        installer.verify_native(staging, tmp_path)
