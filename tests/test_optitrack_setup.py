"""Fresh-checkout SDK setup, ownership and actionable native import errors."""
import hashlib
import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from mavis_track import _optitrack as loader

SETUP = Path(__file__).resolve().parents[1] / "external" / "setup_optitrack.py"
spec = importlib.util.spec_from_file_location("mavis_sdk_setup", SETUP)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


@pytest.fixture
def runtime_pair(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    payload = {"binding.pyd": b"binding", "library.dll": b"runtime"}
    monkeypatch.setattr(setup, "RUNTIME_FILES", {
        name: {"sha256": hashlib.sha256(data).hexdigest(), "source": name}
        for name, data in payload.items()
    })
    for name, data in payload.items():
        (source / name).write_bytes(data)
    return tmp_path / "checkout", source, payload


def test_runtime_recovery_verifies_pair_and_can_be_repeated(runtime_pair):
    root, source, payload = runtime_pair
    target = setup.restore_runtime(root, source)
    assert setup.check_runtime(root) == target
    for name, data in payload.items():
        assert (target / name).read_bytes() == data
    assert setup.restore_runtime(root, source) == target
    assert sorted(path.name for path in target.iterdir()) == sorted(payload)


def test_invalid_source_creates_no_runtime_files(runtime_pair):
    root, source, payload = runtime_pair
    (source / "library.dll").write_bytes(b"wrong SDK version")
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        setup.restore_runtime(root, source)
    assert not setup.runtime_folder(root).exists()


def test_existing_custom_runtime_is_never_overwritten(runtime_pair):
    root, source, payload = runtime_pair
    target = setup.runtime_folder(root)
    target.mkdir(parents=True)
    (target / "library.dll").write_bytes(b"user installation")
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        setup.restore_runtime(root, source)
    assert (target / "library.dll").read_bytes() == b"user installation"
    assert not (target / "binding.pyd").exists()


def test_history_failure_explains_shallow_clone_and_copy_option(monkeypatch, runtime_pair):
    root, source, payload = runtime_pair
    def missing(*args, **kwargs):
        raise subprocess.CalledProcessError(128, args[0])
    monkeypatch.setattr(setup.subprocess, "run", missing)
    with pytest.raises(RuntimeError, match="fetch --unshallow.*--runtime-dir"):
        setup.restore_runtime(root)
    assert not setup.runtime_folder(root).exists()


def test_wrong_interpreter_is_rejected_before_restoration(monkeypatch):
    monkeypatch.setattr(setup.sys, "version_info", (3, 12, 1))
    with pytest.raises(RuntimeError, match="64-bit CPython 3.11"):
        setup.validate_interpreter()


def test_explicit_runtime_precedes_checkout_and_legacy_locations(monkeypatch, tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname="mavis-track"')
    source = root / "src"
    explicit = tmp_path / "custom"
    paths = loader.sdk_search_paths(source, str(explicit))
    for path in paths:
        path.mkdir(parents=True)
    monkeypatch.setenv("OPTITRACK_CAM_PY_PATH", str(explicit))
    monkeypatch.setattr(loader.sys, "path", [str(paths[-1]), str(explicit)])
    sdk = SimpleNamespace()
    def imported(name):
        assert name == "optitrack_cam"
        assert loader.sys.path[:len(paths)] == [str(path) for path in paths]
        return sdk
    monkeypatch.setattr(loader.importlib, "import_module", imported)
    assert loader.load_sdk(source) == (sdk, "")


def test_source_checkout_auto_discovers_external_without_environment(monkeypatch, tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "pyproject.toml").touch()
    source = root / "src"
    runtime = root / "external" / "optitrack-sdk-1.10" / "runtime"
    runtime.mkdir(parents=True)
    monkeypatch.delenv("OPTITRACK_CAM_PY_PATH", raising=False)
    monkeypatch.setattr(loader.sys, "path", [])
    sdk = SimpleNamespace()
    monkeypatch.setattr(loader.importlib, "import_module", lambda _: sdk)
    assert loader.load_sdk(source) == (sdk, "")
    assert loader.sys.path[0] == str(runtime)
    assert runtime not in loader.sdk_search_paths(tmp_path / "site-packages")


def test_missing_native_dependency_is_not_reported_as_missing_module(monkeypatch, tmp_path):
    def broken(name):
        raise ImportError("DLL load failed: dependent CameraLibrary was not found")
    monkeypatch.setattr(loader.importlib, "import_module", broken)
    sdk, error = loader.load_sdk(tmp_path / "src")
    assert sdk is None
    assert error == "ImportError: DLL load failed: dependent CameraLibrary was not found"


def test_camera_start_reports_native_error_before_sdk_initialization(monkeypatch):
    from unity_hand_tracking.optitrack_cam_py import multi_mjpeg
    monkeypatch.setattr(multi_mjpeg, "optitrack_cam", None)
    monkeypatch.setattr(multi_mjpeg, "SDK_IMPORT_ERROR", "ImportError: missing dependent DLL")
    with pytest.raises(RuntimeError, match="missing dependent DLL.*external/setup_optitrack.py"):
        multi_mjpeg.CameraManager()


@pytest.mark.parametrize("available", [True, False])
def test_doctor_distinguishes_optional_environment_from_sdk_failure(monkeypatch, capsys, available):
    from handtrack.applications import _doctor
    binding = SimpleNamespace(__file__="C:/MAVIS/external/runtime/optitrack_cam.pyd")
    config = SimpleNamespace(optitrack_cam=binding if available else None,
                             SDK_IMPORT_ERROR="ImportError: required DLL is missing")
    monkeypatch.delenv("OPTITRACK_CAM_PY_PATH", raising=False)
    monkeypatch.setattr(_doctor, "_check_import", lambda _: (True, "ok"))
    monkeypatch.setattr(_doctor.importlib, "import_module", lambda _: config)
    assert _doctor.main(["--backend", "optitrack"]) == (0 if available else 1)
    output = capsys.readouterr().out
    assert "<not-set> (optional; automatic search)" in output
    if available:
        assert "optitrack:binding" in output and binding.__file__ in output
        assert "optitrack:import-error" not in output
    else:
        assert "optitrack:import-error" in output
        assert config.SDK_IMPORT_ERROR in output
        assert "optitrack:binding" not in output
