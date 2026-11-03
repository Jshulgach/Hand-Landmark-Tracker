"""Transfer integrity, asset preservation and offline install command contracts."""
import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("offline_setup_test", SCRIPTS / "wilor_offline.py")
offline = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(SCRIPTS))
try:
    spec.loader.exec_module(offline)
finally:
    sys.path.remove(str(SCRIPTS))


def kit(tmp_path):
    root = tmp_path / "kit"
    files = {
        "wheels/pip-26.2.2-py3-none-any.whl": b"test bootstrap",
        "wheels/mavis_track-0.1.0-py3-none-any.whl": b"test application",
        "offline-requirements.txt": b"test lock",
        "assets/WiLoR/mano_data/MANO_RIGHT.pkl": b"licensed asset",
    }
    records = []
    for name, data in files.items():
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(data)
        records.append({"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {"schema": 1, "target": offline.TARGET, "files": records}
    (root / "kit-manifest.json").write_text(json.dumps(manifest))
    return root, manifest


@pytest.mark.parametrize("name", ["../secret", "/absolute", "C:/elsewhere", "a/../b",
                                  "a//b", "a\\b", "https://host/file"])
def test_transfer_paths_cannot_escape_root(tmp_path, name):
    with pytest.raises(RuntimeError, match="path"):
        offline.safe_path(tmp_path, name)


def test_kit_verification_detects_corruption_and_extra_wheels(tmp_path):
    root, _ = kit(tmp_path)
    offline.verify(root)
    wheel = root / "wheels/extra.whl"
    wheel.write_bytes(b"unexpected")
    with pytest.raises(RuntimeError, match="Unexpected"):
        offline.verify(root)
    wheel.unlink()
    asset = root / "assets/WiLoR/mano_data/MANO_RIGHT.pkl"
    asset.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="corrupt"):
        offline.verify(root)


def test_conflicting_assets_are_detected_before_installing(tmp_path):
    root, manifest = kit(tmp_path)
    target = tmp_path / "existing"
    destination = target / "external/WiLoR/mano_data/MANO_RIGHT.pkl"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"participant asset to preserve")
    with pytest.raises(RuntimeError, match="preserved"):
        offline.asset_targets(root, target, manifest)
    assert destination.read_bytes() == b"participant asset to preserve"
    assert not (target / ".venv-wilor").exists()


def test_source_copies_are_independent_and_conflicts_preserved(tmp_path):
    source = tmp_path / "model.py"
    source.write_bytes(b"original")
    target = tmp_path / "copy.py"
    offline.copy_preserving(source, target)
    source.write_bytes(b"edited source")
    assert target.read_bytes() == b"original"
    with pytest.raises(RuntimeError, match="preserved"):
        offline.copy_preserving(source, target)


def test_installer_uses_only_local_wheels_and_preserves_regular_environment(monkeypatch, tmp_path):
    bundle, _ = kit(tmp_path)
    target = tmp_path / "target"
    python = target / ".venv-wilor/Scripts/python.exe"
    python.parent.mkdir(parents=True)
    python.touch()
    (python.parent.parent / "pyvenv.cfg").write_text("include-system-site-packages = false")
    regular = target / ".venv/user-file"
    regular.parent.mkdir()
    regular.write_text("keep")
    monkeypatch.setattr(offline, "validate_platform", lambda: None)
    monkeypatch.setattr(offline.subprocess, "check_output", lambda *a, **k: "(3, 11) 8 CPython")
    monkeypatch.setenv("PIP_USE_FEATURE", "truststore")
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://unavailable.invalid/simple")
    monkeypatch.setenv("PYTHONPATH", "foreign-environment")
    calls = []
    def run(command, *, env):
        assert command[0] == python
        assert env["PIP_NO_INDEX"] == "1"
        assert "PIP_USE_FEATURE" not in env and "PIP_EXTRA_INDEX_URL" not in env
        assert "PYTHONPATH" not in env
        if "install" in command:
            assert "--no-index" in command
        calls.append(command)
    monkeypatch.setattr(offline, "run", run)
    offline.install(bundle, target, require_cuda=True)
    assert any("--require-hashes" in command for command in calls)
    assert any("--force-reinstall" in command for command in calls)
    assert "--require-cuda" in calls[-1] and "--block-network" in calls[-1]
    assert regular.read_text() == "keep"
    assert (target / "external/WiLoR/mano_data/MANO_RIGHT.pkl").read_bytes() == b"licensed asset"
    launcher = (target / "external/launch_wilor_multicam.cmd").read_text()
    assert "--backend optitrack" in launcher and "MAVIS_WILOR_ROOT" in launcher


def test_python_network_guard_fails_before_external_dns():
    code = """import sys, socket
sys.path.insert(0, sys.argv[1])
from audit_wilor import deny_network
attempts=[]
deny_network(attempts)
try:
    socket.create_connection(("example.invalid",443), timeout=.01)
except OSError:
    assert attempts and attempts[0]["event"] == "socket.getaddrinfo"
else:
    raise AssertionError("External networking was not blocked")
"""
    result = subprocess.run([sys.executable, "-c", code, str(SCRIPTS)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr



def test_wheel_inventory_ignores_vendored_metadata(tmp_path):
    with zipfile.ZipFile(tmp_path / "setuptools-84.2.1-py3-none-any.whl", "w") as archive:
        archive.writestr("setuptools-84.2.1.dist-info/METADATA", "Name: setuptools\nVersion: 84.2.1\n")
        archive.writestr("setuptools/_vendor/foo-1.dist-info/METADATA", "Name: foo\nVersion: 1\n")
    assert offline.wheel_pins(tmp_path)["setuptools"][0] == "84.2.1"


def test_wheel_sdk_discovery_finds_runtime_beside_environment(monkeypatch, tmp_path):
    from mavis_track._optitrack import sdk_search_paths
    environment = tmp_path / ".venv-wilor"
    monkeypatch.setattr(sys, "prefix", str(environment))
    runtime = tmp_path / "external/optitrack-sdk-1.10/runtime"
    runtime.mkdir(parents=True)
    paths = sdk_search_paths(environment / "Lib/site-packages")
    assert runtime in paths
    assert sdk_search_paths(environment / "Lib/site-packages", "explicit")[0] == Path("explicit")
