"""Downloader integrity and isolation without network or heavyweight models."""
import hashlib
import importlib.util
import os
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts/setup_wilor.py"
_SPEC = importlib.util.spec_from_file_location("wilor_setup", _PATH)
setup = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(setup)


def test_existing_checkpoint_is_verified_without_network(monkeypatch, tmp_path):
    content = b"verified checkpoint"
    path = tmp_path / "model.pt"
    path.write_bytes(content)
    monkeypatch.setattr(setup, "_run", lambda *_: pytest.fail("Unexpected network operation"))
    setup.download_checkpoint(tmp_path, path.name, len(content), hashlib.sha256(content).hexdigest(), "curl")
    assert path.read_bytes() == content


def test_different_existing_checkpoint_is_preserved(monkeypatch, tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"existing research checkpoint")
    monkeypatch.setattr(setup, "_run", lambda *_: pytest.fail("Unexpected network operation"))
    with pytest.raises(RuntimeError, match="preserved"):
        setup.download_checkpoint(tmp_path, path.name, 1, "0" * 64, "curl")
    assert path.read_bytes() == b"existing research checkpoint"


@pytest.mark.parametrize("valid", [True, False])
def test_download_only_publishes_matching_size_and_hash(monkeypatch, tmp_path, valid):
    content = b"new checkpoint"
    def download(command):
        assert "--proto" in command and "--proto-redir" in command
        assert "--insecure" not in command
        Path(command[command.index("--output") + 1]).write_bytes(content if valid else b"bad bytes")
    monkeypatch.setattr(setup, "_run", download)
    destination = tmp_path / "model.pt"
    if valid:
        setup.download_checkpoint(tmp_path, destination.name, len(content),
                                  hashlib.sha256(content).hexdigest(), "curl")
        assert destination.read_bytes() == content
        assert not destination.with_name("model.pt.download").exists()
    else:
        with pytest.raises(RuntimeError, match="verification failed"):
            setup.download_checkpoint(tmp_path, destination.name, len(content),
                                      hashlib.sha256(content).hexdigest(), "curl")
        assert not destination.exists()
        assert destination.with_name("model.pt.download").read_bytes() == b"bad bytes"


def test_install_commands_target_only_the_isolated_environment(monkeypatch, tmp_path):
    # The test never invokes a real installer or writes to the working .venv.
    monkeypatch.setattr(setup, "REPO", tmp_path)
    monkeypatch.setattr(setup.sys, "platform", "win32")
    monkeypatch.setattr(setup.sys, "version_info", (3, 11, 4, "final", 0))
    folder = tmp_path / ".venv-wilor"
    python = folder / "Scripts/python.exe"
    python.parent.mkdir(parents=True)
    python.touch()
    (folder / "pyvenv.cfg").write_text("include-system-site-packages = false")
    monkeypatch.setattr(setup.subprocess, "check_output", lambda *a, **k: "3.11")
    monkeypatch.setattr(setup.shutil, "which", lambda _: "uv.exe")
    calls = []
    monkeypatch.setattr(setup, "_run", lambda command, **kwargs: calls.append((command, kwargs)))
    setup.install_environment()
    assert calls
    for command, kwargs in calls:
        assert kwargs["env"]["VIRTUAL_ENV"] == str(folder)
        assert kwargs["env"]["PATH"].split(os.pathsep)[0] == str(python.parent)
        assert kwargs["env"]["YOLO_AUTOINSTALL"] == "false"
        if "uv.exe" in command:
            assert command[command.index("--python") + 1] == python
        else:
            assert command[0] == python


def test_foreign_existing_environment_is_preserved(monkeypatch, tmp_path):
    monkeypatch.setattr(setup, "REPO", tmp_path)
    monkeypatch.setattr(setup.sys, "platform", "win32")
    monkeypatch.setattr(setup.sys, "version_info", (3, 11, 4, "final", 0))
    folder = tmp_path / ".venv-wilor"
    folder.mkdir()
    (folder / "user-file.txt").write_text("preserve")
    with pytest.raises(RuntimeError, match="preserved"):
        setup.install_environment()
    assert (folder / "user-file.txt").read_text() == "preserve"



def test_fresh_pip_bootstrap_does_not_require_uninstalled_truststore(monkeypatch, tmp_path):
    monkeypatch.setattr(setup, "REPO", tmp_path)
    monkeypatch.setattr(setup.sys, "platform", "win32")
    monkeypatch.setattr(setup.sys, "version_info", (3, 11, 4, "final", 0))
    folder = tmp_path / ".venv-wilor"
    python = folder / "Scripts/python.exe"
    python.parent.mkdir(parents=True)
    python.touch()
    (folder / "pyvenv.cfg").write_text("include-system-site-packages = false")
    monkeypatch.setattr(setup.subprocess, "check_output", lambda *a, **k: "3.11")
    monkeypatch.setattr(setup.shutil, "which", lambda _: None)
    calls = []
    def run(command, *, env):
        assert command[0] == python
        assert env["VIRTUAL_ENV"] == str(folder)
        assert env["PATH"].split(os.pathsep)[0] == str(python.parent)
        # This feature fails on fresh pip 23.1 before it can download anything.
        assert "--use-feature=truststore" not in command
        calls.append(command)
    monkeypatch.setattr(setup, "_run", run)
    setup.install_environment(installer="pip")
    assert "pip==26.2.2" in calls[0] and "--upgrade" in calls[0]
    assert calls[-1] == [python, "-m", "pip", "check"]
