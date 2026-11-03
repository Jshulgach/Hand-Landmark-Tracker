"""Downloader integrity and isolation without network or heavyweight models."""
import hashlib
import importlib.util
import os
import subprocess
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



@pytest.fixture
def upstream(monkeypatch, tmp_path):
    source = tmp_path / "upstream"
    source.mkdir()
    files = {
        "wilor/models/wilor.py": b"# test inference source",
        "pretrained_models/model_config.yaml": b"MODEL: test",
        "mano_data/mano_mean_params.npz": b"test source parameter fixture",
        "README.md": b"test source readme",
        ".gitignore": b"mano_data/MANO_RIGHT.pkl\npretrained_models/*.pt\n*.download\n",
    }
    for name, data in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], stderr=subprocess.PIPE, text=True).strip()
    git("init", "--quiet")
    git("add", ".")
    git("-c", "user.name=WiLoR setup tests", "-c", "user.email=tests@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "fixture source")
    monkeypatch.setattr(setup, "SOURCE_URL", str(source))
    monkeypatch.setattr(setup, "SOURCE_REVISION", git("rev-parse", "HEAD"))
    return source


def test_source_merge_keeps_mano_checkpoints_and_partial_downloads_and_reruns_without_clone(
        monkeypatch, tmp_path, upstream):
    root = tmp_path / "target"
    assets = {
        "mano_data/MANO_RIGHT.pkl": b"my existing licensed MANO",
        "pretrained_models/detector.pt": b"my checkpoint",
        "pretrained_models/wilor_final.ckpt.download": b"resumable download",
        "research-notes.txt": b"keep my notes",
    }
    before = {}
    for name, data in assets.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        before[name] = (setup.sha256(path), path.stat().st_mtime_ns)
    setup.prepare_source(root)
    assert (root / "wilor/models/wilor.py").is_file()
    assert (root / ".git").is_dir()
    assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == setup.SOURCE_REVISION
    assert Path(subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"], text=True).strip()) == root
    subprocess.run(["git", "-C", str(root), "diff", "--exit-code", "HEAD", "--",
                    "wilor", "pretrained_models/model_config.yaml", "mano_data/mano_mean_params.npz"],
                   check=True, capture_output=True)
    for name, original in before.items():
        assert (setup.sha256(root / name), (root / name).stat().st_mtime_ns) == original
    monkeypatch.setattr(setup, "_clone_source", lambda _: pytest.fail("Unexpected source download on rerun"))
    setup.prepare_source(root)
    assert not list(tmp_path.glob(".mavis-wilor-source-*"))


def test_source_merge_preflights_all_conflicts_without_altering_existing_files(tmp_path, upstream):
    root = tmp_path / "target"
    custom = root / "wilor/models/wilor.py"
    custom.parent.mkdir(parents=True)
    custom.write_bytes(b"my custom source")
    mano = root / "mano_data/MANO_RIGHT.pkl"
    mano.parent.mkdir()
    mano.write_bytes(b"keep my licensed model")
    with pytest.raises(RuntimeError, match="preserved"):
        setup.prepare_source(root)
    assert custom.read_bytes() == b"my custom source"
    assert mano.read_bytes() == b"keep my licensed model"
    assert not (root / "README.md").exists()
    assert not (root / ".git").exists()
    assert not list(tmp_path.glob(".mavis-wilor-source-*"))


def test_failed_clone_cleans_staging_without_changing_mano(monkeypatch, tmp_path):
    root = tmp_path / "target"
    mano = root / "mano_data/MANO_RIGHT.pkl"
    mano.parent.mkdir(parents=True)
    mano.write_bytes(b"preserve")
    def interrupted(checkout):
        checkout.mkdir()
        (checkout / "partial-source").write_bytes(b"partial")
        raise RuntimeError("interrupted Git transfer")
    monkeypatch.setattr(setup, "_clone_source", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        setup.prepare_source(root)
    assert mano.read_bytes() == b"preserve"
    assert not (root / "partial-source").exists()
    assert not list(tmp_path.glob(".mavis-wilor-source-*"))


def test_failed_merge_rolls_back_only_new_source_files_and_can_retry(monkeypatch, tmp_path, upstream):
    root = tmp_path / "target"
    mano = root / "mano_data/MANO_RIGHT.pkl"
    mano.parent.mkdir(parents=True)
    mano.write_bytes(b"preserve")
    original_copy = setup.shutil.copyfileobj
    def interrupted(*args, **kwargs):
        raise OSError("simulated interrupted copy")
    monkeypatch.setattr(setup.shutil, "copyfileobj", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        setup.prepare_source(root)
    assert mano.read_bytes() == b"preserve"
    assert not (root / "README.md").exists() and not (root / ".git").exists()
    monkeypatch.setattr(setup.shutil, "copyfileobj", original_copy)
    setup.prepare_source(root)
    assert (root / "wilor/models/wilor.py").is_file()
    assert mano.read_bytes() == b"preserve"


def test_different_git_revision_is_preserved_without_downloading(monkeypatch, tmp_path):
    root = tmp_path / "existing"
    (root / ".git").mkdir(parents=True)
    monkeypatch.setattr(setup.subprocess, "check_output", lambda *a, **k: "different-revision")
    monkeypatch.setattr(setup, "_clone_source", lambda _: pytest.fail("Unexpected download"))
    with pytest.raises(RuntimeError, match="Different WiLoR revision preserved"):
        setup.prepare_source(root)


def test_source_parent_symlink_cannot_write_outside_target(tmp_path, upstream):
    root = tmp_path / "target"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (root / "wilor").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks requires privileges on this Windows host")
    with pytest.raises(RuntimeError, match="Redirected|Symlink"):
        setup.prepare_source(root)
    assert list(outside.iterdir()) == []
    assert not (root / "README.md").exists()


def test_one_command_runs_all_stages_in_order_and_requires_mano_first(monkeypatch, tmp_path):
    monkeypatch.setattr(setup, "REPO", tmp_path)
    monkeypatch.setattr(setup, "_require_runtime_python", lambda: None)
    calls = []
    monkeypatch.setattr(setup, "prepare_source", lambda root: calls.append(("source", root)))
    monkeypatch.setattr(setup, "download_checkpoints", lambda root: calls.append(("weights", root)))
    monkeypatch.setattr(setup, "install_environment", lambda installer: calls.append(("install", installer)))
    monkeypatch.setattr(setup, "check_environment", lambda root, **kwargs: calls.append(("check", kwargs)))
    assert setup.main(["--setup", "--require-cuda"]) == 1
    assert calls == []
    root = tmp_path / "external/WiLoR"
    mano = root / "mano_data/MANO_RIGHT.pkl"
    mano.parent.mkdir(parents=True)
    mano.write_bytes(b"licensed model")
    assert setup.main(["--setup", "--require-cuda"]) == 0
    assert calls == [("source", root), ("weights", root), ("install", "auto"),
                     ("check", {"require_cuda": True})]
    assert mano.read_bytes() == b"licensed model"
