from types import SimpleNamespace

import pytest

from scripts import check_dependencies


WARNING = "mediapipe 0.10.14 is not supported on this platform\n"


@pytest.mark.parametrize("output,system,machine,version,allowed", [
    (WARNING, "darwin", "arm64", "0.10.14", True),
    (WARNING + "numpy has incompatible dependencies\n", "darwin", "arm64", "0.10.14", False),
    (WARNING, "win32", "arm64", "0.10.14", False),
    (WARNING, "darwin", "x86_64", "0.10.14", False),
    (WARNING, "darwin", "arm64", "0.10.21", False),
    ("mediapipe requires numpy, which is not installed", "darwin", "arm64", "0.10.14", False),
])
def test_only_known_metadata_defect_is_allowed(output, system, machine, version, allowed):
    assert check_dependencies.known_macos_metadata_warning(output, system, machine, version) == allowed


def test_metadata_exception_requires_native_tracking(monkeypatch):
    monkeypatch.setattr(check_dependencies.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=WARNING))
    monkeypatch.setattr(check_dependencies.sys, "platform", "darwin")
    monkeypatch.setattr(check_dependencies.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(check_dependencies, "version", lambda name: "0.10.14")

    def broken_native_runtime():
        raise RuntimeError("native model could not initialize")

    monkeypatch.setattr(check_dependencies, "verify_native_tracking", broken_native_runtime)
    with pytest.raises(RuntimeError, match="native model could not initialize"):
        check_dependencies.main()
