import io
import json
from urllib.error import HTTPError, URLError

import pytest

from scripts import check_release


@pytest.fixture
def release_files(tmp_path):
    for name in ("mavis_track-0.1.0-py3-none-any.whl", "mavis_track-0.1.0.tar.gz"):
        (tmp_path / name).touch()
    return tmp_path


def response(monkeypatch, filenames):
    def fetch(*args, **kwargs):
        return io.BytesIO(json.dumps({"urls": [{"filename": name} for name in filenames]}).encode())
    monkeypatch.setattr(check_release, "urlopen", fetch)


def test_complete_release_skips_upload(monkeypatch, release_files):
    response(monkeypatch, [path.name for path in release_files.iterdir()])
    assert check_release.already_published("mavis-track", "0.1.0", release_files, "pypi")


def test_partial_release_still_needs_upload(monkeypatch, release_files):
    response(monkeypatch, ["mavis_track-0.1.0-py3-none-any.whl"])
    assert not check_release.already_published("mavis-track", "0.1.0", release_files, "testpypi")


@pytest.mark.parametrize("error", [HTTPError("url", 404, "missing", None, None),
                                   HTTPError("url", 503, "unavailable", None, None),
                                   URLError("network unavailable")])
def test_only_not_found_allows_upload(monkeypatch, release_files, error):
    def fetch(*args, **kwargs):
        raise error
    monkeypatch.setattr(check_release, "urlopen", fetch)
    if isinstance(error, HTTPError) and error.code == 404:
        assert not check_release.already_published("mavis-track", "0.1.0", release_files, "pypi")
    else:
        with pytest.raises(type(error)):
            check_release.already_published("mavis-track", "0.1.0", release_files, "pypi")
