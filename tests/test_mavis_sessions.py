import builtins
import io
import zipfile
import csv
from pathlib import Path
import numpy as np
import pytest

from mavis_track import load_session
from handtrack.applications import _record, _replay, _export
from test_hand_tracker import FakeCapture
from test_mavis_api import detection, results, solution


def test_record_export_replay_roundtrip_with_two_and_missing_hands(monkeypatch, tmp_path):
    a, b = detection(.2, "Left"), detection(.7, "Right")
    state = solution(monkeypatch, [results(), results(a, b), results(b, a), results()])
    cap = FakeCapture([np.zeros((64, 64, 3), np.uint8) for _ in range(4)], fps=29.97)
    monkeypatch.setattr(_record.cv2, "VideoCapture", lambda _: cap)
    assert _record.main(["--source", "sample.mp4", "--output-dir", str(tmp_path),
                         "--session-name", "demo", "--max-hands", "2"]) == 0
    session = load_session(tmp_path / "demo")
    assert session.landmarks.shape == (4, 2, 21, 3)
    assert session.valid.tolist() == [[False, False], [True, True], [True, True], [False, False]]
    assert np.array_equal(session.track_ids[1], session.track_ids[2])
    assert session.sampling_rate == 29.97
    assert session.image_sizes.tolist() == [[64, 64]] * 4
    assert np.allclose(np.diff(session.time_vector), 1 / 29.97)
    assert np.isnan(session.landmarks[~session.valid]).all()
    assert state["closed"] and cap._released
    assert _export.main([str(tmp_path / "demo")]) == 0
    for name in ("landmarks.csv", "angles.csv", "hands.csv"):
        with (tmp_path / "demo" / "exports" / name).open(newline="") as handle:
            rows = list(csv.reader(handle))
        assert len({len(row) for row in rows}) == 1
    assert _replay.main([str(tmp_path / "demo"), "--no-display"]) == 0
    assert _record.main(["--output-dir", str(tmp_path), "--session-name", "demo"]) == 1


class PickleProbe:
    def __init__(self, marker):
        self.marker = str(marker)
    def __reduce__(self):
        return builtins.eval, (f"__import__('pathlib').Path({self.marker!r}).touch()",)


def test_shared_object_array_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    np.savez(tmp_path / "landmarks.npz", landmarks=np.array([PickleProbe(marker)], dtype=object))
    with pytest.raises(ValueError, match="unsafe"):
        load_session(tmp_path)
    assert not marker.exists()
    assert _export.main([str(tmp_path)]) == 1
    assert not marker.exists()


@pytest.mark.parametrize("values", [dict(sampling_rate=0), dict(sampling_rate=float("nan")),
                                     dict(time_vector=[0, 0]), dict(time_vector=[0]),
                                     dict(landmarks=np.zeros((2, 22, 3)))])
def test_invalid_session_metadata_rejected(tmp_path, values):
    data = dict(landmarks=np.ones((2, 21, 3)), sampling_rate=29.97)
    data.update(values)
    np.savez(tmp_path / "landmarks.npz", **data)
    with pytest.raises(ValueError):
        load_session(tmp_path)


def test_capture_and_detector_close_when_processing_fails(monkeypatch, tmp_path):
    state = solution(monkeypatch, [results()])
    cap = FakeCapture([np.zeros((8, 8, 3), np.uint8)])
    monkeypatch.setattr(_record.cv2, "VideoCapture", lambda _: cap)
    from mavis_track import HandTracker
    monkeypatch.setattr(HandTracker, "process", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("device error")))
    assert _record.main(["--source", "input.mp4", "--output-dir", str(tmp_path)]) == 1
    assert cap._released and state["closed"]


def test_archive_declaring_huge_array_is_rejected_before_allocation(tmp_path):
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, dict(descr='<f4', fortran_order=False, shape=(10**12,)))
    with zipfile.ZipFile(tmp_path / 'landmarks.npz', 'w') as archive:
        archive.writestr('landmarks.npy', header.getvalue())
    with pytest.raises(ValueError, match='array size'):
        load_session(tmp_path)
