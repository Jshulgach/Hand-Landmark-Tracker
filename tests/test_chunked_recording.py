"""Long-session streaming, checkpoint recovery and compatibility contracts."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from mavis_track import iter_session_chunks, load_session
from mavis_track import session as session_module
from mavis_track._recording import SessionWriter
from handtrack.applications import _record, _export, _replay
from test_hand_tracker import FakeCapture
from test_hand_body import observation, install_model


def record(monkeypatch, tmp_path, *, count=9, failure=None):
    state = install_model(monkeypatch, [observation() for _ in range(count)])
    capture = FakeCapture([np.zeros((32, 48, 3), np.uint8) for _ in range(count)], fps=29.97)
    monkeypatch.setattr(_record.cv2, 'VideoCapture', lambda _: capture)
    if failure is not None:
        from mavis_track import HandBodyTracker
        original = HandBodyTracker.process
        calls = [0]
        def process(self, *args, **kwargs):
            calls[0] += 1
            if calls[0] == 4:
                raise failure
            return original(self, *args, **kwargs)
        monkeypatch.setattr(HandBodyTracker, 'process', process)
    code = _record.main(['--mode', 'hand-body', '--source', 'input.mp4', '--output-dir', str(tmp_path),
                         '--session-name', 'long', '--chunk-frames', '2'])
    assert state['closed'] and capture._released
    return code, tmp_path / 'long'


def test_bounded_chunks_preserve_timing_ids_and_stream_exports(monkeypatch, tmp_path):
    pending = []
    original = SessionWriter.append
    def append(self, value):
        original(self, value)
        pending.append(len(self.buffer.get('time_vector', [])))
    monkeypatch.setattr(SessionWriter, 'append', append)
    code, folder = record(monkeypatch, tmp_path)
    assert code == 0 and max(pending) < 2
    manifest = json.loads((folder / 'session.json').read_text())
    assert manifest['schema_version'] == 3 and manifest['complete'] is True
    assert [entry['frame_count'] for entry in manifest['chunks']] == [2, 2, 2, 2, 1]
    assert not (folder / 'landmarks.npz').exists()
    chunks = list(iter_session_chunks(folder))
    combined = load_session(folder)
    assert combined.body_landmarks.shape == (9, 33, 3)
    assert np.allclose(combined.time_vector, np.arange(9) / 29.97)
    assert all(np.array_equal(chunk.track_ids[0], combined.track_ids[0]) for chunk in chunks)
    assert np.array_equal(np.concatenate([c.body_visibility for c in chunks]), combined.body_visibility)
    assert not combined.time_vector.flags.writeable
    assert not combined.quality_flags.flags.writeable
    from handtrack.io import SessionLoader
    legacy_points, legacy_rate, legacy_times = SessionLoader(str(folder), label="").load_landmarks()
    assert legacy_points.shape == combined.landmarks.shape
    assert legacy_rate == combined.sampling_rate and np.array_equal(legacy_times, combined.time_vector)
    with pytest.raises(ValueError, match='iter_session_chunks'):
        load_session(folder, max_bytes=100)
    # Streaming consumers must not call the whole-session materializer.
    monkeypatch.setattr(session_module, 'load_session', lambda *_a, **_k: pytest.fail('materialized whole session'))
    assert _export.main([str(folder)]) == 0
    assert _replay.main([str(folder), '--no-display']) == 0
    with (folder / 'exports' / 'body_landmarks.csv').open(newline='') as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 1 + 9 * 33
    assert sorted({int(row[0]) for row in rows[1:]}) == list(range(9))
    assert json.loads((folder / 'exports' / 'export_manifest.json').read_text())['frame_count'] == 9


@pytest.mark.parametrize('failure,code', [(RuntimeError('model failed'), 1), (KeyboardInterrupt(), 130)])
def test_interrupted_run_keeps_completed_frames(monkeypatch, tmp_path, failure, code):
    result, folder = record(monkeypatch, tmp_path, failure=failure)
    assert result == code
    manifest = json.loads((folder / 'session.json').read_text())
    assert manifest['complete'] is False and manifest['frame_count'] == 3
    assert [len(c.time_vector) for c in iter_session_chunks(folder)] == [2, 1]
    assert len(load_session(folder).time_vector) == 3


def test_write_failure_leaves_last_committed_checkpoint_readable(monkeypatch, tmp_path):
    from mavis_track import _recording
    save = _recording.np.savez_compressed
    def fail(path, **data):
        if Path(path).name == '000002.tmp.npz':
            raise OSError('disk full')
        return save(path, **data)
    monkeypatch.setattr(_recording.np, 'savez_compressed', fail)
    code, folder = record(monkeypatch, tmp_path)
    assert code == 1
    manifest = json.loads((folder / 'session.json').read_text())
    assert manifest['frame_count'] == 4 and manifest['complete'] is False
    assert len(load_session(folder).time_vector) == 4


@pytest.mark.parametrize('corruption', ['traversal', 'count', 'overlap', 'layout', 'missing'])
def test_corrupt_chunked_sessions_are_rejected(monkeypatch, tmp_path, corruption):
    _, folder = record(monkeypatch, tmp_path, count=4)
    path = folder / 'session.json'
    manifest = json.loads(path.read_text())
    if corruption == 'traversal':
        manifest['chunks'][0]['file'] = '../outside.npz'
    elif corruption == 'count':
        manifest['frame_count'] += 1
    else:
        chunk = folder / manifest['chunks'][1]['file']
        if corruption == 'missing':
            chunk.unlink()
        else:
            with np.load(chunk, allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
            if corruption == 'overlap':
                data['time_vector'] -= 1
            else:
                data['quality_max_gap'] = np.array(.9)
            np.savez_compressed(chunk, **data)
    path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, OSError)):
        list(iter_session_chunks(folder))


def test_short_recording_keeps_flat_format(monkeypatch, tmp_path):
    state = install_model(monkeypatch, [observation()])
    capture = FakeCapture([np.zeros((32, 48, 3), np.uint8)], fps=30)
    monkeypatch.setattr(_record.cv2, 'VideoCapture', lambda _: capture)
    assert _record.main(['--mode', 'hand-body', '--source', 'input.mp4', '--output-dir', str(tmp_path),
                         '--session-name', 'short']) == 0
    assert state['closed'] and (tmp_path / 'short' / 'landmarks.npz').is_file()
    assert len(list(iter_session_chunks(tmp_path / 'short'))) == 1
