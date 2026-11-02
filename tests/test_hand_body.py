"""Public combined API and synchronized recording contracts, without a camera."""

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from mavis_track import (
    BODY_LANDMARK_NAMES, QUALITY_REGION_NAMES, HandBodyTracker, QualityFlag, load_session,
)
from handtrack.applications import _cli, _export, _record, _replay
from test_hand_tracker import FakeCapture


def landmarks(count, x=0.2, visibility=1.0):
    return SimpleNamespace(landmark=[SimpleNamespace(
        x=x + i * .001, y=.3 + i * .002, z=-i * .001, visibility=visibility,
    ) for i in range(count)])


def observation(left=True, right=True, body=True, world=True):
    return SimpleNamespace(
        left_hand_landmarks=landmarks(21, .2) if left else None,
        right_hand_landmarks=landmarks(21, .7) if right else None,
        pose_landmarks=landmarks(33) if body else None,
        pose_world_landmarks=landmarks(33, 2.0) if body and world else None,
    )


def install_model(monkeypatch, outputs):
    import mediapipe as mp
    outputs = iter(outputs)
    state = dict(calls=0, closed=False, options=None)

    def process(image):
        assert not image.flags.writeable
        state['calls'] += 1
        value = next(outputs)
        if isinstance(value, Exception):
            raise value
        return value

    def create(**kwargs):
        state['options'] = kwargs
        return SimpleNamespace(process=process, close=lambda: state.update(closed=True))

    monkeypatch.setattr(mp.solutions.holistic, 'Holistic', create)
    return state


def test_combined_frame_owns_snapshots_and_uses_one_inference(monkeypatch):
    import cv2
    monkeypatch.setattr(cv2, 'VideoCapture', lambda *_: pytest.fail('API opened capture'))
    first_raw, second_raw = observation(), observation()
    first_raw.left_hand_landmarks.landmark[0].x = .1
    state = install_model(monkeypatch, [first_raw, second_raw])
    image = np.zeros((80, 120, 3), np.uint8)
    with HandBodyTracker(source_id='camera-a') as tracker:
        first = tracker.process(image, timestamp=1.25)
        tracker.process(image, color='rgb', timestamp=1.3)
    assert state['calls'] == 2 and state['closed']
    assert state['options']['smooth_landmarks'] is False
    assert first.timestamp == 1.25 and first.timestamp_clock == 'source'
    assert first.source_id == 'camera-a' and first.image_size == (120, 80)
    assert first.frame_id == 0 and first.body.landmarks.shape == (33, 3)
    assert first.body.landmark_names == BODY_LANDMARK_NAMES
    assert first.body.world_landmarks[0, 0] == 2
    assert first.quality.body == QualityFlag.OK  # World meters are not image bounds.
    for hand in first.hands:
        assert hand.world_landmarks is None and hand.angle_space == 'image_scaled'
        assert np.isnan(hand.handedness_score)  # Holistic supplies no hand score.
        assert not hand.landmarks.flags.writeable and not hand.raw_landmarks.flags.writeable
    assert first.hands[0].raw_landmarks[0, 0] == pytest.approx(.1)
    assert not first.body.landmarks.flags.writeable
    assert not first.body.visibility.flags.writeable
    assert not first.body.world_landmarks.flags.writeable
    assert image.flags.writeable and not image.any()


def test_low_visibility_invalid_data_and_dropout_are_explicit(monkeypatch):
    first = observation()
    first.pose_landmarks.landmark[11].visibility = .1
    first.left_hand_landmarks.landmark[1].x = 1.1
    invalid = observation()
    invalid.left_hand_landmarks.landmark[0].z = float('nan')
    invalid.pose_landmarks.landmark.pop()
    install_model(monkeypatch, [first, observation(False, False, False), invalid])
    image = np.zeros((20, 30, 3), np.uint8)
    with HandBodyTracker(smoothing=True) as tracker:
        result = tracker.process(image, timestamp=0)
        assert result.quality.body & QualityFlag.LOW_VISIBILITY
        assert result.quality.left_hand & QualityFlag.OUT_OF_FRAME
        assert result.body is not None and len(result.hands) == 2
        missing = tracker.process(image, timestamp=.1)
        assert missing.body is None and missing.hands == ()
        assert missing.quality.body == missing.quality.left_hand == missing.quality.right_hand == QualityFlag.MISSING
        bad = tracker.process(image, timestamp=.2)
        assert bad.body is None and len(bad.hands) == 1
        assert bad.quality.left_hand == QualityFlag.MISSING | QualityFlag.NONFINITE
        assert bad.quality.body == QualityFlag.MISSING | QualityFlag.NONFINITE


def test_reappearance_gap_and_wrist_jump_reset_hand_identity(monkeypatch):
    jumped = observation(right=False)
    jumped.left_hand_landmarks = landmarks(21, .6)
    install_model(monkeypatch, [observation(right=False), observation(False, False, False),
                               observation(right=False), observation(right=False), jumped])
    image = np.zeros((20, 30, 3), np.uint8)
    with HandBodyTracker(max_gap=.3, smoothing=True) as tracker:
        first = tracker.process(image, timestamp=0)
        tracker.process(image, timestamp=.1)
        returned = tracker.process(image, timestamp=.2)
        gap = tracker.process(image, timestamp=.8)
        jump = tracker.process(image, timestamp=.9)
    assert len({r.hands[0].track_id for r in (first, returned, gap, jump)}) == 4
    assert returned.quality.left_hand & QualityFlag.TRACKING_RESET
    assert gap.quality.left_hand & QualityFlag.TEMPORAL_GAP
    assert gap.quality.body & QualityFlag.TEMPORAL_GAP
    assert gap.quality.right_hand == QualityFlag.MISSING | QualityFlag.TEMPORAL_GAP
    assert jump.quality.left_hand & QualityFlag.TRACKING_RESET
    assert not jump.quality.left_hand & QualityFlag.TEMPORAL_GAP


def test_mirrored_input_labels_body_indices_and_quality_remain_consistent(monkeypatch):
    raw = observation(right=False)
    raw.left_hand_landmarks.landmark[0].x = 1.2
    install_model(monkeypatch, [raw])
    with HandBodyTracker(mirrored=True) as tracker:
        result = tracker.process(np.zeros((20, 30, 3), np.uint8))
    assert result.hands[0].handedness == 'Right'
    assert result.quality.left_hand == QualityFlag.MISSING
    assert result.quality.right_hand == QualityFlag.OUT_OF_FRAME
    assert result.body.landmarks[11, 0] == pytest.approx(raw.pose_landmarks.landmark[12].x)
    assert result.body.landmarks[12, 0] == pytest.approx(raw.pose_landmarks.landmark[11].x)
    assert result.body.world_landmarks[11, 0] == pytest.approx(raw.pose_world_landmarks.landmark[12].x)


@pytest.mark.parametrize('kwargs', [dict(confidence=-.1), dict(tracking_confidence=1.1),
                                   dict(visibility_threshold=float('nan')), dict(max_gap=0)])
def test_invalid_configuration_rejected_before_model_creation(monkeypatch, kwargs):
    state = install_model(monkeypatch, [])
    with pytest.raises(ValueError):
        HandBodyTracker(**kwargs)
    assert state['options'] is None


def test_input_clock_validation_and_error_cleanup(monkeypatch):
    state = install_model(monkeypatch, [observation(), RuntimeError('model error')])
    tracker = HandBodyTracker()
    image = np.zeros((20, 30, 3), np.uint8)
    with pytest.raises(ValueError, match='uint8'):
        tracker.process(image.astype(float))
    with pytest.raises(ValueError, match='color'):
        tracker.process(image, color='xyz')
    with pytest.raises(ValueError, match='nonnegative'):
        tracker.process(image, timestamp=float('nan'))
    tracker.process(image, timestamp=1)
    with pytest.raises(ValueError, match='increase'):
        tracker.process(image, timestamp=1)
    assert state['calls'] == 1
    with pytest.raises(RuntimeError, match='model error'):
        with tracker:
            tracker.process(image, timestamp=2)
    tracker.close()
    assert state['closed']
    with pytest.raises(RuntimeError, match='closed'):
        tracker.process(image)
    with pytest.raises(RuntimeError, match='closed'):
        tracker.__enter__()


@pytest.fixture
def combined_recording(monkeypatch, tmp_path):
    state = install_model(monkeypatch, [observation(), observation(False, False, False),
                                      observation(right=False, world=False)])
    capture = FakeCapture([np.zeros((64, 96, 3), np.uint8) for _ in range(3)], fps=29.97)
    monkeypatch.setattr(_record.cv2, 'VideoCapture', lambda _: capture)
    assert _cli.main(['record', '--mode', 'hand-body', '--source', 'sample.mp4',
                      '--session-name', 'combined', '--output-dir', str(tmp_path),
                      '--visibility-threshold', '.6', '--max-gap', '.4']) == 0
    assert state['calls'] == 3 and state['closed'] and capture._released
    return tmp_path / 'combined'


def test_synchronized_record_load_export_and_replay(combined_recording, monkeypatch):
    folder = combined_recording
    session = load_session(folder)
    assert session.tracking_mode == 'hand-body'
    assert session.body_landmarks.shape == (3, 33, 3)
    assert session.body_visibility.shape == (3, 33)
    assert session.landmarks.shape == (3, 2, 21, 3)
    assert session.quality_region_names == QUALITY_REGION_NAMES
    assert session.body_valid.tolist() == [True, False, True]
    assert np.allclose(session.time_vector, np.arange(3) / 29.97)
    assert np.isnan(session.body_landmarks[1]).all()
    assert np.isnan(session.body_world_landmarks[2]).all()
    assert np.isnan(session.landmarks[1]).all()
    assert session.quality_flags[1].tolist() == [1, 1, 1]
    assert session.quality_flags[2, 0] & QualityFlag.TRACKING_RESET
    assert not session.quality_flags.flags.writeable
    assert not session.body_landmarks.flags.writeable
    assert session.quality_visibility_threshold == .6 and session.quality_max_gap == .4
    manifest = json.loads((folder / 'session.json').read_text())
    assert manifest['schema_version'] == 2 and manifest['smoothing_scope'] == 'hands_only'
    assert manifest['quality_flag_definitions']['MISSING'] == 1
    assert _export.main([str(folder)]) == 0
    for filename, row_count in [('body_landmarks.csv', 3 * 33 + 1), ('tracking_quality.csv', 3 * 3 + 1)]:
        with (folder / 'exports' / filename).open(newline='') as handle:
            rows = list(csv.reader(handle))
        assert len(rows) == row_count and len({len(row) for row in rows}) == 1
        assert rows[1][1] == '0.000000'
    drawn = []
    monkeypatch.setattr(_replay, '_draw_body', lambda *args: drawn.append(args))
    assert _replay.main([str(folder), '--no-display']) == 0
    assert len(drawn) == 2 and all(args[3] == .6 for args in drawn)


@pytest.mark.parametrize('field,value', [
    ('body_landmarks', np.zeros((3, 32, 3))),
    ('body_valid', np.array([1, 2, 1])),
    ('body_visibility', np.full((3, 33), 1.5)),
    ('quality_flags', np.zeros((3, 3))),
    ('quality_flags', np.full((3, 3), 128, dtype=np.uint16)),
    ('quality_flags', np.zeros((3, 3), dtype=np.uint16)),
    ('body_world_coordinate_space', np.array('calibrated_world_meters')),
    ('quality_max_gap', np.array(float('nan'))),
    ('quality_region_names', np.array(['body', 'left_hand', 'right_hand'])),
])
def test_combined_session_rejects_corrupt_metadata(combined_recording, tmp_path, field, value):
    with np.load(combined_recording / 'landmarks.npz', allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    data[field] = value
    filename = tmp_path / 'corrupt.npz'
    np.savez_compressed(filename, **data)
    with pytest.raises(ValueError):
        load_session(filename)


def test_record_closes_capture_and_model_on_combined_failure(monkeypatch, tmp_path):
    state = install_model(monkeypatch, [RuntimeError('failed inference')])
    capture = FakeCapture([np.zeros((20, 30, 3), np.uint8)])
    monkeypatch.setattr(_record.cv2, 'VideoCapture', lambda _: capture)
    assert _record.main(['--mode', 'hand-body', '--source', 'input.mp4',
                         '--output-dir', str(tmp_path)]) == 1
    assert state['closed'] and capture._released


def test_invalid_world_and_visibility_preserve_usable_image_observations(monkeypatch):
    raw = observation()
    raw.pose_world_landmarks.landmark[11].x = float('inf')
    raw.pose_landmarks.landmark[12].visibility = float('nan')
    install_model(monkeypatch, [raw])
    with HandBodyTracker() as tracker:
        result = tracker.process(np.zeros((20, 30, 3), np.uint8))
    assert result.body is not None and np.isfinite(result.body.landmarks).all()
    assert result.body.world_landmarks is None
    assert np.isnan(result.body.visibility).all()
    assert result.quality.body == QualityFlag.NONFINITE | QualityFlag.LOW_VISIBILITY


def test_duplicate_anatomical_side_rejected(combined_recording, tmp_path):
    with np.load(combined_recording / 'landmarks.npz', allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    data['handedness'][0] = ['Left', 'Left']
    path = tmp_path / 'duplicate-side.npz'
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError, match='unique anatomical'):
        load_session(path)
