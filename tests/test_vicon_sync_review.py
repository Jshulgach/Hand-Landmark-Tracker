"""Real C3D/video round trips and explicit, loss-aware alignment evidence."""
import csv
import json

import numpy as np
import pytest

from mavis_track._camera_recording import CameraRecording
from mavis_track._sync_analysis import analyze_session, load_vicon, load_session, exposure_edges
from mavis_track._trial import trial_directory, trial_slug
from handtrack.applications._sync_replay import ReplayData


@pytest.fixture
def trial(tmp_path):
    ezc3d = pytest.importorskip('ezc3d')
    c3d = ezc3d.c3d()
    c3d['parameters']['POINT']['RATE']['value'] = [50.]
    c3d['parameters']['POINT']['LABELS']['value'] = ['marker']
    c3d['parameters']['POINT']['UNITS']['value'] = ['mm']
    c3d['data']['points'] = np.ones((4, 1, 40))
    c3d['data']['points'][:3] *= 1000
    c3d['data']['points'][0, 0] = np.arange(40)*10
    c3d['header']['points']['first_frame'] = 100
    c3d['parameters']['ANALOG']['RATE']['value'] = [1000.]
    c3d['parameters']['ANALOG']['LABELS']['value'] = ['Exposure']
    analog = np.zeros((1, 1, 800))
    for sample in range(10, 800, 20):
        analog[0, 0, sample:sample+5] = 4
    c3d['data']['analogs'] = analog
    c3d_path = tmp_path/'Trial001.c3d'
    c3d.write(str(c3d_path))
    session = tmp_path/'videos'
    recorder = CameraRecording(session, nominal_fps=50, trial_name='Reach 01',
        camera_ids=[0], camera_settings=[dict(camera_index=0, serial=123, width=64, height=48)])
    ids = [100, 101, 102, 104, 108, 109, 110]
    for frame_id in ids:
        recorder.frame(0, np.full((48, 64, 3), (frame_id-100)*10, np.uint8),
                       dict(sdk_frame_id=frame_id, host_received_monotonic=100+(frame_id-100)/50))
    recorder.close()
    first = load_vicon(c3d_path)['first_frame']
    return session, c3d_path, first, ids


def anchors(path, first, *, checks=True, check_error=0):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['camera_index', 'sdk_frame_id', 'vicon_frame', 'role'])
        writer.writeheader()
        for sdk, role in [(102, 'fit'), (108, 'fit')] + ([(100, 'check'), (110, 'check')] if checks else []):
            writer.writerow(dict(camera_index=0, sdk_frame_id=sdk,
                                 vicon_frame=first+sdk-100+(check_error if role == 'check' else 0), role=role))
    return path


def test_timestamped_trial_files_and_diagnostics_preserve_sdk_gaps(trial):
    session, c3d, first, ids = trial
    manifest, frames = load_session(session)
    name = manifest['video_files']['0']
    assert name.startswith('Reach_01_') and 'Z_camera_0.avi' in name
    assert manifest['trial_name'] == 'Reach 01'
    assert manifest['frame_diagnostics']['0']['sdk_missing'] == 4
    assert (session/'diagnostics.json').is_file() and (session/'events.jsonl').is_file()
    assert frames[0][3]['sdk_missing_since_previous_written'] == 1


def test_trial_names_are_portable_and_directories_never_collide(tmp_path):
    assert trial_slug('../CON') == 'trial_CON'
    assert '/' not in trial_slug('subject/../reach')
    one, two = trial_directory(tmp_path, 'Reach'), trial_directory(tmp_path, 'Reach')
    assert one != two and one.parent == two.parent == tmp_path
    assert 'Z_' in one.name
    with pytest.raises(ValueError): trial_slug('...')


def test_c3d_preserves_export_range_units_and_extracts_exposure_edges(trial):
    session, path, first, ids = trial
    vicon = load_vicon(path)
    assert first == 100 and vicon['last_frame'] == 139
    assert vicon['points'][0, 0, 2] == 1.
    edges = exposure_edges(vicon, 'Exposure', 2.)
    assert edges[0]['analog_sample'] == 10
    assert edges[0]['vicon_frame'] == 100.5
    with pytest.raises(ValueError): exposure_edges(vicon, 'missing', 2.)


def test_explicit_anchors_map_gaps_and_check_independent_events(trial, tmp_path):
    session, c3d, first, ids = trial
    anchor_path = anchors(tmp_path/'anchors.csv', first)
    report = analyze_session(session, vicon_path=c3d, anchors_path=anchor_path,
                             exposure_channel='Exposure')
    stats = report['cameras']['0']
    assert stats['status'] == 'checked' and stats['check_max_error_ms'] < 1e-6
    assert report['diagnostics']['0']['sdk_missing'] == 4
    with ReplayData(session, c3d) as replay:
        assert replay.image(0, first) is not None
        assert replay.image(0, first+3) is None  # SDK 103 was never saved; do not hold frame 102.
        assert replay.mapping[0][first+8][0] == 4  # Missing images never shift later correspondence.
    assert (session/'alignment'/'exposure_edges.csv').is_file()


def test_no_anchors_and_fit_only_do_not_claim_validation(trial, tmp_path):
    session, c3d, first, ids = trial
    report = analyze_session(session, vicon_path=c3d)
    assert report['cameras']['0']['status'] == 'unmapped'
    report = analyze_session(session, vicon_path=c3d, anchors_path=anchors(tmp_path/'anchors.csv', first, checks=False))
    assert report['cameras']['0']['status'] == 'unvalidated'


def test_sdk_reset_prevents_automatic_segment_alignment(trial, tmp_path):
    session, c3d, first, ids = trial
    path = session/'frames.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[3]['sdk_frame_id'] = 0
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    report = analyze_session(session, vicon_path=c3d, anchors_path=anchors(tmp_path/'anchors.csv', first))
    assert report['cameras']['0']['status'] == 'unmapped'
    assert 'reset' in report['cameras']['0']['reason']
    assert report['diagnostics']['0']['sdk_resets_or_duplicates'] == 1


def test_hardware_ticks_are_fitted_in_their_own_clock_domain(trial, tmp_path):
    session, c3d, first, ids = trial
    path = session/'frames.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row.update(hardware_timestamp_valid=True, hardware_timestamp_frequency=1000000,
                   hardware_timestamp_ticks=900000000 + (row['sdk_frame_id']-100)*20002)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    report = analyze_session(session, vicon_path=c3d, anchors_path=anchors(tmp_path/'anchors.csv', first))
    assert report['cameras']['0']['basis'] == 'hardware_clock'
    assert report['cameras']['0']['drift_ppm'] == pytest.approx(-99.99, abs=.02)


def test_failed_independent_check_and_changed_mapping_are_visible(trial, tmp_path):
    session, c3d, first, ids = trial
    report = analyze_session(session, vicon_path=c3d,
        anchors_path=anchors(tmp_path/'anchors.csv', first, check_error=1))
    assert report['cameras']['0']['status'] == 'unvalidated'
    assert report['cameras']['0']['check_max_error_ms'] == pytest.approx(20)
    mapping = session/'alignment'/'frame_mapping.csv'
    mapping.write_text(mapping.read_text()+'\n')
    with pytest.raises(ValueError, match='changed'):
        ReplayData(session, c3d)


def test_projection_requires_physical_camera_identity_and_valid_transform(trial, tmp_path):
    session, c3d, first, ids = trial
    analyze_session(session, vicon_path=c3d, anchors_path=anchors(tmp_path/'anchors.csv', first))
    data = dict(schema='mavis-vicon-projection-v1', coordinate_units='meters', cameras={'0': dict(
        serial=123, image_size=[64, 48], camera_matrix=[[50, 0, 32], [0, 50, 24], [0, 0, 1]],
        dist_coeffs=[0]*5, vicon_to_camera=np.eye(4).tolist())})
    path = tmp_path/'projection.json'
    path.write_text(json.dumps(data))
    with ReplayData(session, c3d, projection=path) as replay:
        assert replay.image(0, first) is not None
    data['cameras']['0']['serial'] = 999
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='serial'):
        ReplayData(session, c3d, projection=path)


def test_sync_commands_route_without_starting_cameras(trial, tmp_path):
    from handtrack.applications._cli import main
    session, c3d, first, ids = trial
    anchor_path = anchors(tmp_path/'anchors.csv', first)
    assert main(['sync-report', str(session), '--vicon', str(c3d), '--anchors', str(anchor_path)]) == 0
    assert main(['sync-replay', str(session), '--vicon', str(c3d), '--no-display']) == 0


def test_separate_review_window_renders_and_closes_offscreen(trial, tmp_path, monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QApplication, QLabel
    from handtrack.applications._sync_replay import show_replay
    session, c3d, first, ids = trial
    analyze_session(session, vicon_path=c3d, anchors_path=anchors(tmp_path/'anchors.csv', first))
    app = QApplication.instance() or QApplication([])
    observed = []
    def finish():
        for widget in app.topLevelWidgets():
            if widget.windowTitle().startswith('Synchronized review'):
                observed.extend(label.pixmap() is not None for label in widget.findChildren(QLabel))
                widget.close()
        app.quit()
    QTimer.singleShot(50, finish)
    with ReplayData(session, c3d) as replay:
        assert show_replay(replay) == 0
    assert sum(observed) >= 2
