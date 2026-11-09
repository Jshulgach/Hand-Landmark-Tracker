"""Camera preview must survive absent/rejected geometry without publishing 3D data."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from types import SimpleNamespace
import time

import numpy as np
import pytest
from unity_hand_tracking.optitrack_cam_py import mocap_tracker as backend


POINTS = np.array([
    [0,0,1.2],[.01,.01,1.18],[.025,.025,1.16],[.04,.04,1.15],[.055,.055,1.14],
    [.02,.055,1.2],[.02,.085,1.19],[.02,.11,1.18],[.02,.13,1.17],
    [0,.065,1.21],[0,.1,1.2],[0,.13,1.19],[0,.15,1.18],
    [-.02,.06,1.22],[-.02,.095,1.21],[-.02,.12,1.20],[-.02,.14,1.19],
    [-.04,.05,1.23],[-.04,.075,1.22],[-.04,.095,1.21],[-.04,.11,1.20],
])


def write_calibration(path, **overrides):
    data = dict(schema_version=1, num_cameras=2, camera_ids=np.arange(2),
                image_sizes=np.array([[640,480],[640,480]]), quality_passed=True,
                coordinate_units='meters')
    for index in range(2):
        data[f'camera_matrix_{index}'] = np.array([[500.,0,320],[0,500,240],[0,0,1]])
        data[f'dist_coeffs_{index}'] = np.zeros(5)
        data[f'R_{index}'] = np.eye(3)
        data[f'T_{index}'] = np.array([-.15*index,0,0])
    data.update(overrides)
    np.savez(path, **data)


@pytest.fixture
def rig(monkeypatch, tmp_path):
    calibration = tmp_path/'calibration.npz'
    state = SimpleNamespace(managers=[], detectors=[], results=[])
    for index in range(2):
        camera_points = POINTS + np.array([-.15*index,0,0])
        pixels = camera_points[:,:2]/camera_points[:,2,None]*500 + [320,240]
        landmarks = [SimpleNamespace(x=float(x/640), y=float(y/480), z=0.) for x,y in pixels]
        state.results.append(SimpleNamespace(
            multi_hand_landmarks=[SimpleNamespace(landmark=landmarks)],
            multi_handedness=[SimpleNamespace(classification=[SimpleNamespace(score=.99,label='Right')])],
        ))

    class Manager:
        num_cameras = 2
        camera_indices = [0,1]
        def __init__(self):
            self.started=False
            self.stopped=False
            self.exposures=[]
            self.frames=[np.full((480,640,3),30,np.uint8) for _ in range(2)]
            state.managers.append(self)
        def start_all(self):self.started=True
        def wait_ready(self):pass
        def get_resolution(self,index):return (640,480)
        def get_fresh_frames(self):return self.frames
        def set_exposure(self,index,value):self.exposures.append((index,value))
        def stop_all(self):self.stopped=True

    class Detector:
        def __init__(self,**kwargs):
            self.index=len(state.detectors)%2
            self.closed=False
            state.detectors.append(self)
        def process(self,image):return state.results[self.index]
        def close(self):self.closed=True

    monkeypatch.setattr(backend,'CALIBRATION_FILE',str(calibration))
    monkeypatch.setattr(backend,'CameraManager',Manager)
    monkeypatch.setattr(backend.MultiCameraTracker,'_load_mediapipe_hands_class',staticmethod(lambda:Detector))
    return calibration,state


@pytest.mark.parametrize('condition',['missing','corrupt','count','resolution','quality','object-array'])
def test_unusable_calibration_keeps_live_frames_and_2d_hands(monkeypatch, rig, condition):
    calibration,state=rig
    if condition=='corrupt':calibration.write_bytes(b'not a numeric calibration')
    elif condition=='count':write_calibration(calibration,num_cameras=3)
    elif condition=='resolution':write_calibration(calibration,image_sizes=np.array([[1280,1024]]*2))
    elif condition=='quality':write_calibration(calibration,quality_passed=False)
    elif condition=='object-array':np.savez(calibration,unsafe=np.array([{}],dtype=object))
    tracker=backend.MultiCameraTracker()
    try:
        assert tracker.initialize_cameras()
        assert state.managers[-1].started and not state.managers[-1].stopped
        assert not tracker.calibration_available and tracker.calibration_error
        def no_geometry(*args,**kwargs):raise AssertionError('Uncalibrated preview used camera geometry')
        monkeypatch.setattr(tracker,'extract_landmarks_2d',no_geometry)
        monkeypatch.setattr(tracker,'triangulate_hand',no_geometry)
        for _ in range(3):
            frames,hands3d,results,valid=tracker.process_frame()
            assert len(frames)==2 and all(frame.shape==(480,640,3) for frame in frames)
            assert all(result.multi_hand_landmarks for result in results)
            assert hands3d==[] and valid=={0:{},1:{}}
        assert tracker.get_fps_stats()['triangulation_ms']==0
    finally:
        tracker.cleanup()
    assert state.managers[-1].stopped and all(det.closed for det in state.detectors)


def test_calibration_restored_on_restart_and_stale_geometry_is_cleared(rig):
    calibration,state=rig
    tracker=backend.MultiCameraTracker()
    try:
        assert tracker.initialize_cameras() and not tracker.calibration_available
        tracker.cleanup()
        write_calibration(calibration)
        assert tracker.initialize_cameras() and tracker.calibration_available
        frames,hands3d,results,valid=tracker.process_frame()
        assert len(hands3d)==1
        np.testing.assert_allclose(hands3d[0][0],POINTS,atol=1e-5)
        tracker.cleanup()
        calibration.unlink()
        assert tracker.initialize_cameras() and not tracker.calibration_available
        assert tracker.projection_matrices==[] and tracker.camera_matrices==[]
        assert tracker.process_frame()[1]==[]
    finally:
        tracker.cleanup()
    assert all(manager.stopped for manager in state.managers)


def test_explicit_calibration_load_remains_strict_and_discards_old_geometry(rig):
    calibration,state=rig
    write_calibration(calibration)
    tracker=backend.MultiCameraTracker()
    try:
        assert tracker.initialize_cameras() and tracker.calibration_available
        write_calibration(calibration,num_cameras=3)
        with pytest.raises(ValueError,match='cameras'):
            tracker.load_calibration()
        assert not tracker.calibration_available and tracker.projection_matrices==[]
        assert not state.managers[-1].stopped
        assert tracker.process_frame()[1]==[]
    finally:
        tracker.cleanup()


def test_camera_errors_still_propagate_and_release_hardware(monkeypatch,rig):
    calibration,state=rig
    def fail(self):raise RuntimeError('USB acquisition failed')
    monkeypatch.setattr(backend.CameraManager,'start_all',fail)
    tracker=backend.MultiCameraTracker()
    with pytest.raises(RuntimeError,match='USB acquisition failed'):
        tracker.initialize_cameras()
    assert state.managers[-1].stopped


def test_gui_preview_renders_without_3d_then_enables_it_after_calibration(monkeypatch,rig):
    from PyQt5.QtWidgets import QApplication
    from unity_hand_tracking.optitrack_cam_py import mocap_handtrack_gui as gui
    calibration,state=rig
    recorded=[]
    class Broadcaster:
        def send_landmarks(self,*packet):recorded.append(('landmarks',packet[-1]))
        def send_angles(self,*packet):recorded.append(('angles',packet[-1]))
    broadcaster=Broadcaster()
    monkeypatch.setattr(gui.StereoHandTrackerGUI,'start_udp_broadcast',lambda self:None)
    monkeypatch.setattr(gui.StereoHandTrackerGUI,'stop_udp_broadcast',lambda self:None)
    monkeypatch.setattr(gui,'LSLBroadcaster',lambda **kwargs:broadcaster)
    monkeypatch.setattr(gui,'CALIBRATION_FILE',str(calibration))
    app=QApplication.instance() or QApplication([])
    window=gui.StereoHandTrackerGUI()
    def wait_for_frames():
        deadline=time.monotonic()+5
        while window.frame_count<2 and time.monotonic()<deadline:
            app.processEvents()
            time.sleep(.002)
        assert window.frame_count>=2,window.status_label.text()
    try:
        window.show()
        assert not window.show_3d_checkbox.isEnabled()
        window.start_tracking()
        wait_for_frames()
        assert window.is_running and not state.managers[-1].stopped
        assert '2D only' in window.status_label.text()
        assert 'unavailable' in window.calibration_label.text()
        assert str(calibration) in window.calibration_label.toolTip()
        assert all(label.pixmap() is not None for label in window.video_labels)
        assert all(not cb.isEnabled() for cb in window.cam_enabled_checkboxes)
        assert window.exposure_spinboxes[0].isEnabled() and window.min_det_spin.isEnabled()
        assert not window.views_3d_group.isEnabled() and not window.smoothing_group.isEnabled()
        window.show_3d_checkbox.setChecked(True)
        assert not window.show_3d_checkbox.isChecked() and not window.vis_3d_label.isVisible()
        assert recorded and all(hands==[] for kind,hands in recorded)
        assert '2D hands/view: 1'==window.hands_label.text()
        window.stop_tracking()
        write_calibration(calibration)
        recorded.clear()
        window.start_tracking()
        wait_for_frames()
        assert window.tracker.calibration_available and window.show_3d_checkbox.isEnabled()
        assert window.views_3d_group.isEnabled() and window.smoothing_group.isEnabled()
        assert all(cb.isEnabled() for cb in window.cam_enabled_checkboxes)
        assert 'ready' in window.calibration_label.text()
        window.show_3d_checkbox.setChecked(True)
        app.processEvents()
        assert window.vis_3d_label.isVisible()
        assert any(kind=='landmarks' and hands for kind,hands in recorded)
    finally:
        window.close()
        app.processEvents()
    assert all(manager.stopped for manager in state.managers)


def test_preview_startup_log_supports_windows_legacy_console_encoding(rig):
    import contextlib
    import io
    tracker=backend.MultiCameraTracker()
    try:
        with io.TextIOWrapper(io.BytesIO(),encoding='cp1252') as console:
            with contextlib.redirect_stdout(console):
                assert tracker.initialize_cameras()
        assert not tracker.calibration_available
        assert tracker.process_frame()[1]==[]
    finally:
        tracker.cleanup()


def test_camera_only_start_does_not_load_mediapipe_and_can_enable_later(monkeypatch, rig):
    _, state = rig
    original = backend.MultiCameraTracker._load_mediapipe_hands_class
    def unavailable():
        raise AssertionError('Camera-only startup loaded inference')
    monkeypatch.setattr(backend.MultiCameraTracker, '_load_mediapipe_hands_class', staticmethod(unavailable))
    tracker = backend.MultiCameraTracker()
    try:
        assert tracker.initialize_cameras(inference_enabled=False)
        assert state.managers[-1].started and not state.detectors
        assert all(frame is not None for frame in tracker.capture_frames())
        monkeypatch.setattr(backend.MultiCameraTracker, '_load_mediapipe_hands_class', staticmethod(original))
        tracker.initialize_inference()
        assert len(state.detectors) == 2
        assert all(result is not None for result in tracker.process_frame()[2])
    finally:
        tracker.cleanup()
