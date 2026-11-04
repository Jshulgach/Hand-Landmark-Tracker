"""Keyboard exposure and calibration capture controls must preserve safe publication."""
from types import SimpleNamespace
import threading

import numpy as np
import pytest
from handtrack.applications._calibration_preview import CalibrationControls
from handtrack.applications import _calibration_capture as capture


class ExposureManager:
    num_cameras=4
    def __init__(self):self.requests=[]
    def set_exposure(self,index,value):self.requests.append((index,value))


def test_keyboard_exposure_targets_and_limits(monkeypatch):
    manager=ExposureManager()
    controls=CalibrationControls(manager)
    assert controls.handle_key('up')
    assert manager.requests==[(i,11) for i in range(4)]
    controls.handle_key('2')
    controls.handle_key('down',shift=True)
    assert controls.exposures==[11,11,1,11]
    manager.requests.clear()
    controls.handle_key('down')
    assert manager.requests==[]
    assert not controls.handle_key('9')
    controls.handle_key('a')
    controls.adjust(1000)
    assert controls.exposures==[480]*4
    assert not controls.capture_ready
    controls.last_change=float('-inf')
    assert controls.capture_ready
    controls.handle_key('space')
    assert not controls.capture_ready
    controls.handle_key('space')
    assert controls.capture_ready


def test_exposure_failure_keeps_prior_value_and_reports_camera():
    manager=ExposureManager()
    def fail(index,value):raise RuntimeError('device unavailable')
    manager.set_exposure=fail
    controls=CalibrationControls(manager)
    controls.handle_key('1')
    controls.handle_key('up')
    assert controls.exposures==[10]*4
    assert 'Camera 1' in controls.last_error
    assert not controls.quit_requested


@pytest.mark.parametrize('action,should_save',[('q',True),('escape',False)])
def test_finish_solves_but_cancel_preserves_previous_calibration(monkeypatch,tmp_path,action,should_save):
    published=tmp_path/'calibration.npz'
    published.write_bytes(b'previous validated result')
    calls=[]
    ticks=iter(range(100000))
    monkeypatch.setattr(capture.time,'monotonic',lambda:float(next(ticks)))
    class Manager(ExposureManager):
        num_cameras=2
        def __init__(self):super().__init__();self.stopped=False
        def start_all(self,**kwargs):pass
        def wait_ready(self):pass
        def get_resolution(self,index):return (100,80)
        def get_all_frames(self):return [np.zeros((80,100,3),np.uint8)]*2
        def stop_all(self):self.stopped=True
    manager=Manager()
    class Preview:
        def __init__(self):self.controls=CalibrationControls(manager)
        def show_frames(self,frames,status):
            if status.startswith('Captured 10/'):
                self.controls.handle_key(action)
        def shutdown(self):pass
    monkeypatch.setattr(capture,'create_calibration_preview',lambda *args:Preview())
    corners=np.zeros((6,1,2),np.float32)
    ids=np.arange(6,dtype=np.int32).reshape(-1,1)
    def intrinsics(all_corners,all_ids,board,size,count):
        assert all(len(values)==10 for values in all_corners)
        calls.append('intrinsics')
        return [np.eye(3)]*2,[np.zeros(5)]*2
    def stereo(*args):
        calls.append('stereo')
        return [np.eye(3)]*2,[np.zeros(3)]*2
    def save(*args):calls.append('save');published.write_bytes(b'validated replacement')
    detectors=[]
    used=set()
    def build(**kwargs):
        detector=object()
        detectors.append(detector)
        return object(),detector
    def detect(args):
        used.add(args[1])
        return True,(corners,ids)
    module=SimpleNamespace(CameraManager=lambda:manager,CHARUCO_SQUARES_X=7,CHARUCO_SQUARES_Y=5,
        NUM_CALIBRATION_IMAGES=20,auto_detect_dictionary=lambda frame:('DICT_5X5_250',6),
        build_charuco_detector=build,
        detect_charuco=detect,annotate_frame=lambda frame,*args:frame,
        calibrate_intrinsics=intrinsics,calibrate_stereo_pairs=stereo,save_calibration=save)
    capture.run_capture(module)
    assert manager.stopped
    assert len(detectors)==2 and used==set(detectors)
    assert calls==(['intrinsics','stereo','save'] if should_save else [])
    assert published.read_bytes()==(b'validated replacement' if should_save else b'previous validated result')


def test_detection_error_releases_cameras_without_solving(monkeypatch):
    class Manager(ExposureManager):
        num_cameras=1
        stopped=False
        def start_all(self,**kwargs):pass
        def get_resolution(self,index):return (10,8)
        def get_all_frames(self):return [np.zeros((8,10,3),np.uint8)]
        def stop_all(self):self.stopped=True
    manager=Manager()
    controls=CalibrationControls(manager)
    preview=SimpleNamespace(controls=controls,show_frames=lambda *args:None,shutdown=lambda:None)
    monkeypatch.setattr(capture,'create_calibration_preview',lambda *args:preview)
    def fail(frame):raise RuntimeError('detector failed')
    module=SimpleNamespace(CameraManager=lambda:manager,CHARUCO_SQUARES_X=7,CHARUCO_SQUARES_Y=5,
        NUM_CALIBRATION_IMAGES=20,auto_detect_dictionary=fail)
    with pytest.raises(RuntimeError,match='detector failed'):
        capture.run_capture(module)
    assert manager.stopped


def test_calibration_window_routes_real_key_events_without_camera_hardware(monkeypatch):
    from PyQt5.QtCore import Qt,QEvent
    from PyQt5.QtGui import QKeyEvent
    from PyQt5.QtWidgets import QApplication
    from handtrack.applications._calibration_qt import CalibrationPreview
    manager=ExposureManager()
    controls=CalibrationControls(manager)
    app=QApplication.instance() or QApplication([])
    window=CalibrationPreview(controls,'Calibration keyboard test')
    def key(code,text='',modifiers=Qt.NoModifier):
        app.sendEvent(window,QKeyEvent(QEvent.KeyPress,code,modifiers,text))
    try:
        window.show_frames([np.zeros((80,100,3),np.uint8)]*4,'Show the board')
        assert window.font().pointSizeF()>=11
        assert 'Up/Down' in window.help.text() and 'Esc' in window.help.text()
        key(Qt.Key_3,'3')
        key(Qt.Key_Up,modifiers=Qt.ShiftModifier)
        assert manager.requests==[(3,20)]
        key(Qt.Key_A,'a')
        key(Qt.Key_Down)
        assert controls.exposures==[9,9,9,19]
        key(Qt.Key_Space,' ')
        assert controls.paused
        assert not controls.capture_ready
        key(Qt.Key_Escape)
        assert controls.cancelled
        assert all(label.pixmap() is not None for label in window.labels)
    finally:
        window.shutdown()
