"""IR controls respect per-camera limits, hub capabilities, SDK failures and ownership."""
from types import SimpleNamespace
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')

import numpy as np
import pytest
from mavis_track._illumination import read_illumination, set_illumination
from handtrack.applications._illumination_controls import IlluminationControls
from handtrack.applications._calibration_preview import CalibrationControls


class Camera:
    def __init__(self,maximum=15,supported=True,available=True):
        self.level=4
        self.maximum=maximum
        self.supported=supported
        self.available=available
        self.power=False
        self.writes=[]
    def get_intensity(self):return self.level
    def min_intensity(self):return 0
    def max_intensity(self):return self.maximum
    def set_intensity(self,value):self.writes.append(('intensity',value));self.level=value
    def is_high_power_supported(self):return self.supported
    def is_high_power_available(self):return self.available
    def get_high_power(self):return self.power
    def set_high_power(self,value):self.writes.append(('power',value));self.power=value


class Manager:
    def __init__(self,cameras):self.cameras=cameras;self.num_cameras=len(cameras)
    def get_illumination(self,index):return read_illumination(self.cameras[index])
    def set_illumination(self,index,**kwargs):return set_illumination(self.cameras[index],**kwargs)
    def set_exposure(self,index,value):pass


def test_maximum_uses_reported_limit_and_available_power():
    camera=Camera(maximum=9)
    before=read_illumination(camera)
    assert before.intensity==4 and before.maximum==9 and not before.high_power
    assert camera.writes==[]
    after=set_illumination(camera,maximize=True)
    assert after.intensity==9 and after.high_power
    assert camera.writes==[('power',True),('intensity',9)]


@pytest.mark.parametrize('value',[-1,16,True,2.5])
def test_invalid_intensity_never_changes_power_or_level(value):
    camera=Camera()
    with pytest.raises(ValueError):set_illumination(camera,intensity=value,high_power=True)
    assert camera.writes==[]


def test_unsupported_or_unavailable_power_is_never_enabled():
    for camera in (Camera(supported=False),Camera(available=False)):
        with pytest.raises(RuntimeError):set_illumination(camera,high_power=True)
        assert camera.writes==[]
        result=set_illumination(camera,maximize=True)
        assert result.intensity==15
        assert camera.writes==[('intensity',15)]
    camera=Camera(available=False)
    camera.power=True
    assert set_illumination(camera,high_power=False).high_power is False


def test_readback_failure_is_explicit_and_actual_state_is_retained():
    camera=Camera()
    camera.set_intensity=lambda value:None
    with pytest.raises(RuntimeError,match='readback'):
        set_illumination(camera,intensity=12)
    assert read_illumination(camera).intensity==4


def test_old_bindings_keep_preview_usable_and_explain_upgrade():
    state=read_illumination(SimpleNamespace())
    assert not state.intensity_supported and not state.high_power_supported
    assert 'Binding upgrade' in state.message
    with pytest.raises(RuntimeError,match='upgrade'):
        set_illumination(SimpleNamespace(),maximize=True)


def test_all_camera_targets_skip_unavailable_and_report_individual_results():
    manager=Manager([Camera(9),Camera(7,supported=False),SimpleNamespace()])
    changed=[]
    controls=IlluminationControls(manager,lambda:changed.append(True))
    controls.apply([0,1,2],maximize=True)
    assert manager.cameras[0].level==9 and manager.cameras[0].power
    assert manager.cameras[1].level==7 and not manager.cameras[1].power
    assert len(changed)==1
    assert any('C2:' in message and 'skipped' in message for message in controls.errors)
    summary=controls.summary([0,1])
    assert 'C0:' in summary and chr(10) in summary
    assert 'high power not adjustable' in summary


def test_partial_failure_settles_calibration_and_reports_actual_power():
    camera=Camera()
    def fail(value):raise RuntimeError('camera disconnected')
    camera.set_intensity=fail
    controls=CalibrationControls(Manager([camera]))
    controls.handle_key('m')
    assert camera.power and camera.level==4
    assert not controls.capture_ready
    assert 'disconnected' in controls.last_error


def test_calibration_keyboard_selects_camera_clamps_and_toggles_power():
    manager=Manager([Camera(8),Camera(6)])
    controls=CalibrationControls(manager)
    controls.handle_key('1')
    controls.handle_key('i')
    assert [c.level for c in manager.cameras]==[4,5]
    controls.handle_key('i');controls.handle_key('i')
    assert manager.cameras[1].level==6
    controls.handle_key('k')
    assert manager.cameras[1].level==5
    controls.handle_key('h')
    assert manager.cameras[1].power and not manager.cameras[0].power
    controls.handle_key('a');controls.handle_key('m')
    assert [c.level for c in manager.cameras]==[8,6]
    assert all(c.power for c in manager.cameras)
    assert not controls.capture_ready
    controls.handle_key('h')
    assert not any(c.power for c in manager.cameras)


def test_camera_manager_methods_validate_index_and_delegate(monkeypatch):
    from unity_hand_tracking.optitrack_cam_py.multi_mjpeg import CameraManager
    manager=CameraManager.__new__(CameraManager)
    import threading
    manager._stop_lock=threading.RLock()
    manager._shutdown=False
    camera=Camera(11)
    manager.workers=[SimpleNamespace(camera=camera, capture_settings={})]
    manager.recorder=None
    with pytest.raises(ValueError):manager.get_illumination(-1)
    with pytest.raises(ValueError):manager.set_illumination(2,maximize=True)
    assert manager.set_illumination(0,maximize=True).intensity==11
    assert manager.get_illumination(0).high_power


def test_native_panels_apply_targeted_controls_without_startup_writes():
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from handtrack.applications._illumination_qt import IlluminationPanel
    from handtrack.applications._calibration_qt import CalibrationPreview
    app=QApplication.instance() or QApplication([])
    manager=Manager([Camera(9),Camera(7,supported=False)])
    panel=IlluminationPanel(IlluminationControls(manager))
    assert not any(c.writes for c in manager.cameras)
    panel.target.setCurrentIndex(1)
    panel.max_button.click()
    assert manager.cameras[0].level==9 and manager.cameras[0].power
    assert manager.cameras[1].level==4
    panel.target.setCurrentIndex(2)
    assert not panel.high_power.isEnabled()
    panel.bind_manager(None)
    assert not panel.max_button.isEnabled()
    preview=CalibrationPreview(CalibrationControls(manager),'Calibration lighting test')
    try:
        assert preview.illumination.target.focusPolicy()==Qt.NoFocus
        preview.show_frames([np.zeros((240,320,3),np.uint8)]*2,'Lighting test')
        QTest.keyClick(preview,Qt.Key_1)
        assert preview.illumination.target.currentData()==1
        QTest.keyClick(preview,Qt.Key_M)
        assert manager.cameras[1].level==7
        assert 'IR 7' in preview.illumination.status.text()
        assert not preview.controls.capture_ready
        assert 'I/K' in preview.help.text()
        app.processEvents()
    finally:
        preview.shutdown()
        panel.close()


def test_explicit_power_off_is_preserved_when_maximizing_intensity():
    camera=Camera()
    camera.power=True
    state=set_illumination(camera,maximize=True,high_power=False)
    assert state.intensity==15 and state.high_power is False
    assert camera.writes==[('power',False),('intensity',15)]


def test_truthy_non_boolean_cannot_accidentally_maximize_power():
    camera=Camera()
    with pytest.raises(ValueError,match='Maximize'):
        set_illumination(camera,maximize='false')
    assert camera.writes==[]
