"""OptiHub mode changes must be explicit and allow recovery without input pulses."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import threading
from types import SimpleNamespace

import pytest

from unity_hand_tracking.optitrack_cam_py import multi_mjpeg


@pytest.fixture
def rig(monkeypatch):
    calls = []
    values = dict(available=True, input_source=0, input_trigger=3, input_divider=1,
                  global_offset_us=0, internal_frequency_hz=50, device_name="OptiHub 2")
    def configure(preset):
        calls.append(preset)
        values["input_source"] = 2 if preset == "vicon_external_50hz" else 0
    sdk = SimpleNamespace(get_sync_status=lambda: dict(values), configure_sync_preset=configure)
    monkeypatch.setattr(multi_mjpeg, "optitrack_cam", sdk)
    manager = multi_mjpeg.CameraManager.__new__(multi_mjpeg.CameraManager)
    manager._stop_lock = threading.RLock()
    manager._sync_lock = threading.RLock()
    manager._expected_sync_hz = None
    manager._shutdown = False
    manager.recorder = None
    started = threading.Event()
    started.set()
    manager.workers = [SimpleNamespace(index=0, name="Camera 0", status="waiting for frames", error="",
                                      ready=threading.Event(), frame_timestamp=None,
                                      capture_started=started, is_alive=lambda: True)]
    manager._sample_sync_status()
    return manager, sdk, calls, values


def test_presets_record_expectation_without_claiming_a_measured_signal(rig):
    manager, sdk, calls, values = rig
    assert calls == []  # Reading status never applies settings.
    status = manager.configure_sync_preset("vicon_external_50hz")
    assert calls == ["vicon_external_50hz"] and status["input_source"] == 2
    assert status["expected_frequency_hz"] == 50
    assert status["input_edges"] is None and status["input_level"] is None
    assert not status["input_edge_recording_supported"]
    manager.configure_sync_preset("internal_50hz")
    assert calls[-1] == "internal_50hz" and manager.get_sync_status()["input_source"] == 0


def test_recording_or_closed_session_prevents_mode_changes(rig):
    manager, sdk, calls, values = rig
    manager.recorder = object()
    with pytest.raises(RuntimeError, match="Stop recording"):
        manager.configure_sync_preset("vicon_external_50hz")
    manager.recorder = None
    manager._shutdown = True
    with pytest.raises(RuntimeError, match="Start cameras"):
        manager.configure_sync_preset("vicon_external_50hz")
    assert calls == []


def test_missing_extension_and_invalid_presets_fail_without_writes(rig):
    manager, sdk, calls, values = rig
    with pytest.raises(ValueError, match="Unknown"):
        manager.configure_sync_preset("arbitrary")
    del sdk.configure_sync_preset
    assert not manager._sample_sync_status()["configurable"]
    with pytest.raises(RuntimeError, match="Rebuild"):
        manager.configure_sync_preset("vicon_external_50hz")
    assert calls == []


def test_sdk_rejection_refreshes_status_and_does_not_claim_success(rig):
    manager, sdk, calls, values = rig
    def reject(preset):
        raise RuntimeError("SDK rejected sync")
    sdk.configure_sync_preset = reject
    with pytest.raises(RuntimeError, match="SDK rejected"):
        manager.configure_sync_preset("vicon_external_50hz")
    assert manager.get_sync_status()["input_source"] == 0
    assert manager.get_sync_status()["expected_frequency_hz"] is None


def test_external_sync_can_start_without_pulses_but_camera_failure_still_fails(rig):
    manager, sdk, calls, values = rig
    values["input_source"] = 2
    manager.wait_ready(timeout=0)
    assert manager.workers[0].frame_timestamp is None
    manager.workers[0].error = "USB disconnected"
    with pytest.raises(RuntimeError, match="USB disconnected"):
        manager.wait_ready(timeout=0)


def test_internal_mode_still_requires_camera_frames(rig):
    manager, sdk, calls, values = rig
    with pytest.raises(RuntimeError, match="did not supply"):
        manager.wait_ready(timeout=0)


def test_exposure_output_preserves_input_and_is_blocked_during_recording(rig):
    manager, sdk, calls, values = rig
    def output():
        calls.append('exposure')
        values['output1_type'] = 0
    sdk.configure_exposure_output = output
    values['input_source'] = 2
    manager.configure_exposure_output()
    assert calls == ['exposure'] and manager.get_sync_status()['input_source'] == 2
    manager.recorder = object()
    with pytest.raises(RuntimeError, match='Stop recording'):
        manager.configure_exposure_output()
    assert calls == ['exposure']


def test_gui_requires_apply_and_blocks_changes_while_recording(monkeypatch, rig):
    from PyQt5.QtWidgets import QApplication
    from unity_hand_tracking.optitrack_cam_py import mocap_handtrack_gui as gui
    monkeypatch.setattr(gui.StereoHandTrackerGUI, "start_udp_broadcast", lambda self: None)
    monkeypatch.setattr(gui.StereoHandTrackerGUI, "stop_udp_broadcast", lambda self: None)
    app = QApplication.instance() or QApplication([])
    window = gui.StereoHandTrackerGUI()
    manager, sdk, calls, values = rig
    try:
        assert window.sync_preset_combo.window() == window.sync_dialog
        assert window.record_fps.window() == window.sync_dialog
        assert window.nexus_network_checkbox.window() == window.sync_dialog
        assert window.trial_name_edit.window() == window
        assert not window.nexus_network_checkbox.isChecked()
        assert not window.sync_apply_btn.isEnabled()
        window.tracker = SimpleNamespace(cam_mgr=manager)
        window.is_running = True
        window._update_capture_controls()
        assert window.sync_apply_btn.isEnabled()
        window.sync_preset_combo.setCurrentIndex(1)
        assert not calls
        window.sync_apply_btn.click()
        assert calls == ["internal_50hz"] and window.record_fps.value() == 50
        window.sync_preset_combo.setCurrentIndex(0)
        window.sync_apply_btn.click()
        assert calls[-1] == "vicon_external_50hz"
        assert "No fresh camera frames" in window.sync_status.text()
        manager.recorder = SimpleNamespace(status=lambda: dict(error=None, written=0, dropped=0, pending=0),
                                          directory="test")
        window._update_capture_controls()
        assert not window.sync_apply_btn.isEnabled()
        assert not window.sync_preset_combo.isEnabled()
    finally:
        window.tracker = None
        window.is_running = False
        window.close()
        app.processEvents()
