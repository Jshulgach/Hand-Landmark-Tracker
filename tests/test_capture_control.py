"""Local capture commands must preserve the existing recording lifecycle."""
import json
import socket
from types import SimpleNamespace

import pytest

from mavis_track._capture_control import CaptureControl


class Field:
    def __init__(self, value):
        self.value = value

    def text(self):
        return self.value

    def setText(self, value):
        self.value = value

    def isChecked(self):
        return self.value


def setup_control():
    manager = SimpleNamespace(recorder=None)
    calls = []
    window = SimpleNamespace(
        _capture_manager=lambda: manager, is_running=True, _record_stop_thread=None,
        record_directory_edit=Field('output'), trial_name_edit=Field('old'),
        nexus_network_checkbox=Field(True), record_status=Field('Ready'))

    def toggle():
        calls.append(window.trial_name_edit.text())
        if manager.recorder is None:
            manager.recorder = object()
        else:
            window._record_stop_thread = object()

    window._toggle_recording = toggle
    control = CaptureControl.__new__(CaptureControl)
    control.window = window
    return control, window, manager, calls


def test_start_then_stop_uses_existing_path_once():
    control, window, _manager, calls = setup_control()
    assert control.handle({'action': 'start', 'trial_name': 'Reach01'})['recording']
    with pytest.raises(ValueError, match='already recording'):
        control.handle({'action': 'start', 'trial_name': 'Other'})
    assert window.trial_name_edit.text() == 'Reach01'
    assert control.handle({'action': 'stop'})['finishing']
    control.handle({'action': 'stop'})
    assert calls == ['Reach01', 'Reach01']


@pytest.mark.parametrize('field,value,message', [
    ('is_running', False, 'Start cameras'),
    ('_record_stop_thread', object(), 'still finishing'),
    ('record_directory_edit', Field(''), 'output folder'),
    ('nexus_network_checkbox', Field(False), 'Enable Nexus')])
def test_start_preconditions(field, value, message):
    control, window, _manager, calls = setup_control()
    setattr(window, field, value)
    with pytest.raises(ValueError, match=message):
        control.handle({'action': 'start', 'trial_name': 'Reach01'})
    assert not calls


def test_invalid_name_does_not_start_recording():
    control, _window, _manager, calls = setup_control()
    with pytest.raises(ValueError):
        control.handle({'action': 'start', 'trial_name': '../trial'})
    assert not calls


def test_loopback_transport_rejects_bad_packets_and_recovers():
    _control, window, _manager, calls = setup_control()
    listener = CaptureControl(window, 0)
    try:
        assert listener.socket.getsockname()[0] == '127.0.0.1'
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
            client.settimeout(1)
            for packet in (b'not json', b'[]', b'{"action":"unknown"}'):
                client.sendto(packet, listener.socket.getsockname())
                listener.poll()
                assert not json.loads(client.recv(4096))['ok']
            for action in ('start', 'status', 'stop'):
                client.sendto(json.dumps({'action': action, 'trial_name': 'Reach01',
                                          'request_id': action}).encode(),
                              listener.socket.getsockname())
                listener.poll()
                result = json.loads(client.recv(4096))
                assert result['ok'] and result['request_id'] == action
            assert calls == ['Reach01', 'Reach01']
    finally:
        listener.close()


def test_failed_recording_start_is_reported():
    control, window, _manager, _calls = setup_control()
    window._toggle_recording = lambda: None
    window.record_status.setText('Recording error: disk full')
    with pytest.raises(ValueError, match='disk full'):
        control.handle({'action': 'start', 'trial_name': 'Reach01'})
