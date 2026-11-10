"""Loopback-only recording control, polled on the Qt GUI thread."""
import json
import socket


class CaptureControl:
    def __init__(self, window, port):
        self.window = window
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.socket.bind(('127.0.0.1', port))
            self.socket.setblocking(False)
        except Exception:
            self.socket.close()
            raise

    def close(self):
        self.socket.close()

    def handle(self, request):
        window = self.window
        manager = window._capture_manager()
        action = request.get('action')
        if action not in ('status', 'start', 'stop'):
            raise ValueError('Expected status, start or stop')
        if action == 'start':
            if getattr(window, '_closing_capture', False):
                raise ValueError('MAVIS is closing')
            if not window.nexus_network_checkbox.isChecked():
                raise ValueError('Enable Nexus network requests in MAVIS first')
            if manager is None or not window.is_running:
                raise ValueError('Start cameras in MAVIS first')
            if window._record_stop_thread is not None:
                raise ValueError('Previous recording is still finishing')
            if manager.recorder is not None:
                raise ValueError('MAVIS is already recording; start was not repeated')
            if not window.record_directory_edit.text().strip():
                raise ValueError('Choose the video output folder in MAVIS first')
            from mavis_track._nexus_remote import NexusRemote
            name = request.get('trial_name', '')
            if not isinstance(name, str):
                raise TypeError('Trial name must be text')
            NexusRemote.validate_name(name)
            window.trial_name_edit.setText(name)
            window._toggle_recording()
            if manager.recorder is None:
                raise ValueError(window.record_status.text())
        elif (action == 'stop' and manager is not None and manager.recorder is not None
              and window._record_stop_thread is None):
            window._toggle_recording()
        return {'recording': manager is not None and manager.recorder is not None,
                    'finishing': window._record_stop_thread is not None,
                    'nexus_enabled': window.nexus_network_checkbox.isChecked(),
                    'detail': window.record_status.text()}

    def poll(self):
        # Bound work per tick so an accidental packet flood cannot freeze Qt.
        for _ in range(8):
            try:
                packet, peer = self.socket.recvfrom(4096)
            except OSError:
                return
            try:
                request = {}
                request = json.loads(packet)
                if not isinstance(request, dict):
                    raise TypeError('Expected a JSON object')
                response = dict(ok=True, **self.handle(request))
            except Exception as exc:  # noqa: BLE001 - keep Qt alive on command failures
                response = {'ok': False, 'error': str(exc)}
            response['request_id'] = request.get('request_id') if isinstance(request, dict) else None
            try:
                self.socket.sendto(json.dumps(response).encode(), peer)
            except OSError:
                pass
