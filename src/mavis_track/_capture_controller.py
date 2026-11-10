"""Standalone local controller for MAVIS recording and its Nexus trigger."""
import argparse
import json
import socket
import sys
import uuid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=5015)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import (
        QApplication,
        QLabel,
        QLineEdit,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    app = QApplication.instance() or QApplication(sys.argv)

    class Controller(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle('Nexus + MAVIS capture')
            self.client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.client.connect(('127.0.0.1', args.port))
            self.client.setblocking(False)
            self.pending = None
            self.request_id = None
            self.layout = QVBoxLayout(self)
            self.layout.addWidget(QLabel('Trial name'))
            self.name = QLineEdit('Trial001')
            self.layout.addWidget(self.name)
            self.start = QPushButton('Start both recordings')
            self.stop = QPushButton('Stop both recordings')
            self.layout.addWidget(self.start)
            self.layout.addWidget(self.stop)
            self.status = QLabel('Checking MAVIS…')
            self.status.setWordWrap(True)
            self.layout.addWidget(self.status)
            self.result = QLabel('')
            self.result.setWordWrap(True)
            self.layout.addWidget(self.result)
            self.start.clicked.connect(lambda: self.send('start'))
            self.stop.clicked.connect(lambda: self.send('stop'))
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.receive)
            self.timer.start(50)
            self.refresh = QTimer(self)
            self.refresh.timeout.connect(lambda: self.send('status'))
            self.refresh.start(1000)
            self.timeout = QTimer(self)
            self.timeout.setSingleShot(True)
            self.timeout.timeout.connect(self.no_reply)
            self.send('status')

        def send(self, action):
            if self.pending is not None:
                return
            try:
                if action == 'start':
                    from mavis_track._nexus_remote import NexusRemote
                    NexusRemote.validate_name(self.name.text())
                self.request_id = uuid.uuid4().hex
                self.client.send(json.dumps({'action': action, 'trial_name': self.name.text(),
                                                 'request_id': self.request_id}).encode())
                self.pending = action
                self.start.setEnabled(False)
                self.stop.setEnabled(False)
                self.timeout.start(2000)
            except (OSError, ValueError) as exc:
                self.status.setText(str(exc))

        def receive(self):
            try:
                packet = self.client.recv(4096)
            except BlockingIOError:
                return
            except OSError:
                self.no_reply()
                return
            try:
                response = json.loads(packet)
                if response.get('request_id') != self.request_id or self.pending is None:
                    return
                self.timeout.stop()
                action = self.pending
                self.pending = None
                if not response['ok']:
                    self.result.setText(response['error'])
                    return
                if action in ('start', 'stop'):
                    self.result.setText(f'MAVIS handled {action}; verify Nexus capture in Nexus.')
                active = response['recording'] or response['finishing']
                self.name.setEnabled(not active)
                self.start.setEnabled(not active and response['nexus_enabled'])
                self.stop.setEnabled(response['recording'] and not response['finishing'])
                nexus = ('Nexus requests enabled; capture unconfirmed.' if response['nexus_enabled']
                         else 'Enable Nexus network requests in MAVIS to start both.')
                self.status.setText(response['detail'] + '\n' + nexus)
            except (ValueError, KeyError, TypeError, AttributeError):
                self.status.setText('Invalid response from MAVIS')

        def no_reply(self):
            action = self.pending
            self.pending = None
            self.start.setEnabled(False)
            self.stop.setEnabled(False)
            self.status.setText('No MAVIS reply. Check MAVIS directly; command outcome is unknown.'
                                if action in ('start', 'stop') else 'MAVIS unavailable. Check local control setup.')

        def closeEvent(self, event):
            self.client.close()
            event.accept()

    window = Controller()
    window.resize(400, 220)
    window.show()
    return app.exec_()


if __name__ == '__main__':
    raise SystemExit(main())
