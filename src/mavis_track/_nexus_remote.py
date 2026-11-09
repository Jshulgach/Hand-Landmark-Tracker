"""Optional Nexus UDP capture requests; sending is not an acknowledgement."""
import ipaddress
import secrets
import socket
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import PureWindowsPath

_packet_id = secrets.randbits(31)
_id_lock = threading.Lock()


class NexusRemote:
    def __init__(self, address, port=30):
        address = ipaddress.IPv4Address(address)
        if address.is_multicast or address.is_unspecified or str(address) == '255.255.255.255':
            raise ValueError('Enter the specific Nexus computer IPv4 address')
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError('Nexus UDP port must be between 1 and 65535')
        self.address, self.port = str(address), port

    @staticmethod
    def validate_name(name):
        if (not name or any(c in '<>:"/\\|?*' or ord(c) < 32 for c in name) or
                name.endswith(('.', ' ')) or PureWindowsPath(name).is_reserved()):
            raise ValueError('Nexus trial name must be a valid Windows file name')

    def send(self, action, trial_name):
        global _packet_id
        if action not in ('start', 'stop'):
            raise ValueError('Nexus action must be start or stop')
        self.validate_name(trial_name)
        with _id_lock:
            _packet_id = (_packet_id + 1) % (2**31)
            packet_id = _packet_id
        root = ET.Element('CaptureStart' if action == 'start' else 'CaptureStop',
                          {} if action == 'start' else {'RESULT': 'SUCCESS'})
        for key, value in [('Name', trial_name), ('DatabasePath', ''), ('Delay', '0'), ('PacketID', str(packet_id))]:
            ET.SubElement(root, key, VALUE=value)
        packet = b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>' + ET.tostring(root, encoding='utf-8') + b'\0'
        if len(packet) > 1400:
            raise ValueError('Nexus capture request exceeds the supported single-packet size')
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
            client.settimeout(.5)
            client.sendto(packet, (self.address, self.port))
        return {'action': action, 'packet_id': packet_id, 'trial_name': trial_name,
                    'address': self.address, 'port': self.port, 'host_sent_monotonic': time.perf_counter(),
                    'status': 'sent_unconfirmed', 'note': 'UDP send does not confirm Nexus capture or frame alignment'}
