"""Remote requests are opt-in and must never be reported as acknowledged capture."""
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest

from mavis_track import _nexus_remote as nexus


def test_nexus_packets_are_null_terminated_escaped_and_unique(monkeypatch):
    sent = []
    class Socket:
        def settimeout(self, timeout): pass
        def sendto(self, packet, target): sent.append((packet, target))
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(nexus.socket, 'socket', lambda *args: Socket())
    remote = nexus.NexusRemote('127.0.0.1', 30)
    start, stop = remote.send('start', 'Reach & test'), remote.send('stop', 'Reach & test')
    assert start['packet_id'] != stop['packet_id']
    assert start['status'] == stop['status'] == 'sent_unconfirmed'
    for (packet, address), tag in zip(sent, ('CaptureStart', 'CaptureStop')):
        assert packet.endswith(b'\0') and address == ('127.0.0.1', 30)
        element = ET.fromstring(packet[:-1])
        assert element.tag == tag and element.find('Name').get('VALUE') == 'Reach & test'
        assert element.find('Delay').get('VALUE') == '0'


@pytest.mark.parametrize('name', ['', '../trial', 'CON', 'trial.', 'a:b'])
def test_invalid_nexus_names_fail_before_sending(name):
    with pytest.raises(ValueError): nexus.NexusRemote.validate_name(name)


def test_network_configuration_requires_specific_ipv4_and_valid_port():
    for host, port in [('0.0.0.0', 30), ('255.255.255.255', 30), ('224.0.0.1', 30), ('localhost', 30), ('127.0.0.1', 0)]:
        with pytest.raises(ValueError): nexus.NexusRemote(host, port)


def test_failed_send_does_not_claim_success(monkeypatch):
    def fail(*args): raise OSError('No network')
    monkeypatch.setattr(nexus.socket, 'socket', fail)
    with pytest.raises(OSError, match='No network'):
        nexus.NexusRemote('127.0.0.1').send('start', 'Trial001')
