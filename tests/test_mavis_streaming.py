import json
import numpy as np

from handtrack.io import broadcast


def test_udp_sends_hand_loss_and_valid_json_with_missing_coordinates(monkeypatch):
    packets = []
    class Socket:
        def setsockopt(self, *_):
            pass
        def sendto(self, message, address):
            packets.append((message, address))
        def close(self):
            pass
    monkeypatch.setattr(broadcast.socket, 'socket', lambda *_: Socket())
    sender = broadcast.UDPBroadcaster(coordinate_space='image_normalized', timestamp_clock='source')
    sender.send_landmarks(1, 0.1, [])
    sender.send_landmarks(2, 0.2, [{'hand_index': 0, 'landmarks': np.full((21, 3), np.nan)}])
    assert json.loads(packets[0][0])['num_hands'] == 0
    packet = json.loads(packets[1][0])
    assert packet['hands'][0]['landmarks'][0] == [None, None, None]
    assert packet['coordinate_space'] == 'image_normalized'
    assert b'NaN' not in packets[1][0]
    sender.close()


def test_lsl_hand_loss_is_all_missing_and_close_releases_outlet_references():
    samples = []
    class Outlet:
        def push_sample(self, sample, stamp):
            samples.append((sample, stamp))
    sender = broadcast.LSLBroadcaster.__new__(broadcast.LSLBroadcaster)
    sender.outlet_landmarks = Outlet()
    sender.outlet_angles = None
    sender.send_landmarks(1, 12.0, [])
    assert len(samples[0][0]) == 126 and np.isnan(samples[0][0]).all()
    sender.close()
    assert sender.outlet_landmarks is None
