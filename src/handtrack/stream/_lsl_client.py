import threading
from collections import deque
from pylsl import StreamInlet, resolve_byprop


class LSLClient:
    def __init__(self, maxlen=10000, stream_type="EMG"):
        print(f"[LSLClient] Looking for a stream of type '{stream_type}'...")
        streams = resolve_byprop("type", stream_type, timeout=5)
        if not streams:
            raise RuntimeError(f"No LSL stream with type '{stream_type}' found.")

        self.inlet = StreamInlet(streams[0])
        self.stream_info = self.inlet.info()
        self.n_channels = self.stream_info.channel_count()
        self.fs = self._get_sampling_rate()
        self.channel_labels, self.units = self._get_channel_metadata()

        self.buffers = [deque(maxlen=maxlen) for _ in range(self.n_channels)]
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._pull_data_loop, daemon=True)
        self.thread.start()
        print(f"[LSLClient] Connected to stream '{self.stream_info.name()}'")
        print(f"  Channels: {self.n_channels}, Sampling Rate: {self.fs} Hz")

    def _get_sampling_rate(self):
        try:
            rate = self.stream_info.nominal_srate()
            return float(rate) if rate > 0 else None
        except Exception:
            return None

    def _get_channel_metadata(self):
        try:
            ch_info = self.stream_info.desc().child("channels").child("channel")
            labels = []
            units = []
            for _ in range(self.n_channels):
                labels.append(ch_info.child_value("label") or f"Ch{_}")
                units.append(ch_info.child_value("unit") or "unknown")
                ch_info = ch_info.next_sibling()
            return labels, units
        except Exception:
            return [f"Ch{i}" for i in range(self.n_channels)], ["unknown"] * self.n_channels

    def _pull_data_loop(self):
        while self.running:
            sample, _ = self.inlet.pull_sample(timeout=0.1)
            if sample is not None:
                with self.lock:
                    for ch, val in enumerate(sample):
                        if ch < self.n_channels:
                            self.buffers[ch].append(val)

    def get_samples(self, channel: int, n_samples: int):
        with self.lock:
            buf = list(self.buffers[channel])
        if len(buf) < n_samples:
            buf = [0.0] * (n_samples - len(buf)) + buf
        return buf[-n_samples:]

    def stop(self):
        self.running = False
        self.thread.join()
        print("[LSLClient] Stopped.")