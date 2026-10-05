"""Shared camera snapshot semantics for the optional multi-camera backends."""

import time

import numpy as np


class FrameBuffer:
    """Workers publish immutable frames with host receipt timestamps.

    These clocks measure receipt time; they do not establish hardware exposure
    synchronization. A snapshot expires after 250 ms by default.
    """

    def publish_frame(self, frame):
        frame.flags.writeable = False
        with self.lock:
            self.height, self.width = frame.shape[:2]
            self.latest_frame = frame
            self.frame_timestamp = time.perf_counter()
            self.status = "streaming"
        self.ready.set()

    def get_frame(self, copy=True):
        with self.lock:
            frame = self.latest_frame
            if frame is not None:
                return frame.copy() if copy else frame
        return np.zeros((self.height, self.width, 3), np.uint8)

    def get_snapshot(self, max_age=0.25):
        with self.lock:
            timestamp = getattr(self, "frame_timestamp", None)
            fresh = timestamp is not None and 0 <= time.perf_counter() - timestamp <= max_age
            return (self.latest_frame if fresh else None, timestamp)


def snapshots(workers, max_skew=0.05):
    """Drop expired frames and cameras outside a receipt-time skew window."""
    values = [worker.get_snapshot() for worker in workers]
    latest = max((stamp for frame, stamp in values if frame is not None), default=None)
    return [frame if frame is not None and latest - stamp <= max_skew else None
            for frame, stamp in values]
