"""Frame identity loss is separate from host delivery jitter and input edges."""
import math


class FrameDiagnostics:
    def __init__(self):
        self.cameras = {}
        self._previous = {}

    def observe(self, index, metadata):
        stats = self.cameras.setdefault(str(index), dict(frames=0, sdk_gaps=0, sdk_missing=0,
            sdk_resets_or_duplicates=0, sdk_timing_frames=0, host_nonmonotonic=0,
            max_host_interval_seconds=0., first_host_time=None, last_host_time=None))
        stats['frames'] += 1
        previous = self._previous.setdefault(index, {})
        frame_id = metadata.get('sdk_frame_id')
        if frame_id is not None:
            stats['sdk_timing_frames'] += 1
            last_id = previous.get('sdk_frame_id')
            if last_id is not None:
                gap = frame_id - last_id - 1
                if gap > 0:
                    stats['sdk_gaps'] += 1
                    stats['sdk_missing'] += int(gap)
                    metadata['sdk_missing_since_previous_written'] = int(gap)
                elif gap < 0:
                    stats['sdk_resets_or_duplicates'] += 1
                    metadata['sdk_sequence_discontinuity'] = True
            previous['sdk_frame_id'] = frame_id
        stamp = metadata.get('host_received_monotonic')
        if stamp is not None and math.isfinite(stamp):
            if stats['first_host_time'] is None:
                stats['first_host_time'] = stamp
            last = previous.get('host_received_monotonic')
            if last is not None:
                if stamp <= last:
                    stats['host_nonmonotonic'] += 1
                else:
                    stats['max_host_interval_seconds'] = max(stats['max_host_interval_seconds'], stamp-last)
            previous['host_received_monotonic'] = stamp
            stats['last_host_time'] = stamp

    def summary(self):
        return {key: dict(value) for key, value in self.cameras.items()}
