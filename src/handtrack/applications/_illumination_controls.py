"""Shared targeting and failure reporting for tracking and calibration lighting."""
from mavis_track._illumination import IlluminationState


class IlluminationControls:
    def __init__(self, manager, on_change=None):
        self.manager = manager
        self.on_change = on_change
        self.errors = []

    def read(self, index):
        if self.manager is None or not callable(getattr(self.manager, 'get_illumination', None)):
            return IlluminationState(message='Binding upgrade required for illumination controls')
        try:
            return self.manager.get_illumination(index)
        except Exception as exc:
            return IlluminationState(message=f'Illumination query failed: {exc}')

    def apply(self, indices, *, intensity=None, delta=None, high_power=None, maximize=False):
        self.errors = []
        attempted = False
        for index in indices:
            state = self.read(index)
            if not callable(getattr(self.manager, 'set_illumination', None)):
                self.errors.append(f'C{index}: binding upgrade required; skipped')
                continue
            if (intensity is not None or delta is not None) and not state.intensity_supported:
                self.errors.append(f'C{index}: {state.message or "intensity unavailable"}; skipped')
                continue
            if high_power is not None and (not state.high_power_supported or
                    (high_power and not state.high_power_available)):
                self.errors.append(f'C{index}: high power unavailable; skipped')
                continue
            if maximize and not (state.intensity_supported or
                    state.high_power_supported and state.high_power_available):
                self.errors.append(f'C{index}: {state.message or "illumination unavailable"}; skipped')
                continue
            level = intensity
            if delta is not None:
                level = max(state.minimum, min(state.maximum, state.intensity + delta))
            try:
                attempted = True
                self.manager.set_illumination(index, intensity=level, high_power=high_power, maximize=maximize)
            except Exception as exc:
                self.errors.append(f'C{index}: {exc}')
        if attempted and self.on_change is not None:
            # Also settle after partial failures, which may have changed one property.
            self.on_change()

    def summary(self, indices):
        return '\n'.join(f'C{index}: {self.read(index).describe()}' for index in indices)
