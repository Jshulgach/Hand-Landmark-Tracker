"""Bounded OptiTrack IR controls with SDK-reported capability and setting readback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IlluminationState:
    intensity: int | None = None
    minimum: int | None = None
    maximum: int | None = None
    high_power: bool | None = None
    high_power_supported: bool = False
    high_power_available: bool = False
    message: str = ""

    @property
    def intensity_supported(self) -> bool:
        return self.intensity is not None and self.minimum is not None and self.maximum is not None

    def describe(self) -> str:
        level = f"IR {self.intensity} [{self.minimum}-{self.maximum}]" if self.intensity_supported else "IR unavailable"
        if self.high_power_supported:
            power = "on" if self.high_power else "off"
            level += f"; high power {power}"
            if not self.high_power_available:
                level += " (unavailable on this connection)"
        else:
            level += "; high power not adjustable"
        return level + (f"; {self.message}" if self.message else "")


def read_illumination(camera: Any) -> IlluminationState:
    intensity_methods = ("get_intensity", "min_intensity", "max_intensity", "set_intensity")
    power_methods = ("is_high_power_supported", "is_high_power_available", "get_high_power", "set_high_power")
    intensity_api = all(callable(getattr(camera, name, None)) for name in intensity_methods)
    power_api = all(callable(getattr(camera, name, None)) for name in power_methods)
    if not intensity_api and not power_api:
        return IlluminationState(message="Binding upgrade required for illumination controls")
    try:
        current = int(camera.get_intensity()) if intensity_api else None
        low = int(camera.min_intensity()) if intensity_api else None
        high = int(camera.max_intensity()) if intensity_api else None
        if low is not None and high is not None and (low < 0 or high < low):
            raise ValueError("Camera returned invalid intensity limits")
        supported = bool(camera.is_high_power_supported()) if power_api else False
        available = bool(camera.is_high_power_available()) if supported else False
        enabled = bool(camera.get_high_power()) if supported else None
        return IlluminationState(current, low, high, enabled, supported, available)
    except Exception as exc:  # noqa: BLE001 - Native device boundary must preserve preview on failures.
        return IlluminationState(message=f"Illumination query failed: {exc}")


def set_illumination(camera: Any, *, intensity: int | None = None,
                     high_power: bool | None = None, maximize: bool = False) -> IlluminationState:
    if type(maximize) is not bool:
        raise ValueError("Maximize must be true or false")
    state = read_illumination(camera)
    if intensity is not None:
        if type(intensity) is not int:
            raise ValueError("Intensity must be an integer SDK value")
        if maximize:
            raise ValueError("Choose intensity or maximize")
        if not state.intensity_supported:
            raise RuntimeError(state.message or "Intensity is not adjustable")
        assert state.minimum is not None and state.maximum is not None
        if not state.minimum <= intensity <= state.maximum:
            raise ValueError(f"Intensity must be between {state.minimum} and {state.maximum}")
    if high_power is not None and type(high_power) is not bool:
        raise ValueError("High power must be true or false")
    requested_power = high_power
    if maximize and high_power is None and state.high_power_supported and state.high_power_available:
        requested_power = True
    if requested_power is not None:
        if not state.high_power_supported:
            raise RuntimeError(state.message or "High power is not adjustable on this camera")
        if requested_power and not state.high_power_available:
            raise RuntimeError("High power unavailable: check supported camera and powered OptiHub2")
    if maximize and not state.intensity_supported and requested_power is None:
        raise RuntimeError(state.message or "Illumination is not adjustable on this camera")
    if requested_power is not None:
        camera.set_high_power(requested_power)
        state = read_illumination(camera)
        if state.high_power is not requested_power:
            raise RuntimeError("High-power SDK readback does not match the requested setting")
    requested_level = state.maximum if maximize and state.intensity_supported else intensity
    if requested_level is not None:
        # The power mode can change supported limits; validate after its readback too.
        assert state.minimum is not None and state.maximum is not None
        if not state.minimum <= requested_level <= state.maximum:
            raise ValueError("Requested intensity is outside the current camera limits")
        camera.set_intensity(requested_level)
    actual = read_illumination(camera)
    if requested_level is not None and actual.intensity != requested_level:
        raise RuntimeError("Intensity SDK readback does not match the requested setting")
    return actual
