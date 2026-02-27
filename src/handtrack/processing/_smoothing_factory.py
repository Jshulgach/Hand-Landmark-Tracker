from typing import Callable, Tuple

from ._smoothing import EMA1D, EMA3D, AdaptiveKalman3D, Kalman1D


def build_smoother_factories(
    method: str,
    kalman_3d_process_noise: float,
    kalman_3d_measurement_noise: float,
    kalman_1d_process_noise: float,
    kalman_1d_measurement_noise: float,
    ema_3d_alpha: float,
    ema_1d_alpha: float,
) -> Tuple[Callable[[], object], Callable[[], object]]:
    method = (method or "kalman").strip().lower()
    if method == "ema":
        landmark_factory = lambda: EMA3D(alpha=ema_3d_alpha)
        angle_factory = lambda: EMA1D(alpha=ema_1d_alpha)
        return landmark_factory, angle_factory

    landmark_factory = lambda: AdaptiveKalman3D(
        process_noise=kalman_3d_process_noise,
        measurement_noise=kalman_3d_measurement_noise,
    )
    angle_factory = lambda: Kalman1D(
        process_noise=kalman_1d_process_noise,
        measurement_noise=kalman_1d_measurement_noise,
    )
    return landmark_factory, angle_factory
