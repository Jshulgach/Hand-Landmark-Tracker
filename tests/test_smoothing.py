import numpy as np
from handtrack.processing import AdaptiveKalman3D, build_smoother_factories


def test_factory_kalman_mode():
    landmark_factory, angle_factory = build_smoother_factories(
        method="kalman",
        kalman_3d_process_noise=1e-3,
        kalman_3d_measurement_noise=1e-2,
        kalman_1d_process_noise=0.1,
        kalman_1d_measurement_noise=1.0,
        ema_3d_alpha=0.45,
        ema_1d_alpha=0.35,
    )
    assert landmark_factory().__class__.__name__ == "AdaptiveKalman3D"
    assert angle_factory().__class__.__name__ == "Kalman1D"


def test_factory_ema_mode():
    landmark_factory, angle_factory = build_smoother_factories(
        method="ema",
        kalman_3d_process_noise=1e-3,
        kalman_3d_measurement_noise=1e-2,
        kalman_1d_process_noise=0.1,
        kalman_1d_measurement_noise=1.0,
        ema_3d_alpha=0.45,
        ema_1d_alpha=0.35,
    )
    assert landmark_factory().__class__.__name__ == "EMA3D"
    assert angle_factory().__class__.__name__ == "EMA1D"


def test_reprojection_error_influences_kalman_output():
    filt_low = AdaptiveKalman3D(process_noise=1e-3, measurement_noise=1e-2)
    filt_high = AdaptiveKalman3D(process_noise=1e-3, measurement_noise=1e-2)

    # Seed filter state with identical non-zero baseline measurements.
    # (A zero vector is treated as missing data by AdaptiveKalman3D.)
    for _ in range(5):
        baseline = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        filt_low.update(baseline, reprojection_error=0.0, num_cameras=4)
        filt_high.update(baseline, reprojection_error=0.0, num_cameras=4)

    # Apply the same movement with different reprojection confidence.
    # Keep magnitude moderate so outlier scaling doesn't dominate this test.
    measurement = np.array([2.0, 0.0, 0.0], dtype=np.float32)
    out_low = filt_low.update(measurement, reprojection_error=0.5, num_cameras=4)
    out_high = filt_high.update(measurement, reprojection_error=60.0, num_cameras=4)

    # High reprojection error should inflate measurement covariance.
    assert filt_high.R[0, 0] > filt_low.R[0, 0]

    # And the filtered output should stay farther from the raw measurement.
    dist_low = np.linalg.norm(measurement - out_low)
    dist_high = np.linalg.norm(measurement - out_high)
    assert dist_high > dist_low


def _run_all():
    test_factory_kalman_mode()
    test_factory_ema_mode()
    test_reprojection_error_influences_kalman_output()
    print("test_smoothing: all tests passed")


if __name__ == "__main__":
    _run_all()
