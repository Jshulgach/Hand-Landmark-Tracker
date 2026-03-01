from ._features import (
    extract_features,
    integrated_emg,
    mean_absolute_value,
    root_mean_square,
    slope_sign_changes,
    variance,
    waveform_length,
    zero_crossings,
)
from ._filters import (
    bandpass_filter,
    compute_rms,
    lowpass_filter,
    notch_filter,
    rectify,
)
from ._joint_angles import (
    angle_between_points,
    compute_3point_finger_angles,
    compute_all_joint_angles,
)
from ._kalman_filter import Kalman3D, KalmanAngle
from ._preprocessing import EMGPreprocessor
from ._smoothing import (
    EMA1D,
    EMA3D,
    AdaptiveKalman3D,
    Kalman1D,
    angle_between,
    finger_bend_angles,
    finger_splay_angles,
)
from ._anatomical_constraints import enforce_pip_constraints
from ._smoothing_factory import build_smoother_factories
# from ._pipeline import run_pipeline
