import numpy as np


class AdaptiveKalman3D:
    """3D Kalman filter with adaptive measurement noise and outlier damping."""

    MAX_JUMP_MM = 80.0
    VELOCITY_DAMPING = 0.95

    def __init__(self, dt=1 / 30, process_noise=1e-3, measurement_noise=1e-2):
        # State vector x = [px, py, pz, vx, vy, vz]^T
        self.x = np.zeros((6, 1))

        # Constant-velocity motion model:
        # p_t = p_{t-1} + v_{t-1} * dt
        # v_t = v_{t-1}
        self.F = np.eye(6)
        for i in range(3):
            self.F[i, i + 3] = dt

        # Observation model: we directly measure only position (x,y,z), not velocity.
        self.H = np.hstack((np.eye(3), np.zeros((3, 3))))

        # Covariances:
        # P: state uncertainty, Q: process (model) noise, R: measurement noise.
        self.P = np.eye(6)
        self.Q = np.eye(6) * process_noise
        self.R_base = measurement_noise
        self.R = np.eye(3) * measurement_noise
        self._initialized = False
        self._frames_lost = 0

    def update(self, z, reprojection_error=0.0, num_cameras=2):
        # Missing observation path: run a pure prediction step.
        # For short gaps, keep extrapolating; for long gaps, zero velocity and output origin.
        if z is None or (isinstance(z, np.ndarray) and np.linalg.norm(z) < 1e-3):
            self._frames_lost += 1
            if self._frames_lost > 5:
                self.x[3:] = 0
                return np.zeros(3)
            self.x[3:] *= 0.5
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q
            return self.x[:3].flatten()

        self._frames_lost = 0
        z = np.reshape(z, (3, 1))

        if not self._initialized:
            # First valid sample initializes position directly.
            self.x[:3] = z
            self.x[3:] = 0
            self._initialized = True
            return self.x[:3].flatten()

        # FreeMoCap-style confidence adaptation:
        # - Higher reprojection error => trust measurements less (increase R).
        # - Fewer cameras => trust measurements less (increase R).
        # Squared scaling makes high-error regions penalized more strongly.
        error_scale = 1.0 + (reprojection_error / 10.0) ** 2
        camera_scale = 2.0 / max(num_cameras, 1)
        adaptive_noise = self.R_base * error_scale * camera_scale
        self.R = np.eye(3) * adaptive_noise

        # Predict step (x-, P-).
        # Velocity damping reduces drift/overshoot when tracking confidence drops.
        self.x[3:] *= self.VELOCITY_DAMPING
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Outlier handling: if current measurement is a large jump from predicted state,
        # inflate measurement noise so Kalman gain drops for this update.
        predicted_pos = self.H @ self.x
        jump = np.linalg.norm(z - predicted_pos)
        if jump > self.MAX_JUMP_MM:
            outlier_scale = (jump / self.MAX_JUMP_MM) ** 2
            self.R *= outlier_scale

        # Correct step:
        # y = innovation, S = innovation covariance, K = Kalman gain.
        # x = x- + K y, P = (I - K H) P-
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x[:3].flatten()


class Kalman1D:
    """1D Kalman filter for scalar values (e.g., joint angles)."""

    def __init__(self, dt=1 / 30, process_noise=0.1, measurement_noise=1.0):
        # 1D constant-velocity state: [value, derivative]^T
        self.x = np.zeros((2, 1))
        self.F = np.array([[1, dt], [0, 1]])
        self.H = np.array([[1, 0]])
        self.P = np.eye(2)
        self.Q = np.eye(2) * process_noise
        self.R = np.array([[measurement_noise]])

    def update(self, z):
        # Standard linear Kalman predict + correct for scalar observations.
        z = np.array([[z]])
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x[0, 0]


class EMA3D:
    """Low-latency exponential smoother for 3D landmarks."""

    def __init__(self, alpha=0.45):
        self.alpha = float(np.clip(alpha, 1e-3, 1.0))
        self.value = None

    def update(self, z, reprojection_error=0.0, num_cameras=2):
        # EMA recursion: x_t = alpha*z_t + (1-alpha)*x_{t-1}
        # This is lower-latency than Kalman but does not model velocity/state uncertainty.
        if z is None:
            return np.zeros(3) if self.value is None else self.value.copy()
        z = np.asarray(z, dtype=np.float32)
        if self.value is None:
            self.value = z.copy()
            return self.value.copy()
        self.value = self.alpha * z + (1.0 - self.alpha) * self.value
        return self.value.copy()


class EMA1D:
    """Low-latency exponential smoother for scalar values."""

    def __init__(self, alpha=0.35):
        self.alpha = float(np.clip(alpha, 1e-3, 1.0))
        self.value = None

    def update(self, z):
        # Scalar EMA with identical behavior to EMA3D per-channel update.
        z = float(z)
        if self.value is None:
            self.value = z
            return z
        self.value = self.alpha * z + (1.0 - self.alpha) * self.value
        return self.value


def angle_between(v1, v2):
    # Returns principal angle in degrees between vectors v1 and v2.
    # Dot product is clipped for numerical stability to avoid arccos domain errors.
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm < 1e-8 or v2_norm < 1e-8:
        return 0.0
    v1 = v1 / v1_norm
    v2 = v2 / v2_norm
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def finger_bend_angles(landmarks):
    def joint_angle(a, b, c):
        # Joint angle at b from segments (a-b) and (c-b).
        v1 = a - b
        v2 = c - b
        ang = angle_between(v1, v2)
        # Convert anatomical "straight=0 bent=positive" convention from geometric angle.
        bend = 180.0 - ang
        return np.clip(bend, 0.0, 180.0)

    angles = {}
    fingers = {
        "index": [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring": [13, 14, 15, 16],
        "pinky": [17, 18, 19, 20],
    }
    wrist = landmarks[0]
    for name, (mcp, pip, dip, tip) in fingers.items():
        angles[f"{name}_mcp"] = joint_angle(wrist, landmarks[mcp], landmarks[pip])
        angles[f"{name}_pip"] = joint_angle(
            landmarks[mcp], landmarks[pip], landmarks[dip]
        )
        angles[f"{name}_dip"] = joint_angle(
            landmarks[pip], landmarks[dip], landmarks[tip]
        )
    angles["thumb_cmc_mcp"] = joint_angle(landmarks[1], landmarks[2], landmarks[3])
    angles["thumb_ip"] = joint_angle(landmarks[2], landmarks[3], landmarks[4])
    print(
        "\r[Bend MCP] "
        f"Th:{angles['thumb_cmc_mcp']:.1f}° "
        f"I:{angles['index_mcp']:.1f}° "
        f"M:{angles['middle_mcp']:.1f}° "
        f"R:{angles['ring_mcp']:.1f}° "
        f"P:{angles['pinky_mcp']:.1f}°",
        end="",
        flush=True,
    )
    return angles


# Maps each finger to (MCP index, TIP index) for splay computation.
# MCP→TIP spans the full finger length — maximally stable, minimally affected by jitter.
# Each finger uses wrist → its own MCP as the reference direction.
FINGER_SEGMENTS = {
    "index": (5, 8),  # MCP=5,  PIP=6
    "middle": (9, 12),  # MCP=9,  PIP=10
    "ring": (13, 16),  # MCP=13, PIP=14
    "pinky": (17, 20),  # MCP=17, PIP=18
}

# Legacy bias storage kept for compatibility. Splay now uses an explicit
# per-finger reference vector captured by calibration instead of bias angles.
SPLAY_BIAS_DEG = {
    "index": 0.0,
    "middle": 0.0,
    "ring": 0.0,
    "pinky": 0.0,
}

# Per-finger calibrated reference vectors stored in palm-plane XY coordinates.
SPLAY_REFERENCE_2D = {name: None for name in FINGER_SEGMENTS}

# Approximate physiological MCP abduction/adduction limits (degrees) used to
# compress large projected splay angles into a realistic range.
SPLAY_PHYSIO_MAX_DEG = {
    "index": 25.0,
    "middle": 20.0,
    "ring": 20.0,
    "pinky": 30.0,
}


def _compress_splay_angle(angle_deg: float, finger_name: str) -> float:
    """
    Physiological saturation: linear near 0°, smoothly bounded at ±max_deg.

    scaled = max_deg * tanh(raw / max_deg)
    """
    max_deg = float(SPLAY_PHYSIO_MAX_DEG.get(finger_name, 25.0))
    if max_deg <= 1e-6:
        return angle_deg
    return max_deg * np.tanh(angle_deg / max_deg)


def _coerce_landmarks_3d(landmarks):
    lm = np.array(landmarks, dtype=float)
    if lm.ndim != 2 or lm.shape[0] != 21:
        return None
    if lm.shape[1] == 2:
        lm = np.hstack([lm, np.zeros((21, 1))])
    return lm


def _build_palm_frame(lm):
    # Palm normal from knuckle row × proximal-distal axis
    u = lm[5] - lm[17]  # index_MCP − pinky_MCP
    v = lm[9] - lm[0]  # middle_MCP − wrist
    n = np.cross(u, v)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-6:
        return None
    n = n / n_norm

    ex = u - np.dot(u, n) * n
    ex_norm = np.linalg.norm(ex)
    if ex_norm < 1e-6:
        return None
    ex = ex / ex_norm

    ey = np.cross(n, ex)
    ey_norm = np.linalg.norm(ey)
    if ey_norm < 1e-6:
        return None
    ey = ey / ey_norm
    return n, ex, ey


def set_splay_reference(landmarks):
    """Capture the current projected MCP→TIP directions as splay references."""
    lm = _coerce_landmarks_3d(landmarks)
    if lm is None:
        return None

    frame = _build_palm_frame(lm)
    if frame is None:
        return None
    _, ex, ey = frame

    def to_palm_xy(vec3: np.ndarray) -> np.ndarray:
        return np.array([np.dot(vec3, ex), np.dot(vec3, ey)])

    refs = {}
    for name, (mcp_idx, tip_idx) in FINGER_SEGMENTS.items():
        ref_2d = to_palm_xy(lm[tip_idx] - lm[mcp_idx])
        ref_norm = np.linalg.norm(ref_2d)
        if ref_norm < 1e-6:
            continue
        ref_2d = ref_2d / ref_norm
        ref_2d[1] = abs(ref_2d[1])
        refs[name] = ref_2d

    if not refs:
        return None

    for name in FINGER_SEGMENTS:
        SPLAY_REFERENCE_2D[name] = refs.get(name)
    return {
        name: (None if vec is None else vec.copy())
        for name, vec in SPLAY_REFERENCE_2D.items()
    }


def clear_splay_reference():
    for name in FINGER_SEGMENTS:
        SPLAY_REFERENCE_2D[name] = None


# ── v2 splay: scalar-offset zeroing ──────────────────────────────────────────
# Tip landmark indices per finger (wrist→tip vectors used in v2).
_FINGER_TIPS_V2 = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
_MIDDLE_TIP_IDX = 12  # wrist→middle_tip is the shared reference

# Scalar angle offsets captured on button press. Each frame subtracts these
# so the captured pose reads 0°.
SPLAY_ZERO_OFFSETS: dict[str, float] = {name: 0.0 for name in _FINGER_TIPS_V2}

# Reference unit vector (palm-plane 2D) frozen at button press.
# Stored relative to the palm frame so it tracks natural palm orientation.
# Mutable dict so importers always see the live value through the same object.
# SPLAY_REF_STORE["unit_2d"] is None until set_splay_zero() is called.
SPLAY_REF_STORE: dict = {"unit_2d": None}


def set_splay_zero(landmarks):
    """
    Freeze the current wrist→middle_tip direction as the shared reference
    vector (in palm-plane 2D), then measure each finger's current angle
    against it and store those as SPLAY_ZERO_OFFSETS.

    After this call, finger_splay_angles_v2 will output 0° for the pose
    captured here, and non-zero for any deviation from it.

    Returns {name: offset_deg} on success, None on failure.
    """
    lm = _coerce_landmarks_3d(landmarks)
    if lm is None:
        return None
    frame = _build_palm_frame(lm)
    if frame is None:
        return None
    _, ex, ey = frame

    def to_palm_xy(vec3: np.ndarray) -> np.ndarray:
        return np.array([np.dot(vec3, ex), np.dot(vec3, ey)])

    v_ref_2d = to_palm_xy(lm[_MIDDLE_TIP_IDX] - lm[0])
    ref_norm = np.linalg.norm(v_ref_2d)
    if ref_norm < 1e-6:
        return None
    v_ref_2d = v_ref_2d / ref_norm

    # Freeze this as the shared reference for all future frames.
    SPLAY_REF_STORE["unit_2d"] = v_ref_2d.copy()

    for name, tip_idx in _FINGER_TIPS_V2.items():
        v_f_2d = to_palm_xy(lm[tip_idx] - lm[0])
        f_norm = np.linalg.norm(v_f_2d)
        if f_norm < 1e-6:
            SPLAY_ZERO_OFFSETS[name] = 0.0
            continue
        v_f_2d = v_f_2d / f_norm
        signed_sin = v_ref_2d[0] * v_f_2d[1] - v_ref_2d[1] * v_f_2d[0]
        signed_cos = np.dot(v_ref_2d, v_f_2d)
        SPLAY_ZERO_OFFSETS[name] = np.degrees(np.arctan2(signed_sin, signed_cos))

    return {name: SPLAY_ZERO_OFFSETS[name] for name in _FINGER_TIPS_V2}


def clear_splay_zero():
    """Reset all splay zero offsets and the frozen reference to 0°/None."""
    SPLAY_REF_STORE["unit_2d"] = None
    for name in _FINGER_TIPS_V2:
        SPLAY_ZERO_OFFSETS[name] = 0.0


def finger_splay_angles_v2(landmarks) -> dict[str, float]:
    """
    Compute per-finger splay angles using wrist→tip vectors (v2 method).

    Reference vector: wrist (idx 0) → middle_tip (idx 12), projected onto palm plane.
    Per-finger vector: wrist (idx 0) → that finger's tip, projected onto palm plane.

    Raw angle = signed angle (CCW+) from reference to finger vector in palm plane.
    Zeroed angle = raw_angle − SPLAY_ZERO_OFFSETS[name]  (button press sets zero).
    Output = tanh-compressed zeroed angle.
    """
    lm = _coerce_landmarks_3d(landmarks)
    if lm is None:
        return {f"{name}_splay": 0.0 for name in _FINGER_TIPS_V2}
    frame = _build_palm_frame(lm)
    if frame is None:
        return {f"{name}_splay": 0.0 for name in _FINGER_TIPS_V2}
    _, ex, ey = frame

    def to_palm_xy(vec3: np.ndarray) -> np.ndarray:
        return np.array([np.dot(vec3, ex), np.dot(vec3, ey)])

    if SPLAY_REF_STORE["unit_2d"] is not None:
        # Use the frozen reference direction (captured at button press).
        v_ref_2d = np.array(SPLAY_REF_STORE["unit_2d"], dtype=float)
    else:
        # No reference set yet — compute from current middle tip (middle reads 0°).
        v_ref_2d = to_palm_xy(lm[_MIDDLE_TIP_IDX] - lm[0])
        ref_norm = np.linalg.norm(v_ref_2d)
        if ref_norm < 1e-6:
            return {f"{name}_splay": 0.0 for name in _FINGER_TIPS_V2}
        v_ref_2d = v_ref_2d / ref_norm

    angles = {}
    pinky_v_f_2d = None
    for name, tip_idx in _FINGER_TIPS_V2.items():
        v_f_2d = to_palm_xy(lm[tip_idx] - lm[0])
        f_norm = np.linalg.norm(v_f_2d)
        if f_norm < 1e-6:
            angles[f"{name}_splay"] = 0.0
            continue
        v_f_2d = v_f_2d / f_norm
        if name == "pinky":
            pinky_v_f_2d = v_f_2d.copy()
        signed_sin = v_ref_2d[0] * v_f_2d[1] - v_ref_2d[1] * v_f_2d[0]
        signed_cos = np.dot(v_ref_2d, v_f_2d)
        raw_angle = np.degrees(np.arctan2(signed_sin, signed_cos))
        zeroed = raw_angle - SPLAY_ZERO_OFFSETS[name]
        angles[f"{name}_splay"] = _compress_splay_angle(-zeroed, name)

    debug_msg = (
        f"\r[Splay v2] "
        f"I:{angles['index_splay']:.1f}° "
        f"M:{angles['middle_splay']:.1f}° "
        f"R:{angles['ring_splay']:.1f}° "
        f"P:{angles['pinky_splay']:.1f}°"
    )
    if pinky_v_f_2d is not None:
        debug_msg += f" | pinky_v_f_2d: [{pinky_v_f_2d[0]:.2f}, {pinky_v_f_2d[1]:.2f}]"
    print(debug_msg, end="", flush=True)
    return angles


def finger_splay_angles(landmarks) -> dict[str, float]:
    """
    Compute per-finger lateral splay angles projected onto the palm plane.

    Each finger's reference direction is wrist → that finger's own MCP,
    projected onto the palm plane. This means each finger's splay is measured
    relative to its own neutral metacarpal axis, not a shared middle-finger axis.

    Works correctly for 3D world-space input (21, 3) from stereo triangulation.
    For legacy 2D input (21, 2) the math reduces to a 2D atan2 calculation.

    Sign: positive = CCW (spreading away from middle), negative = CW.
    """
    lm = _coerce_landmarks_3d(landmarks)
    if lm is None:
        return {f"{name}_splay": 0.0 for name in FINGER_SEGMENTS}
    frame = _build_palm_frame(lm)
    if frame is None:
        return {f"{name}_splay": 0.0 for name in FINGER_SEGMENTS}
    _, ex, ey = frame

    def to_palm_xy(vec3: np.ndarray) -> np.ndarray:
        return np.array([np.dot(vec3, ex), np.dot(vec3, ey)])

    angles = {}
    pinky_v_f_2d = None
    for name, (mcp_idx, tip_idx) in FINGER_SEGMENTS.items():
        v_f_2d = to_palm_xy(lm[tip_idx] - lm[mcp_idx])
        f_norm = np.linalg.norm(v_f_2d)
        if f_norm < 1e-6:
            angles[f"{name}_splay"] = 0.0
            continue
        v_f_2d = v_f_2d / f_norm
        v_f_2d[1] = abs(v_f_2d[1])

        stored_ref = SPLAY_REFERENCE_2D.get(name)
        if stored_ref is None:
            v_ref_2d = v_f_2d.copy()
        else:
            v_ref_2d = stored_ref.copy()
            ref_norm = np.linalg.norm(v_ref_2d)
            if ref_norm < 1e-6:
                v_ref_2d = v_f_2d.copy()
            else:
                v_ref_2d = v_ref_2d / ref_norm

        # Store pinky's v_f_2d for debug output
        if name == "pinky":
            pinky_v_f_2d = v_f_2d.copy()

        signed_sin = v_ref_2d[0] * v_f_2d[1] - v_ref_2d[1] * v_f_2d[0]
        signed_cos = np.dot(v_ref_2d, v_f_2d)
        raw_angle = -np.degrees(np.arctan2(signed_sin, signed_cos))
        angles[f"{name}_splay"] = _compress_splay_angle(raw_angle, name)

    # Debug output: print all splay angles and pinky v_f_2d
    debug_msg = (
        f"\r[Splay] I:{angles['index_splay']:.1f}° "
        f"M:{angles['middle_splay']:.1f}° "
        f"R:{angles['ring_splay']:.1f}° "
        f"P:{angles['pinky_splay']:.1f}°"
    )
    if pinky_v_f_2d is not None:
        debug_msg += f" | pinky_v_f_2d: [{pinky_v_f_2d[0]:.2f}, {pinky_v_f_2d[1]:.2f}]"
    print(debug_msg, end="", flush=True)
    return angles
