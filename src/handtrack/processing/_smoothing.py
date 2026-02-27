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
    return angles


def finger_splay_angles(landmarks):
    # Reference axis: wrist -> middle fingertip.
    wrist = landmarks[0]
    ref = landmarks[12] - wrist
    ref_norm = np.linalg.norm(ref)
    if ref_norm < 1e-8:
        return {}
    ref = ref / ref_norm

    # Palm normal disambiguates sign for splay (left/right of reference axis).
    v1 = landmarks[8] - wrist
    v2 = landmarks[20] - wrist
    palm_normal = np.cross(v1, v2)
    norm = np.linalg.norm(palm_normal)
    if norm < 1e-8:
        return {}
    palm_normal = palm_normal / norm

    tips = {
        "index": landmarks[8],
        "middle": landmarks[12],
        "ring": landmarks[16],
        "pinky": landmarks[20],
    }
    rest_offsets = {
        "index": 0.0,
        "middle": 0.0,
        "ring": 0.0,
        "pinky": -15.383,
    }

    angles = {}
    for name, tip in tips.items():
        finger_vec = tip - wrist
        finger_norm = np.linalg.norm(finger_vec)
        if finger_norm < 1e-8:
            angles[f"{name}_splay"] = rest_offsets[name]
            continue
        finger_vec = finger_vec / finger_norm
        # Unsigned angle from reference axis, then signed using palm normal.
        dot = np.clip(np.dot(finger_vec, ref), -1.0, 1.0)
        angle = np.degrees(np.arccos(dot))
        cross = np.cross(ref, finger_vec)
        if np.dot(cross, palm_normal) < 0:
            angle = -angle

        # Optional rest offsets align with a neutral hand posture baseline.
        angles[f"{name}_splay"] = angle + rest_offsets[name]
    return angles
