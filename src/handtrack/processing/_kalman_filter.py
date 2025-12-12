# kalman_filter.py

import numpy as np


class Kalman3D:
    def __init__(self, dt=1/30, process_noise=1e-3, measurement_noise=1e-2):
        self.x = np.zeros((6, 1))
        self.F = np.eye(6)
        for i in range(3): self.F[i, i+3] = dt
        self.H = np.hstack((np.eye(3), np.zeros((3, 3))))
        self.P = np.eye(6)
        self.Q = np.eye(6) * process_noise
        self.R = np.eye(3) * measurement_noise

    def update(self, z):
        z = np.reshape(z, (3, 1))
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x[:3].flatten()


# Need to create a class that can smooth out a single angle
class KalmanAngle:
    def __init__(self, dt=1/30, process_noise=1e-3, measurement_noise=1e-2):
        self.x = np.zeros((2, 1))  # State: [angle, angular velocity]
        self.F = np.array([[1, dt], [0, 1]])  # State transition matrix
        self.H = np.array([[1, 0]])  # Measurement matrix
        self.P = np.eye(2)  # Estimate uncertainty
        self.Q = np.eye(2) * process_noise  # Process noise covariance
        self.R = np.eye(1) * measurement_noise  # Measurement noise covariance

    def update(self, z):
        z = np.reshape(z, (1, 1))  # Ensure z is a column vector
        self.x = self.F @ self.x  # Predict step
        self.P = self.F @ self.P @ self.F.T + self.Q  # Update estimate uncertainty
        y = z - self.H @ self.x  # Measurement residual
        S = self.H @ self.P @ self.H.T + self.R  # Residual covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x += K @ y  # Update state with measurement
        self.P = (np.eye(2) - K @ self.H) @ self.P  # Update estimate uncertainty
        return float(self.x[0])  # Return the angle estimate