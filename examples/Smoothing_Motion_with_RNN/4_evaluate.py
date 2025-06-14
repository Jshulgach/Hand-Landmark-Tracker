# 4_evaluate_and_visualize.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# ----------------------------
# Config
# ----------------------------
DATA_DIR = "data"
Y_KALMAN = os.path.join(DATA_DIR, "Y_smooth.npy")
Y_GRU = os.path.join(DATA_DIR, "Y_pred.npy")
X_RAW = os.path.join(DATA_DIR, "X_raw.npy")
SAVE_FIG = "trajectory_comparison.png"
LANDMARK_INDEX = 12  # Index fingertip
SAMPLE_INDEX = 0

# ----------------------------
# Load Data
# ----------------------------
kalman = np.load(Y_KALMAN)     # [N, T, 21, 3]
print(f"SHape of Kalman data: {kalman.shape}")
gru = np.load(Y_GRU)
print(f"Shape of GRU data: {gru.shape}")
raw = np.load(X_RAW)
print(f"Shape of raw data: {raw.shape}")

# ----------------------------
# MSE Evaluation
# ----------------------------
mse_all = []
for i in range(21):  # landmarks
    mse = mean_squared_error(
        kalman[:, :, i, :].reshape(-1, 3),
        gru[:, :, i, :].reshape(-1, 3)
    )
    mse_all.append(mse)

print("Landmark-wise MSE (Kalman vs GRU):")
for idx, mse in enumerate(mse_all):
    print(f"Landmark {idx:2d}: MSE = {mse:.6f}")


def stitch_last_frames(windows, n_frames):
    return np.stack([windows[i, -1] for i in range(n_frames)], axis=0)  # shape: [n_frames, 21, 3]


# ----------------------------
# Trajectory Comparison Plot
# ----------------------------
# Extract 50 frames from stitched stream
n_frames = 300

raw_50 = stitch_last_frames(raw, n_frames)
kalman_50 = stitch_last_frames(kalman, n_frames)
gru_50 = stitch_last_frames(gru, n_frames)

fig = plt.figure(figsize=(12, 4))
for d, axis in enumerate(['X', 'Y', 'Z']):
    plt.subplot(1, 3, d + 1)
    plt.plot(raw_50[:, LANDMARK_INDEX, d], 'g--', label='Raw')
    plt.plot(kalman_50[:, LANDMARK_INDEX, d], 'b-', label='Kalman')
    plt.plot(gru_50[:, LANDMARK_INDEX, d], 'r-', label='GRU')
    plt.title(f'{axis}-axis')
    plt.xlabel('Frame')
    plt.ylabel('Position')
    plt.grid(True)
    if d == 0:
        plt.legend()

plt.suptitle(f"Landmark {LANDMARK_INDEX} - Over {n_frames} Frames")
plt.tight_layout()
plt.savefig("trajectory_50frames.png")
plt.show()

