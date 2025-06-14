"""
visualize_predictions.py

Visualizes predicted landmarks from EMG features overlaid on the original video.
Loads the trained EMG regressor model and scaler, aligns EMG data to video frames,
and overlays predictions dynamically frame by frame.

Author: Jonathan Shulgach
Date: 06/03/25
"""

import numpy as np
import cv2
import torch
from src.handtrack.ml import EMGRegressor  # Update this import path as needed

# ---------- CONFIG ----------
VIDEO_PATH = r"G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\Open_Ephys\Jonathan\2025_05_07\media\HandDynamic.mp4"
MODEL_PATH = '../../data/emg_regressor.pth'
SCALER_MEAN_PATH = '../../data/feature_scaler.npy'
SCALER_STD_PATH = '../../data/feature_scaler_std.npy'
EMG_DATASET_PATH = '../../data/hand_dynamic_dataset.npz'
SYNC_OFFSET_FILE = '../../data/sync_offset.txt'
EMG_DATA_FILE = '../../data/emg_data.npz'

# ---------- LOAD DATA ----------
print("Loading model and scaler...")
mean = np.load(SCALER_MEAN_PATH)
std = np.load(SCALER_STD_PATH)
scaler = lambda x: (x - mean) / std

# Load EMG dataset
emg_data = np.load(EMG_DATASET_PATH)
X_full = emg_data['emg_features']
landmark_labels = emg_data['landmark_labels']

# Load sync offset
with open(SYNC_OFFSET_FILE, 'r') as f:
    sync_offset = float(f.readlines()[0].split(': ')[1].strip().split(' ')[0])
print(f"Loaded sync offset: {sync_offset:.3f} seconds")

# Load EMG time vector
result = np.load(EMG_DATA_FILE)
emg_time_vector = result['time_vector']

# ---------- LOAD MODEL ----------
model = EMGRegressor(X_full.shape[1], landmark_labels.shape[1])
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# ---------- LOAD VIDEO ----------
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "Failed to open video file."
video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_time_vector = np.arange(n_frames) / video_fps

# ---------- ALIGN EMG to VIDEO ----------
emg_time_aligned = emg_time_vector  # + sync_offset
emg_indices_for_frames = np.searchsorted(emg_time_aligned, video_time_vector, side='left')
emg_indices_for_frames = np.clip(emg_indices_for_frames, 0, X_full.shape[0] - 1)

# ---------- HAND CONNECTIONS ----------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring
    (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

# ---------- VISUALIZE PREDICTIONS ----------
frame_idx = 0

print("Starting visualization... Press ESC to exit.")
while cap.isOpened() and frame_idx < n_frames:
    ret, frame = cap.read()
    if not ret:
        break

    # Dynamic prediction per frame
    emg_index = emg_indices_for_frames[frame_idx]
    X_frame = X_full[emg_index]
    X_frame_scaled = scaler(X_frame)
    X_frame_tensor = torch.tensor(X_frame_scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred_landmarks = model(X_frame_tensor).numpy().reshape((21, 3))

    true_landmarks = landmark_labels[min(frame_idx, landmark_labels.shape[0] - 1)].reshape((21, 3))

    # Draw ground-truth landmarks (blue)
    for lm in true_landmarks:
        cx = int(lm[0] * frame.shape[1])
        cy = int(lm[1] * frame.shape[0])
        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)  # Blue

    # Draw predicted landmarks (cyan)
    for lm in pred_landmarks:
        cx = int(lm[0] * frame.shape[1])
        cy = int(lm[1] * frame.shape[0])
        cv2.circle(frame, (cx, cy), 4, (255, 255, 0), -1)  # Cyan

    # Draw connections for ground-truth (blue)
    for connection in HAND_CONNECTIONS:
        idx_start, idx_end = connection
        x0 = int(true_landmarks[idx_start][0] * frame.shape[1])
        y0 = int(true_landmarks[idx_start][1] * frame.shape[0])
        x1 = int(true_landmarks[idx_end][0] * frame.shape[1])
        y1 = int(true_landmarks[idx_end][1] * frame.shape[0])
        cv2.line(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)

    # Draw connections for predicted (cyan)
    for connection in HAND_CONNECTIONS:
        idx_start, idx_end = connection
        x0 = int(pred_landmarks[idx_start][0] * frame.shape[1])
        y0 = int(pred_landmarks[idx_start][1] * frame.shape[0])
        x1 = int(pred_landmarks[idx_end][0] * frame.shape[1])
        y1 = int(pred_landmarks[idx_end][1] * frame.shape[0])
        cv2.line(frame, (x0, y0), (x1, y1), (255, 255, 0), 2)

    # Show the frame
    cv2.imshow("Predicted Landmarks", frame)
    if cv2.waitKey(int(1000 / video_fps)) & 0xFF == 27:
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

print("Visualization complete!")
