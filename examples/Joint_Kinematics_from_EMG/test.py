"""
overlay_landmarks_video.py

Overlays smoothed landmarks onto the video frames.
Each landmark sample is mapped directly to a single frame (1:1).
This ensures the video frames and landmark samples are synchronized visually.

Author: Jonathan Shulgach
Date: 06/03/25
"""

import numpy as np
import cv2
import torch
from src.handtrack.ml import EMGRegressor

# ---------- CONFIG ----------
VIDEO_PATH = r"G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\Open_Ephys\Jonathan\2025_05_07\media\HandDynamic.mp4"
LANDMARKS_FILE = '../../data/HandDynamic_smoothed_landmarks.npz'
EMG_DATA_FILE = '../../data/hand_dynamic_dataset.npz'
MODEL_PATH = '../../data/emg_regressor.pth'
SCALER_MEAN_PATH = '../../data/feature_scaler.npy'
SCALER_STD_PATH = '../../data/feature_scaler_std.npy'

# ---------- LOAD DATA ----------
print("Loading landmarks...")
landmarks_data = np.load(LANDMARKS_FILE)
landmarks = landmarks_data['landmarks']
lm_fs = int(landmarks_data['sampling_rate'])
print(f"Loaded landmarks shape: {landmarks.shape} at {lm_fs} Hz.")

print("Loading EMG features...")
emg_data = np.load(EMG_DATA_FILE)
emg_features = emg_data['emg_features']
print(f"Loaded EMG features shape: {emg_features.shape}")

print("Loading model and scaler...")
mean = np.load(SCALER_MEAN_PATH)
std = np.load(SCALER_STD_PATH)
scaler = lambda x: (x - mean) / std

# Load model
model = EMGRegressor(emg_features.shape[1], landmarks.shape[1] * landmarks.shape[2])
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Scale EMG features
X_scaled = scaler(emg_features)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

print("Predicting landmarks from EMG...")
with torch.no_grad():
    predicted_landmarks = model(X_tensor).numpy().reshape((-1, 21, 3))
print(f"Predicted landmarks shape: {predicted_landmarks.shape}")

# ---------- HAND CONNECTIONS ----------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring
    (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

# ---------- LOAD VIDEO ----------
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "Failed to open video file."

video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
print(f"Video has {video_frame_count} frames at {video_fps:.2f} fps.")

# Calculate approximate video time vector
video_time = np.arange(video_frame_count) / video_fps

# Calculate approximate landmark time vector
lm_time = np.arange(landmarks.shape[0]) / lm_fs

# Calculate approximate EMG feature time vector
emg_fs = 5000  # Default, adjust if stored somewhere
window_ms = 50  # Should match dataset creation
step_ms = 20
window_size = int(window_ms * emg_fs / 1000)
step_size = int(step_ms * emg_fs / 1000)
emg_time = np.arange(emg_features.shape[0]) * (step_size / emg_fs)

# Align EMG predictions with video frames (simple nearest-neighbor matching)
frame_indices = np.round(video_time * (1 / (step_size / emg_fs))).astype(int)
frame_indices = np.clip(frame_indices, 0, emg_features.shape[0] - 1)

total_frames = min(video_frame_count, landmarks.shape[0])
print(f"Using {total_frames} frames for overlay.")

frame_idx = 0
print("Starting overlay visualization... Press ESC to exit.")
while cap.isOpened() and frame_idx < total_frames:
    ret, frame = cap.read()
    if not ret:
        break

    # Ground-truth landmark
    gt_landmark_frame = landmarks[frame_idx]

    # Predicted landmark
    emg_idx = frame_indices[frame_idx]
    pred_landmark_frame = predicted_landmarks[emg_idx]

    # Draw ground-truth landmarks (blue)
    for lm in gt_landmark_frame:
        cx = int(lm[0] * frame.shape[1])
        cy = int(lm[1] * frame.shape[0])
        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)  # Blue

    # Draw predicted landmarks (cyan)
    for lm in pred_landmark_frame:
        cx = int(lm[0] * frame.shape[1])
        cy = int(lm[1] * frame.shape[0])
        cv2.circle(frame, (cx, cy), 4, (255, 255, 0), -1)  # Cyan

    # Draw connections for ground-truth (blue)
    for connection in HAND_CONNECTIONS:
        idx_start, idx_end = connection
        x0 = int(gt_landmark_frame[idx_start][0] * frame.shape[1])
        y0 = int(gt_landmark_frame[idx_start][1] * frame.shape[0])
        x1 = int(gt_landmark_frame[idx_end][0] * frame.shape[1])
        y1 = int(gt_landmark_frame[idx_end][1] * frame.shape[0])
        cv2.line(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)

    # Draw connections for predicted (cyan)
    for connection in HAND_CONNECTIONS:
        idx_start, idx_end = connection
        x0 = int(pred_landmark_frame[idx_start][0] * frame.shape[1])
        y0 = int(pred_landmark_frame[idx_start][1] * frame.shape[0])
        x1 = int(pred_landmark_frame[idx_end][0] * frame.shape[1])
        y1 = int(pred_landmark_frame[idx_end][1] * frame.shape[0])
        cv2.line(frame, (x0, y0), (x1, y1), (255, 255, 0), 2)

    cv2.imshow("Landmark Overlay", frame)
    if cv2.waitKey(int(1000 / video_fps)) & 0xFF == 27:
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

print("Visualization complete!")