"""
predict_from_full_dataset.py

Overlays both smoothed landmarks (ground truth) and predicted landmarks (from EMG features)
onto the video frames. Each video frame is mapped to a corresponding EMG feature prediction
accounting for the EMG sampling rate, step size, and sync offset.

Author: Jonathan Shulgach
Date: 06/03/25
"""

import numpy as np
import cv2
import torch
from src.handtrack.ml import EMGRegressor  # Assuming you saved it in your package
from src.handtrack.processing import Kalman3D

# ---------- CONFIG ----------
VIDEO_PATH = r"G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\Open_Ephys\Jonathan\2025_05_07\media\HandDynamic.mp4"
LANDMARKS_FILE = '../../data/HandDynamic_smoothed_landmarks.npz'
EMG_DATA_FILE = '../../data/hand_dynamic_dataset.npz'
MODEL_PATH = '../../data/emg_regressor.pth'
SCALER_MEAN_PATH = '../../data/feature_scaler.npy'
SCALER_STD_PATH = '../../data/feature_scaler_std.npy'
SYNC_OFFSET_FILE = '../../data/sync_offset.txt'  # File with sync offset value

# ---------- LOAD SYNC OFFSET ----------
with open(SYNC_OFFSET_FILE, 'r') as f:
    sync_offset = float(f.readlines()[0].split(': ')[1].strip().split(' ')[0])
print(f"Loaded sync offset: {sync_offset:.3f} seconds.")

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
# Initialize Kalman filters for each landmark
kalman_filters = [Kalman3D() for _ in range(21)]

# ---------- LOAD VIDEO ----------
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "Failed to open video file."

video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
print(f"Video has {video_frame_count} frames at {video_fps:.2f} fps.")

# ---------- ALIGN EMG FEATURES TO VIDEO FRAMES ----------
emg_fs = 5000  # EMG sampling rate
window_ms = 50
step_ms = 10
step_size = int(step_ms * emg_fs / 1000)
emg_time = np.arange(emg_features.shape[0]) * (step_size / emg_fs)# + sync_offset

video_time = np.arange(landmarks.shape[0]) / video_fps + sync_offset # Match video frames to landmarks

# Downsample EMG features to match landmark samples
frame_to_emg_idx = np.searchsorted(emg_time, video_time, side='right') - 1
frame_to_emg_idx = np.clip(frame_to_emg_idx, 0, emg_features.shape[0] - 1)
emg_features_downsampled = emg_features[frame_to_emg_idx]
print(f"Downsampled EMG features shape: {emg_features_downsampled.shape}")

# Scale EMG features
X_scaled = scaler(emg_features_downsampled)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# Predict landmarks
print("Predicting landmarks from EMG...")
with torch.no_grad():
    predicted_landmarks = model(X_tensor).numpy().reshape((-1, 21, 3))
print(f"Predicted landmarks shape: {predicted_landmarks.shape}")

# Apply Kalman filter frame-by-frame
smoothed_predictions = []
for t in range(predicted_landmarks.shape[0]):
    filtered_frame = []
    for idx in range(21):
        filtered_pos = kalman_filters[idx].update(predicted_landmarks[t, idx])
        filtered_frame.append(filtered_pos)
    smoothed_predictions.append(filtered_frame)

smoothed_predictions = np.array(smoothed_predictions)  # Shape: (num_frames, 21, 3)

# ---------- HAND CONNECTIONS ----------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring
    (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

# ---------- VISUALIZATION ----------
total_frames = min(video_frame_count, landmarks.shape[0])
print(f"Using {total_frames} frames for overlay visualization.")

frame_idx = 0
print("Starting overlay visualization... Press ESC to exit.")
while cap.isOpened() and frame_idx < total_frames:
    ret, frame = cap.read()
    if not ret:
        break

    # Ground-truth landmark
    gt_landmark_frame = landmarks[frame_idx]

    # Predicted landmark
    pred_landmark_frame = smoothed_predictions[frame_idx]

    # Draw text annotations
    cv2.putText(frame,"Mediapipe (Ground Truth)", (1000, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.7,(139, 0, 0), 2, cv2.LINE_AA)

    # Draw ground-truth landmarks (blue)
    for lm in gt_landmark_frame:
        cx = int(lm[0] * frame.shape[1])
        cy = int(lm[1] * frame.shape[0])
        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)  # Blue

    # Draw connections for ground-truth (blue)
    for connection in HAND_CONNECTIONS:
        idx_start, idx_end = connection
        x0 = int(gt_landmark_frame[idx_start][0] * frame.shape[1])
        y0 = int(gt_landmark_frame[idx_start][1] * frame.shape[0])
        x1 = int(gt_landmark_frame[idx_end][0] * frame.shape[1])
        y1 = int(gt_landmark_frame[idx_end][1] * frame.shape[0])
        cv2.line(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)

    cv2.putText(frame,"Prediction from EMG",(1000, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.7,(255, 255, 0), 2, cv2.LINE_AA)

    # Draw predicted landmarks (cyan)
    for lm in pred_landmark_frame:
        cx = int(lm[0] * frame.shape[1])
        cy = int(lm[1] * frame.shape[0])
        cv2.circle(frame, (cx, cy), 4, (255, 255, 0), -1)  # Cyan

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
