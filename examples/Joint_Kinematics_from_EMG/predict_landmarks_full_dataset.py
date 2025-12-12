"""
predict_landmarks_full_dataset.py

Visualizes predicted vs ground-truth hand landmarks from EMG features using a trained model.
Landmarks are overlaid onto video frames and displayed in real-time.

Author: Jonathan Shulgach
Date: 06/03/25
"""

import os
import numpy as np
import pickle
import cv2
import torch
import logging
import argparse
from scipy.interpolate import interp1d
from handtrack.ml import EMGRegressor
from handtrack.processing import Kalman3D
from handtrack.tracker import get_hand_connections

# Hand connections (for drawing)
HAND_CONNECTIONS = get_hand_connections()


def parse_events_file(file_path):
    """Extract Start and End indices from a single event file."""
    start_idx, end_idx = None, None
    with open(file_path, 'r') as f:
        next(f)  # Skip header line
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue  # Skip malformed lines
            sample_index_str, _, label = parts
            if label == 'Start' and start_idx is None:
                start_idx = int(sample_index_str)
            elif label == 'End' and end_idx is None:
                end_idx = int(sample_index_str)
            if start_idx is not None and end_idx is not None:
                break
    return start_idx, end_idx


def setup_logger():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def run_visualization(root_dir, label=None):
    setup_logger()

    if label is None:
        label = "HandDynamic"

    logging.info(f"Using label: {label}")

    # Define expected filenames in root_dir
    video_path = os.path.join(root_dir, 'media', f"{label}.mp4")
    landmarks_file = os.path.join(root_dir, 'landmarks', f"{label}_smoothed_landmarks.npz")
    emg_file = os.path.join(root_dir, f"{label}_feature_dataset.npz")
    model_path = os.path.join(root_dir, 'model', f"{label}_emg_regressor.pth")
    scaler_path = os.path.join(root_dir, 'model', f"{label}_scaler.pkl")
    sync_offset_file = os.path.join(root_dir, 'events', f"{label}_sync_offset.txt")

    # Sanity check files
    for f in [video_path, landmarks_file, emg_file, model_path, scaler_path, sync_offset_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing required file: {f}")

    video_events_file = os.path.join(root_dir, 'events', f"{label}_video_events.txt")
    if not os.path.exists(video_events_file):
        raise FileNotFoundError(f"Missing video events file: {video_events_file}")
    start_frame_index, _ = parse_events_file(video_events_file)
    logging.info(f"Parsed start frame index from video events: {start_frame_index}")

    # Load sync offset
    with open(sync_offset_file, 'r') as f:
        sync_offset = float(f.readline().split(': ')[1].split(' ')[0])
    logging.info(f"Loaded sync offset: {sync_offset:.3f} seconds.")

    # Load landmark ground truth
    logging.info("Loading landmarks...")
    lm_data = np.load(landmarks_file)
    landmarks = lm_data['landmarks']
    lm_fs = int(lm_data['sampling_rate'])
    print(f"Landmarks shape: {landmarks.shape}, Sampling rate: {lm_fs} Hz")

    # Load EMG features
    logging.info("Loading EMG features...")
    emg_data = np.load(emg_file)
    emg_features = emg_data['emg_features']
    emg_fs = int(emg_data.get('emg_fs', 5000))

    window_ms = int(emg_data.get('window_ms', 50))
    window_size = int(window_ms * emg_fs / 1000)
    step_ms = int(emg_data.get('step_ms', 10))
    step_size = int(step_ms * emg_fs / 1000)
    logging.info(f"Loaded EMG features: {emg_features.shape} | EMG FS: {emg_fs} Hz")

    # Load scaler
    logging.info("Loading model and scaler...")
    print(f" Using scalar path: {scaler_path}")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Load model
    model = EMGRegressor(emg_features.shape[1], landmarks.shape[1] * landmarks.shape[2])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Open video
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Failed to open video file: {video_path}"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    #cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
    logging.info(f"Video has {frame_count} frames at {video_fps:.2f} fps.")

    # Align EMG to video using time offset
    print(f"Using adjusted offset time: {sync_offset:.3f} seconds")

    # Flatten EMG features for interpolation
    emg_time = np.arange(emg_features.shape[0]) * (step_size / emg_fs)  # already adjusted by step
    video_time = np.arange(landmarks.shape[0]) / video_fps - sync_offset  # already sync-adjusted
    #video_time = np.arange(landmarks.shape[0]) / video_fps

    #adjusting the start index to also include the offset
    #start_frame_index += int(abs(sync_offset) * video_fps)

    # Interpolate each EMG feature dimension
    interp_func = interp1d(emg_time, emg_features, axis=0, kind='linear', fill_value='extrapolate')
    emg_features_interpolated = interp_func(video_time)

    print(f"Interpolated EMG features shape: {emg_features_interpolated.shape}")

    # Predict landmarks
    logging.info("Predicting landmarks from EMG...")
    X_scaled = scaler.transform(emg_features_interpolated)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        predicted_landmarks = model(X_tensor).numpy().reshape((-1, landmarks.shape[1], 3))

    # Kalman filter smoothing
    kalman_filters = [Kalman3D() for _ in range(21)]
    smoothed = np.stack([[kf.update(predicted_landmarks[t, j]) for j, kf in enumerate(kalman_filters)] for t in range(predicted_landmarks.shape[0])])

    # Begin overlay visualization
    total_frames = min(frame_count, landmarks.shape[0])
    logging.info("Beginning visualization... press ESC to quit.")

    # Create a mask to skip prediction display before the start frame index
    frame_idx = 0
    while cap.isOpened() and frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Ground-truth landmark
        gt_frame = landmarks[frame_idx]

        # Only draw predictions after the "Start" event
        if frame_idx >= start_frame_index:
            pred_frame = smoothed[frame_idx]

            # Draw Predicted landmarks
            for lm in pred_frame:
                cx, cy = int(lm[0] * frame.shape[1]), int(lm[1] * frame.shape[0])
                cv2.circle(frame, (cx, cy), 4, (255, 255, 0), -1)  # Cyan

            for c in HAND_CONNECTIONS:
                x0, y0 = int(pred_frame[c[0]][0] * frame.shape[1]), int(pred_frame[c[0]][1] * frame.shape[0])
                x1, y1 = int(pred_frame[c[1]][0] * frame.shape[1]), int(pred_frame[c[1]][1] * frame.shape[0])
                cv2.line(frame, (x0, y0), (x1, y1), (255, 255, 0), 2)

            cv2.putText(frame, "Prediction from EMG", (1000, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Draw GT landmarks
        for lm in gt_frame:
            cx, cy = int(lm[0] * frame.shape[1]), int(lm[1] * frame.shape[0])
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)  # Blue

        for c in HAND_CONNECTIONS:
            x0, y0 = int(gt_frame[c[0]][0] * frame.shape[1]), int(gt_frame[c[0]][1] * frame.shape[0])
            x1, y1 = int(gt_frame[c[1]][0] * frame.shape[1]), int(gt_frame[c[1]][1] * frame.shape[0])
            cv2.line(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)

        # Add text
        cv2.putText(frame, "Mediapipe (Ground Truth)", (1000, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (139, 0, 0), 2)

        # Show counter on frame
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show frame
        cv2.imshow("Landmark Overlay", frame)
        if cv2.waitKey(int(1000 / video_fps)) & 0xFF == 27:
            break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Visualization complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize EMG-predicted landmarks vs. ground truth")
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Root directory containing video, model, landmark, and EMG data')
    parser.add_argument('--label', type=str, default=None,
                        help="Unique label associated with the video and landmark data")
    args = parser.parse_args()
    run_visualization(args.root_dir, args.label)
