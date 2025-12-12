"""
visualize_joint_angles_overlay.py

Overlays predicted vs ground-truth joint angles onto video frames using a trained EMG model.
Author: Jonathan Shulgach
Date: 06/15/25
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


ANGLE_NAMES = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']


def parse_events_file(file_path):
    start_idx, end_idx = None, None
    with open(file_path, 'r') as f:
        next(f)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue
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


def run_visualization(root_dir, label="HandDynamic"):
    setup_logger()
    logging.info(f"Using label: {label}")

    # Define paths
    video_path = os.path.join(root_dir, 'media', f"{label}.mp4")
    dataset_path = os.path.join(root_dir, f"{label}_feature_dataset.npz")
    model_path = os.path.join(root_dir, 'model', f"{label}_emg_regressor.pth")
    scaler_path = os.path.join(root_dir, 'model', f"{label}_scaler.pkl")
    sync_offset_file = os.path.join(root_dir, 'events', f"{label}_sync_offset.txt")
    events_file = os.path.join(root_dir, 'events', f"{label}_video_events.txt")

    # Sanity check
    for f in [video_path, dataset_path, model_path, scaler_path, sync_offset_file, events_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing required file: {f}")

    # Load start frame index
    start_frame_index, _ = parse_events_file(events_file)
    logging.info(f"Parsed start frame index from video events: {start_frame_index}")

    # Load sync offset
    with open(sync_offset_file, 'r') as f:
        sync_offset = float(f.readline().split(': ')[1].split(' ')[0])
    logging.info(f"Loaded sync offset: {sync_offset:.3f} seconds.")

    # Load EMG + angles
    data = np.load(dataset_path)
    emg_features = data['emg_features']
    ground_truth_angles = data['landmark_labels']  # Now joint angles
    emg_fs = int(data.get('emg_fs', 5000))
    window_ms = int(data.get('window_ms', 50))
    step_ms = int(data.get('step_ms', 10))
    window_size = int(window_ms * emg_fs / 1000)
    step_size = int(step_ms * emg_fs / 1000)

    logging.info(f"Loaded EMG features: {emg_features.shape} | Angle labels: {ground_truth_angles.shape}")

    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Load model
    model = EMGRegressor(emg_features.shape[1], ground_truth_angles.shape[1])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load video
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Failed to open video file: {video_path}"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    logging.info(f"Video has {frame_count} frames at {video_fps:.2f} fps.")

    # Align EMG to video using time offset
    emg_time = np.arange(emg_features.shape[0]) * (step_size / emg_fs)
    video_time = np.arange(frame_count) / video_fps - sync_offset
    interp_func = interp1d(emg_time, emg_features, axis=0, kind='linear', fill_value='extrapolate')
    emg_features_interpolated = interp_func(video_time)

    # Predict angles
    X_scaled = scaler.transform(emg_features_interpolated)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        predicted_angles = model(X_tensor).numpy()

    # Overlay loop
    frame_idx = 0
    logging.info("Beginning visualization... press ESC to quit.")
    while cap.isOpened() and frame_idx < min(frame_count, len(predicted_angles)):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx >= start_frame_index:
            pred = np.degrees(predicted_angles[frame_idx])
            gt = np.degrees(ground_truth_angles[frame_idx])

            for i, joint in enumerate(ANGLE_NAMES):
                text_gt = f"{joint} (GT): {gt[i]:.1f}°"
                text_pred = f"{joint}: {pred[i]:.1f}°"
                y_pos = 40 + i * 30
                cv2.putText(frame, text_gt, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(frame, text_pred, (300, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Add frame info
        cv2.putText(frame, f"Frame: {frame_idx}/{frame_count}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # Show frame
        cv2.imshow("Joint Angle Overlay", frame)
        if cv2.waitKey(int(1000 / video_fps)) & 0xFF == 27:
            break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Visualization complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay EMG-predicted joint angles on video")
    parser.add_argument('--root_dir', type=str, required=True, help='Directory with model/data')
    parser.add_argument('--label', type=str, default="HandDynamic", help='Dataset/video label prefix')
    args = parser.parse_args()
    run_visualization(args.root_dir, args.label)
