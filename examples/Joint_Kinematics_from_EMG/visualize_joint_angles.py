"""
visualize_joint_angles.py

Visualizes predicted vs ground-truth finger joint angles from EMG using a trained model.
Author: Jonathan Shulgach
Date: 06/14/25
"""

import os
import numpy as np
import pickle
import argparse
import logging
import torch
import matplotlib.pyplot as plt

from handtrack.ml import EMGRegressor


ANGLE_NAMES = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']


def setup_logger():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def visualize_joint_angles(root_dir, label):
    setup_logger()

    # Load dataset
    dataset_path = os.path.join(root_dir, f"{label}_feature_dataset.npz")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    data = np.load(dataset_path)
    X, y_true = data['emg_features'], data['landmark_labels']  # landmark_labels are now angles

    # Load scaler
    scaler_path = os.path.join(root_dir, 'model', f"{label}_scaler.pkl")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(X)

    # Load model
    model_path = os.path.join(root_dir, 'model', f"{label}_emg_regressor.pth")
    model = EMGRegressor(X.shape[1], y_true.shape[1])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Predict
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_pred = model(X_tensor).numpy()

    # Convert radians to degrees
    y_pred = np.degrees(y_pred)
    y_true = np.degrees(y_true)

    # Plot all 5 angles
    plt.figure(figsize=(12, 8))
    time_axis = np.arange(y_true.shape[0]) / 100.0  # Assume ~10ms step size
    for i in range(5):
        plt.subplot(5, 1, i + 1)
        plt.plot(time_axis, y_true[:, i], label='Ground Truth', color='blue')
        plt.plot(time_axis, y_pred[:, i], label='Prediction', color='orange')
        plt.ylabel(ANGLE_NAMES[i])
        plt.legend(loc='upper right')
        plt.grid(True)
    plt.xlabel('Time (s)')
    plt.suptitle('Predicted vs Ground Truth Joint Angles')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize EMG-predicted joint angles vs. ground truth")
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing dataset and model')
    parser.add_argument('--label', type=str, required=True, help='Label used in dataset and model filenames')
    args = parser.parse_args()
    visualize_joint_angles(args.root_dir, args.label)
