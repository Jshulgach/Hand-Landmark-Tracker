import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from handtrack.ml import ModelManager, EMGRegressor


def predict_from_emg_npz(emg_path, model_path, verbose=False):
    # Load the EMG feature data
    if not os.path.exists(emg_path):
        raise FileNotFoundError(f"EMG .npz file not found at: {emg_path}")

    data = np.load(emg_path)
    if 'features' not in data:
        raise KeyError("Expected key 'features' in EMG .npz file.")
    X = data['features']
    print(f"[INFO] Loaded EMG features from {emg_path} with shape {X.shape}")

    # Define and load model
    model = EMGRegressor(input_dim=X.shape[1], output_dim=5)  # Assuming 5 joint angles
    manager = ModelManager(root_dir=os.path.dirname(emg_path), verbose=verbose)
    manager.load_model(model=model, weights=model_path)

    # Predict
    predicted_angles = manager.predict(X)
    print(f"[INFO] Predicted angles shape: {predicted_angles.shape}")

    return predicted_angles


def plot_predicted_angles(predicted_angles):
    plt.figure(figsize=(12, 8))
    time_axis = np.arange(predicted_angles.shape[0]) / 100.0  # Assuming 100 Hz or 10ms step
    for i in range(predicted_angles.shape[1]):
        plt.subplot(predicted_angles.shape[1], 1, i + 1)
        plt.plot(time_axis, predicted_angles[:, i], label=f'Angle {i + 1}', color='orange')
        plt.ylabel(f"Angle {i + 1} (deg)")
        plt.grid(True)
        plt.legend()
    plt.xlabel("Time (s)")
    plt.suptitle("Predicted Joint Angles from EMG Features")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict joint angles from an EMG feature .npz file using a trained model.")
    parser.add_argument("--emg_path", type=str, required=True, help="Path to EMG feature .npz file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model .pth file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    predicted_angles = predict_from_emg_npz(args.emg_path, args.model_path, args.verbose)
    plot_predicted_angles(predicted_angles)
