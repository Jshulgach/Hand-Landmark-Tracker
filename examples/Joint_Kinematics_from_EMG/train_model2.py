"""
train_emg_regressor.py

Trains an EMG-to-landmark regression model using features extracted from EMG data.
Saves the trained model and the scaler for use in future inference scripts.

Author: Jonathan Shulgach
Date: 06/03/25
"""
import os
import json
import pickle
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from handtrack.ml import EMGRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# -------------------------------------------
# Configuration Defaults
# -------------------------------------------
NUM_EPOCHS = 3000
EARLY_STOP_PATIENCE = 5
LEARNING_RATE = 1e-3
VAL_INTERVAL = 20


def setup_logger():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def train_emg_regressor(root_dir, label):
    setup_logger()

    # Load data
    data_path = os.path.join(root_dir, f"{label}_feature_dataset.npz" if label else "feature_dataset.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at {data_path}")

    logging.info("Loading dataset...")
    data = np.load(data_path)
    X, y = data['emg_features'], data['landmark_labels']
    emg_fs = int(data['emg_fs']) if 'emg_fs' in data else None
    lm_fs = int(data['lm_fs']) if 'lm_fs' in data else None
    logging.info(f"Loaded EMG features: {X.shape}, Landmark labels: {y.shape}")
    if emg_fs and lm_fs:
        logging.info(f"Sampling Rates — EMG: {emg_fs} Hz, Landmarks: {lm_fs} Hz")
    assert y.shape[1] == 5, f"Expected 5 output dims for finger angles, got {y.shape[1]}"

    # Train/test data split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing set shape: {X_test.shape}, {y_test.shape}")

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Define the EMG Regressor Model
    model = EMGRegressor(X.shape[1], y.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the Model
    best_val_loss = np.inf
    epochs_no_improve = 0
    loss_curve = []
    model_save_path = os.path.join(root_dir, 'model')
    os.makedirs(model_save_path, exist_ok=True)
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        loss_curve.append(loss.item())

        if (epoch + 1) % VAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_tensor)
                val_loss = criterion(val_outputs, y_test_tensor).item()
            logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {loss.item():.8f} | Val Loss: {val_loss:.8f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(model_save_path, f"{label}_emg_regressor.pth" if label else "emg_regressor.pth"))
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOP_PATIENCE:
                    logging.info("Early stopping triggered.")
                    break

    # Save final model and scaler
    torch.save(model.state_dict(), os.path.join(model_save_path, f"{label}_emg_regressor.pth" if label else "emg_regressor.pth"))
    with open(os.path.join(model_save_path, f"{label}_scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)
    logging.info(f"Model saved to: {model_save_path}")

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy()
        y_true = y_test_tensor.numpy()

        test_loss = criterion(torch.tensor(y_pred), y_test_tensor).item()
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        logging.info("=== Test Performance ===")
        logging.info(f"MSE: {test_loss:.4f} | R²: {r2:.4f} | MAE: {mae:.4f}")

    # Baseline
    #mean_landmark = np.mean(y_train, axis=0)
    #baseline_mse = np.mean((y_test - mean_landmark) ** 2)
    #baseline_r2 = r2_score(y_test, np.tile(mean_landmark, (y_test.shape[0], 1)))
    mean_angle = np.mean(y_train, axis=0)
    baseline_mse = np.mean((y_test - mean_angle) ** 2)
    baseline_r2 = r2_score(y_test, np.tile(mean_angle, (y_test.shape[0], 1)))
    logging.info("=== Baseline Performance ===")
    logging.info(f"MSE: {baseline_mse:.4f} | R²: {baseline_r2:.4f}")

    # Save metadata
    metadata = {
        "input_dim": int(X.shape[1]),
        "output_dim": int(y.shape[1]),
        "target_type": "FingerJointAngles",  # <- Add this new key
        "num_epochs": NUM_EPOCHS,
        "emg_fs": emg_fs,
        "lm_fs": lm_fs,
        "val_loss": float(best_val_loss),
        "r2": float(r2),
        "mae": float(mae)
    }
    with open(os.path.join(model_save_path, f"{label}_training_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Plot loss curve
    plt.plot(loss_curve)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(model_save_path, f"{label}_loss_curve.png"))

    print("Training completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an EMG-to-landmark regression model.")
    parser.add_argument('--root_dir', type=str, required=True, help="Root directory containing the dataset.")
    parser.add_argument("--label", type=str, required=True, help="Label used for EMG and landmark files (e.g., Dynamic5kHz)")
    args = parser.parse_args()
    train_emg_regressor(args.root_dir, args.label)