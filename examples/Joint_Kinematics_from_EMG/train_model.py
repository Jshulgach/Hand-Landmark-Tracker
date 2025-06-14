"""
train_emg_regressor.py

Trains an EMG-to-landmark regression model using features extracted from EMG data.
Saves the trained model and the scaler for use in future inference scripts.

Author: Jonathan Shulgach
Date: 06/03/25
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.handtrack.ml import EMGRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 1️⃣ Load the dataset
    print("Loading dataset...")
    data = np.load('../../data/hand_dynamic_joint_angle_dataset.npz')
    # data = np.load('output_data/hand_dynamic_dataset.npz')
    X = data['emg_features']       # shape: (n_samples, n_features)
    # y = data['landmark_labels']    # shape: (n_samples, 63)
    y = data['joint_angles']


    print(f"Dataset loaded: EMG features shape {X.shape}, landmark labels shape {y.shape}")

    # 2️⃣ Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing set shape: {X_test.shape}, {y_test.shape}")

    # 3️⃣ Normalize EMG data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4️⃣ Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


    # 5️⃣ Define the EMG Regressor Model
    model = EMGRegressor(X.shape[1], y.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 6️⃣ Train the Model
    num_epochs = 1000
    early_stop_patience = 10
    best_val_loss = np.inf
    epochs_no_improve = 0

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_tensor)
                val_loss = criterion(val_outputs, y_test_tensor).item()
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Train Loss: {loss.item():.6f}, '
                  f'Val Loss: {val_loss:.6f}')

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), 'output_data/best_emg_regressor.pth')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    print("Early stopping triggered.")
                    break

    # Save final model
    torch.save(model.state_dict(), 'output_data/joint_angle_emg_regressor.pth')
    np.save('../../data/joint_angle_feature_scaler.npy', scaler.mean_)
    np.save('../../data/joint_angle_feature_scaler_std.npy', scaler.scale_)

    print("Model and scaler saved to 'output_data/'")
    # 8️⃣ Evaluate the Model
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy()
        y_true = y_test_tensor.numpy()

        test_loss = criterion(torch.tensor(y_pred), y_test_tensor).item()
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        print("\n==========================")
        print(f"Test MSE Loss: {test_loss:.6f}")
        print(f"Test R² Score: {r2:.6f}")
        print(f"Test MAE: {mae:.6f}")
        print("==========================\n")

    # 9️⃣ Compute Baseline Performance
    mean_landmark = np.mean(y_train, axis=0)
    baseline_mse = np.mean((y_test - mean_landmark) ** 2)
    baseline_r2 = r2_score(y_test, np.tile(mean_landmark, (y_test.shape[0], 1)))
    print("Baseline Performance (predicting mean landmark):")
    print(f"Baseline MSE: {baseline_mse:.6f}")
    print(f"Baseline R²: {baseline_r2:.6f}")

    # 🔟 Plot sample predictions
    plt.figure(figsize=(12, 6))
    sample_idx = 0  # Pick the first test sample
    plt.plot(y_test[sample_idx], label='Ground Truth')
    plt.plot(y_pred[sample_idx], label='Prediction')
    plt.xlabel('Landmark Index')
    plt.ylabel('Normalized Position')
    plt.legend()
    plt.title('Sample Prediction vs. Ground Truth')
    plt.grid(True)
    plt.show()

    print("Training completed successfully!")
