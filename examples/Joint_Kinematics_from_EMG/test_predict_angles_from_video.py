"""
This script demonstrates how to use the HandTracker and EMGRegressor classes to predict joint angles with jusyt EMG data.

The full pipeline follows:

- Lets you select a video file.
- Uses your MediaPipe-based HandTracker to extract landmarks from the video.
- Computes joint angles for each frame using your compute_finger_angles() function.
- Loads EMG .npz data.
- Computes features and downsamples/interpolates them to match the number of video frames.
- Predicts joint angles from the model.
- Plots predicted vs. ground-truth joint angles.


"""

import os
import numpy as np
import pickle
import torch
import logging
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from handtrack.io import check_video_path, read_config_file, extract_event_indices
from handtrack.tracker import HandTracker
from handtrack.ml import EMGRegressor
from handtrack.processing import notch_filter, bandpass_filter, rectify, lowpass_filter, extract_features

from calculate_sync_offset import get_sync_offset


ANGLE_NAMES = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

def setup_logger():
    import logging
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def apply_filters(emg, fs=5000, band=(20, 450)):
    emg = notch_filter(emg, fs, 60)
    emg = bandpass_filter(emg, band[0], band[1], fs)
    emg = rectify(emg)
    return lowpass_filter(emg, 5, fs)


def compute_finger_angles(landmarks):
    finger_indices = [(0, 5, 8),   # Index
                      (0, 9, 12),  # Middle
                      (0, 13, 16), # Ring
                      (0, 17, 20), # Pinky
                      (0, 1, 4)]   # Thumb

    angles = []
    for wrist, mcp, tip in finger_indices:
        a = landmarks[mcp] - landmarks[wrist]
        b = landmarks[tip] - landmarks[mcp]
        cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        angles.append(angle)
    return np.array(angles)


def main(root_dir, label, visualize=False, config_file=None):

    if root_dir is None:
        raise ValueError("Root directory must be specified.")

    # Define file paths
    emg_file_path = os.path.join(root_dir, 'raw', f'{label}', f'{label}_emg_data.npz')
    video_path = os.path.join(root_dir, 'media', f'{label}.mp4')
    sync_offset_file = os.path.join(root_dir, 'events', f'{label}_sync_offset.txt')

    # ==============================================================================================================
    # Step 1: Load the raw EMG data from the .npz file
    # ==============================================================================================================
    print("\nLoading EMG data...")
    result = np.load(emg_file_path, allow_pickle=True)
    emg = next((result[k] for k in ['emg_data', 'amplifier_data'] if k in result), None)
    emg_fs = next((int(result[k]) for k in ['sampling_rate', 'sample_rate'] if k in result), None)
    emg_t = next((result[k] for k in ['t_amplifier', 'time_vector'] if k in result), None)

    if emg is None or emg_fs is None or emg_t is None:
        raise KeyError("Could not find EMG data, sampling rate, or time vector.")

    print(f"=== EMG data loaded===")
    print(f"|   shape: {emg.shape}, sampling rate: {emg_fs} Hz, time vector: {emg_t.shape}")
    print(f"|   EMG time range: {emg_t[0]:.3f} to {emg_t[-1]:.3f} seconds\n")

    # ==============================================================================================================
    # Step 2: Get the landmarks from the video
    # ==============================================================================================================
    print("\nExtracting landmarks from video...")
    tracker = HandTracker(video_path)
    lm, video_data = tracker.extract_landmarks(visualize=visualize, save_video=args.save_video)
    lm_t = video_data['time_vector']
    lm_fs = video_data['sampling_rate']
    print("|  Landmarks extracted from video.")
    print(f"|  Number of frames: {lm.shape[0]}, Joints: {lm.shape[1]}, Dimensions: {lm.shape[2]}")

    # ==============================================================================================================
    # Step 3: Interpolate the landmark data to the EMG time vector
    # ==============================================================================================================
    print("Interpolating landmarks to EMG time vector...")
    lm_interp = np.zeros((emg.shape[1], lm.shape[1], lm.shape[2]))
    for joint_idx in range(lm.shape[1]):
        for dim in range(lm.shape[2]):
            interp_func = interp1d(lm_t, lm[:, joint_idx, dim], kind='linear', bounds_error=False, fill_value='extrapolate')
            lm_interp[joint_idx, dim] = interp_func(emg_t)
    print(f"|  Shape of Interpolated landmarks: {lm_interp.shape}\n")

    # ==============================================================================================================
    # Step 4: Get the timing offset between the EMG and landmark data (Not finished)
    # ==============================================================================================================
    if config_file and 'SYNC_OFFSET' in config_file:
        sync_offset = config_file['SYNC_OFFSET']
        print(f"| Using sync offset from config: {sync_offset:.3f} seconds")
    if not os.path.exists(sync_offset_file):
        print(f"| Sync offset file not found. Calculating sync offset...")
        sync_offset = get_sync_offset(root_dir, label, 'Start', emg_fs, lm_fs, save_file=True)
    else:
        with open(sync_offset_file, 'r') as f:
            sync_offset = float(f.readlines()[0].split(': ')[1].strip().split(' ')[0])
    print(f"|  Sync offset loaded: {sync_offset:.3f} seconds\n")

    # Shift landmark timestamp vector by sync offset
    lm_t += sync_offset

    # ==============================================================================================================
    # Step 5: Get the start and end indices for the data to use
    # ==============================================================================================================
    print("Searching for Start/End events...")
    event_indices = extract_event_indices(root_dir)  # Will look in the events folder for files ending in '.events'
    # Choose the first event file that contains 'Start' and 'End' labels, but if there are multiple, use the one with the greated sample number, usually aa source type of emg
    start_idx, end_idx = event_indices['start_idx'][0], event_indices['end_idx'][0] # TO-DO, need to finih

    # ==============================================================================================================
    # Step 6: Compute EMG features
    # ==============================================================================================================
    # Preprocess the EMG data before extracting features
    emg_processed = apply_filters(emg, emg_fs)
    print(f"|  Processed EMG shape: {emg_processed.shape} with  {emg.shape[0]} channels\n")

    # Apply rolling window feature extraction, implement a rolling window of 250ms with 50ms step
    print("Performing feature extraction...")
    WINDOW_MS = config_file.get('WINDOW_MS', 50)  # Default to 250ms
    STEP_MS = config_file.get('STEP_MS', 10)      # Default to 50ms
    window_size = int(WINDOW_MS * emg_fs / 1000)
    step_size = int(STEP_MS * emg_fs / 1000)
    print(f"|  window size: {window_size} samples, step size: {step_size} samples")

    emg_features = []
    landmark_labels = []
    for start in tqdm(range(start_idx, emg.shape[1] - window_size + 1, step_size), desc="Extracting features"):

        # Extract features from the EMG window across channels
        end = start + window_size
        features = extract_features(emg_processed[:, start:end])
        emg_features.append(features)

        # Corresponding landmarks for this window
        avg_lm = np.mean(lm_interp[start:end, :, :], axis=0)  # (21, 3)
        angles = compute_finger_angles(avg_lm)  # (5,)
        landmark_labels.append(angles)

    # Convert to numpy arrays
    emg_features = np.array(emg_features)  # Shape (n_windows, n_features)
    landmark_labels = np.array(landmark_labels)  # Shape (n_windows, 5) for angles

    # ==============================================================================================================
    # Step 7: Build regression model
    # ==============================================================================================================

    # If the pretrained model doesn't exist, then create it

    # # Load model
    # model_path = os.path.join(root_dir, 'model', f"{label}_emg_regressor.pth")
    # model = EMGRegressor(X.shape[1], y_true.shape[1])
    # model.load_state_dict(torch.load(model_path))
    # model.eval()
    #
    # # Predict
    # with torch.no_grad():
    #     X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    #     y_pred = model(X_tensor).numpy()
    #

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
            logging.info(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {loss.item():.8f} | Val Loss: {val_loss:.8f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(model_save_path,
                                                            f"{label}_emg_regressor.pth" if label else "emg_regressor.pth"))
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOP_PATIENCE:
                    logging.info("Early stopping triggered.")
                    break

    # Save final model and scaler
    torch.save(model.state_dict(),
               os.path.join(model_save_path, f"{label}_emg_regressor.pth" if label else "emg_regressor.pth"))
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
    # mean_landmark = np.mean(y_train, axis=0)
    # baseline_mse = np.mean((y_test - mean_landmark) ** 2)
    # baseline_r2 = r2_score(y_test, np.tile(mean_landmark, (y_test.shape[0], 1)))
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
    parser = argparse.ArgumentParser(description="Compare EMG-predicted angles with MP joint angles.")
    parser.add_argument('--root_dir',   type=str, default='',      help='Root directory')
    parser.add_argument('--video_file', type=str, default='',      help='Path to video file to extract landmarks')
    parser.add_argument('--label',      type=str, default='',      help='Label used for model and data')
    parser.add_argument('--sync_label', type=str, default='Start', help='Label for sync offset calculation')
    parser.add_argument('--visualize',  action='store_true',       help='Visualize the landmark extraction process')
    args = parser.parse_args()

    root_dir = args.root_dir.strip()
    video_file = args.video.strip()
    label = args.label.strip()

    # Check for a config file in case the root directory was already created
    config_file = read_config_file()

    if config_file is not None:
        root_dir = config_file.get('ROOT_DIR', args.root_dir)
        video_file = config_file.get('VIDEO_FILE', args.video)
        label = config_file.get('LABEL', args.label)

    main(args.root_dir, args.video, args.label, config_file)



