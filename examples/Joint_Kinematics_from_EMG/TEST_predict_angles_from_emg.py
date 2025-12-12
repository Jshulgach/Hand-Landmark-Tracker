import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

from handtrack.ml import ModelManager, EMGRegressor
from handtrack.io import load_yaml_config
from handtrack.processing import KalmanAngle


def predict_angles(cfg):
    root_dir = cfg['root_dir']
    label = cfg.get('label', '')
    verbose = cfg.get('verbose', False)
    video_path = cfg.get('video_path', None)
    frame_offset = cfg.get('frame_offset', 0)
    model_path = cfg.get('model_path', '')

    data_path = os.path.join(root_dir, f"{label}_training_dataset.npz" if label else "training_dataset.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at {data_path}")

    data = np.load(data_path)
    X, y = data['features'], data['labels']
    print(f"Data shape - X: {X.shape}, y: {y.shape}")

    # Use model manager
    manager = ModelManager(root_dir=root_dir,
                           #label=label,
                           label="16chahnel",
                           verbose=verbose
                           )
    print(f"Model path set to: {model_path}")
    manager.load_model(
        model=EMGRegressor(input_dim=X.shape[1], output_dim=y.shape[1]),
        weights=model_path
    )
    manager.scalar_path = os.path.join(root_dir, 'model', '16channel_scalar.pkl')
    print(f"Searching for scalars at '{manager.scalar_path}'")
    manager.load_scalar()

    if not manager.model_exists:
        raise ValueError("Model does not exist. Please train the model first.")

    # Predict angles
    y_pred = manager.predict(X)  # Expects (N, 5)
    print(f"Predicted angles shape: {y_pred.shape}")

    # Add the kalman filter and smooth the angles
    kalman_filters = [KalmanAngle(process_noise=1e-4, measurement_noise=1e-3) for _ in range(y_pred.shape[1])]
    for i in range(y_pred.shape[0]):
        for j in range(y_pred.shape[1]):
            y_pred[i, j] = kalman_filters[j].update(y_pred[i, j])

    # Load the reference data set
    data_path = os.path.join(root_dir, f"{label}_training_dataset.npz" if label else "training_dataset.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at {data_path}")

    if verbose:
        print(f"Loading dataset from {data_path}")
    data = np.load(data_path)
    X, y_true = data['features'], data['labels']
    print(f"Reference Data shape - X: {X.shape}, y: {y.shape}")


    # Just plot the predicted angles
    plt.figure(figsize=(12, 8))
    time_axis = np.arange(y_pred.shape[0]) / 100.0  # Assume ~10ms step size
    for i in range(5):
        plt.subplot(5, 1, i + 1)
        #plt.plot(time_axis, predicted_angles[:, i], label='Prediction', color='orange')
        plt.plot(time_axis, y_true[:, i], label='Ground Truth', color='blue')
        plt.plot(time_axis, y_pred[:, i], label='Prediction', color='orange')
        plt.ylabel(f"Angle {i+1} (degrees)")
        plt.legend(loc='upper right')
        plt.grid(True)
    plt.xlabel('Time (s)')
    plt.suptitle('Predicted Joint Angles from EMG')
    plt.tight_layout()
    plt.show()

    # If the video path was provided, let's visualize the prediction to see how the angles change with the video frames
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file at {video_path}")

        frame_index = 0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in video: {n_frames}")

        # Need to make sure the number of predicted angles matches the number of frames
        # Downsample the predicted angles to match teh video frame count. Need to keep the first and last frame but
        # interpolate the rest
        if predicted_angles.shape[0] < n_frames:
            # Interpolate to match number of frames
            indices = np.linspace(0, predicted_angles.shape[0] - 1, n_frames).astype(int)
            predicted_angles = predicted_angles[indices]
        elif predicted_angles.shape[0] > n_frames:
            # Downsample to match number of frames
            indices = np.linspace(0, predicted_angles.shape[0] - 1, n_frames).astype(int)
            predicted_angles = predicted_angles[indices]

        print(f"Adjusted predicted angles shape: {predicted_angles.shape}")

        # Resize to smaller window
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("Starting playback... press ESC to quit")
        while frame_index < n_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Draw predicted angles on the frame
            angle_idx = frame_index - frame_offset
            if angle_idx >= 0:
                for i, angle in enumerate(predicted_angles[frame_index]):
                    cv2.putText(frame, f"Angle {i+1}: {angle:.2f}", (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            frame_index += 1

            cv2.imshow('Predicted Angles', frame)

            if cv2.waitKey(30) & 0xFF == 27:  # ESC key to exit
                break

        # # Plot all 5 angles
        # plt.figure(figsize=(12, 8))
        # time_axis = np.arange(y_true.shape[0]) / 100.0  # Assume ~10ms step size
        # for i in range(5):
        #     plt.subplot(5, 1, i + 1)
        #     plt.plot(time_axis, y_true[:, i], label='Ground Truth', color='blue')
        #     plt.plot(time_axis, y_pred[:, i], label='Prediction', color='orange')
        #     plt.ylabel(ANGLE_NAMES[i])
        #     plt.legend(loc='upper right')
        #     plt.grid(True)
        # plt.xlabel('Time (s)')
        # plt.suptitle('Predicted vs Ground Truth Joint Angles')
        # plt.tight_layout()
        # plt.show()


    # Save predictions
    #output_file = os.path.join(root_dir, f"{label}_predicted_angles.npy")
    #np.save(output_file, predicted_angles)

    #print(f"Predicted angles saved to {output_file}")
    return predicted_angles



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare EMG-predicted angles with MP joint angles.")
    parser.add_argument('--config_file', type=str, default=None, help='Path to config file with dataset paths and variables')
    parser.add_argument('--root_dir',    type=str, default='',      help='Root directory')
    parser.add_argument('--label',       type=str, default='',      help='Label used for model and data')
    parser.add_argument('--video_path',  type=str, default='',      help='video path to visualize predictions')
    parser.add_argument('--model_path',  type=str, default='',      help='Path to the trained model weights')
    parser.add_argument('--frame_offset', type=int, default=0, help='Number of frames to skip with prediction at the start')
    parser.add_argument('--verbose',     action='store_true',       help='Verbose debugging output')
    args = parser.parse_args()

    #config = load_yaml_config(args.config_file) or {}
    #if config is None:
    #    config = {}
    config = {}

    # Allow command-line override of config values
    config['root_dir'] = args.root_dir or config.get('root_dir', '')
    config['label'] = args.label or config.get('label', '')
    config['video_path'] = args.video_path or config.get('video_path', False)
    config['model_path'] = args.model_path or config.get('model_path', '')
    config['verbose'] = args.verbose or config.get('verbose', False)
    config['frame_offset'] = args.frame_offset or config.get('frame_offset', 0)

    # Run prediction
    angles = predict_angles(config)
