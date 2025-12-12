import os
import argparse
import numpy as np
import yaml
from scipy.signal import resample

from handtrack.io import load_yaml_config, SessionLoader, interpolate_array_to_timebase
from handtrack.processing import EMGPreprocessor, compute_3point_finger_angles


def get_angles_from_landmarks(landmarks_interp, emg_fs, start_idx, end_idx, window_ms=250, step_ms=50):
    print("\n=================== Extracting Angles from Landmarks =================")
    win_size = int(window_ms * emg_fs / 1000)
    step_size = int(step_ms * emg_fs / 1000)

    labels = []
    for i in range(start_idx, end_idx - win_size + 1, step_size):
        avg_landmarks = np.mean(landmarks_interp[i:i + win_size], axis=0)
        angles_dict = compute_3point_finger_angles(avg_landmarks) # Outputs a fictionary, we need to convert to array
        angles_list = [angles_dict['thumb'], angles_dict['index'], angles_dict['middle'], angles_dict['ring'], angles_dict['pinky']]
        labels.append(angles_list)

    return np.array(labels)


def get_feature_dataset(root_dir, label=None, target_emg_fs=None, window_ms=250, step_ms=50, selected_channels=None, show_video=False,  overwrite=False, verbose=False):
    """
    Load or create the feature dataset for EMG and joint angles.
    """
    save_path = os.path.join(root_dir, f"{label}_training_dataset.npz" if label else "training_dataset.npz")
    if os.path.exists(save_path) and not overwrite:
        print(f"[INFO] Feature dataset already exists at {save_path}. Use --overwrite to regenerate.")
        data = np.load(save_path, allow_pickle=True)
        return data['features'], data['labels'], data['emg_fs'], data['lm_fs']

    # Load session data
    loader = SessionLoader(root_dir, label, verbose)

    # Get EMG data
    emg, emg_fs, emg_t = loader.load_emg()
    if selected_channels is not None:
        print(f"[INFO] Using only selected EMG channels: {selected_channels}")
        emg = emg[selected_channels, :]

    original_emg_fs = emg_fs

    # If the target_emg_fs is specified, resample EMG data
    if target_emg_fs is not None and emg_fs != target_emg_fs:
        print(f"[INFO] Resampling EMG from {emg_fs} Hz to {target_emg_fs} Hz")
        num_samples = int(len(emg_t) * target_emg_fs / emg_fs)
        emg = resample(emg, num_samples, axis=1)
        emg_fs = target_emg_fs
        emg_t = np.linspace(emg_t[0], emg_t[-1], num_samples)
    else:
        print(f"[INFO] Using original EMG sampling frequency: {emg_fs} Hz")

    # Get start/end indices
    start_idx, end_idx = loader.load_event_indices()
    if start_idx is None or end_idx is None:
        raise ValueError("Could not determine start and end indices from event files.")

    # Rescale event indices if resampled
    if target_emg_fs is not None and original_emg_fs != target_emg_fs:
        scale = target_emg_fs / original_emg_fs
        start_idx = int(start_idx * scale)
        end_idx = int(end_idx * scale)

    # Check for landmark file
    landmarks, lm_fs, lm_t = loader.load_landmarks()
    if landmarks is None:
        raise FileNotFoundError("Landmark data not found and no fallback video processing available without config.")
        # TO-DO: Option to get landmarks from video?
        #video_path = config_file.get('VIDEO_PATH', None)
        #landmarks, lm_fs, lm_t = loader.get_landmarks_from_video(video_path, verbose=verbose, visualize=show_video)


    # Time alignment
    sync_offset = loader.load_sync_offset()
    lm_t += sync_offset
    emg_t = emg_t[:emg.shape[1]]  # Ensure same length if clipped

    # Interpolate and get angles from landmarks
    landmarks_interp = interpolate_array_to_timebase(lm_t, emg_t, landmarks)
    if verbose:
        print(f"[INFO] Interpolated landmarks to EMG time vector. New shape: {landmarks_interp.shape}")
    y = get_angles_from_landmarks(landmarks_interp, emg_fs, start_idx, end_idx, window_ms=window_ms, step_ms=step_ms)
    print(f"[INFO] Angle data shape: {y.shape} from landmarks.")

    # Preprocess EMG data
    preprocessor = EMGPreprocessor(emg_fs, verbose=verbose)
    emg_filtered = preprocessor.preprocess(emg)

    # Extract data features
    X = preprocessor.extract_emg_features(emg_filtered, start_idx, end_idx, window_ms=window_ms, step_ms=step_ms)

    # Save dataset
    print("")
    if os.path.exists(save_path) and not overwrite:
        print(f"[INFO] Feature dataset already exists at {save_path}. Use --overwrite to regenerate.")
        return
    np.savez(save_path, features=X, labels=y, emg_fs=emg_fs, lm_fs=lm_fs)
    print(f"[INFO] Saved dataset to: {save_path}")
    return X, y, emg_fs, lm_fs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge EMG + angle datasets with optional downsampling.")
    parser.add_argument('--root_dir', required=True)
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--channels', nargs='+', type=int, default=None)
    parser.add_argument('--target_emg_fs', type=int, default=1000)
    parser.add_argument('--window_ms', type=int, default=100, help='Window size in milliseconds for angle extraction')
    parser.add_argument('--step_ms', type=int, default=20, help='Step size in milliseconds for angle extraction')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output_file', type=str, default='combined_training_dataset.npz')
    args = parser.parse_args()

    all_X, all_y = [], []

    for label in args.labels:
        print(f"\nProcessing label: {label}")
        X, y, emg_fs, _ = get_feature_dataset(
            root_dir=args.root_dir,
            label=label,
            target_emg_fs=args.target_emg_fs,
            window_ms=args.window_ms,
            step_ms=args.step_ms,
            selected_channels=args.channels,
            show_video=False,
            overwrite=args.overwrite,
            verbose=args.verbose
        )

        if emg_fs != args.target_emg_fs:
            print(f"[INFO] Resampling EMG from {emg_fs} Hz to {args.target_emg_fs} Hz for label: {label}")
            # You must implement time and landmark resampling logic inside get_feature_dataset if needed
            raise NotImplementedError("Resampling not yet integrated into get_feature_dataset. See resample_emg_and_landmarks().")

        all_X.append(X)
        all_y.append(y)

    X_merged = np.vstack(all_X)
    y_merged = np.vstack(all_y)

    print(f" [INFO] Merged dataset shapes - X: {X_merged.shape}, y: {y_merged.shape}")

    save_path = os.path.join(args.root_dir, args.output_file)
    np.savez(save_path, features=X_merged, labels=y_merged, emg_fs=args.target_emg_fs)
    print(f"\n✅ Merged dataset saved to: {save_path}")
