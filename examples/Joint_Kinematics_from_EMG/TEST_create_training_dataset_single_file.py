import os
import argparse
import numpy as np
import yaml

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


def get_feature_dataset(root_dir, label=None, show_video=False, overwrite=False, verbose=False):
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

    # Check for landmark file
    landmarks, lm_fs, lm_t = loader.load_landmarks()
    if landmarks is None:
        raise FileNotFoundError("Landmark data not found and no fallback video processing available without config.")
        # TO-DO: Option to get landmarks from video?
        #video_path = config_file.get('VIDEO_PATH', None)
        #landmarks, lm_fs, lm_t = loader.get_landmarks_from_video(video_path, verbose=verbose, visualize=show_video)

    # Preprocess EMG data
    preprocessor = EMGPreprocessor(emg_fs, verbose=verbose)
    emg_filtered = preprocessor.preprocess(emg)

    # Get start/end indices
    start_idx, end_idx = loader.load_event_indices()
    if start_idx is None or end_idx is None:
        raise ValueError("Could not determine start and end indices from event files.")

    # Time alignment
    sync_offset = loader.load_sync_offset()
    lm_t += sync_offset
    emg_t = emg_t[:emg_filtered.shape[1]]  # Ensure same length if clipped

    # Interpolate and get angles from landmarks
    landmarks_interp = interpolate_array_to_timebase(lm_t, emg_t, landmarks)
    if verbose:
        print(f"[INFO] Interpolated landmarks to EMG time vector. New shape: {landmarks_interp.shape}")
    y = get_angles_from_landmarks(landmarks_interp, emg_fs, start_idx, end_idx)
    print(f"[INFO] Angle data shape: {y.shape} from landmarks.")

    # Extract data features
    X = preprocessor.extract_emg_features(emg_filtered, start_idx, end_idx)

    # Save dataset
    print("")
    if os.path.exists(save_path) and not overwrite:
        print(f"[INFO] Feature dataset already exists at {save_path}. Use --overwrite to regenerate.")
        return
    np.savez(save_path, features=X, labels=y, emg_fs=emg_fs, lm_fs=lm_fs)
    print(f"[INFO] Saved dataset to: {save_path}")
    return X, y, emg_fs, lm_fs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare EMG-predicted angles with MP joint angles.")
    parser.add_argument('--root_dir',   type=str, default='',      help='Root directory')
    parser.add_argument('--label',      type=str, default='',      help='Label used for model and data')
    parser.add_argument('--overwrite',  action='store_true',       help='Save/overwrite features to dataset')
    parser.add_argument('--save_path',  type=str,                  help='Optional path to save dataset')
    parser.add_argument('--verbose',    action='store_true',       help='Verbose debugging output')
    args = parser.parse_args()

    X, y, emg_fs, lm_fs = get_feature_dataset(
        root_dir=args.root_dir,
        label=args.label,
        show_video=True,
        overwrite=args.overwrite,
        verbose=args.verbose
    )
