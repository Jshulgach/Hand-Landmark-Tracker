import os
import argparse
from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d
from handtrack.processing import notch_filter, bandpass_filter, lowpass_filter, rectify, extract_features

# Local function to calculate sync offset
from calculate_sync_offset import calculate_sync_offset

WINDOW_MS = 50  # Window size in milliseconds
STEP_MS = 10  # Step size in milliseconds
SYNC_LABEL = 'Start'  # Label used for sync offset calculation
PRE_EVENT_BUFFER_SEC = 1.0  # Include 0.5 seconds before the start event


def apply_filters(emg, fs=5000):
    emg = notch_filter(emg, fs, 60)
    emg = bandpass_filter(emg, 20, 450, fs)
    emg = rectify(emg)
    return lowpass_filter(emg, 5, fs)


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


def create_dataset(root_dir, label):
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Directory {root_dir} does not exist.")

    label_folder = label.rstrip('_')
    label = f"{label_folder}_"

    # Define file paths
    emg_file_path = os.path.join(root_dir, 'raw', label_folder, f'{label}emg_data.npz')
    landmarks_file_path = os.path.join(root_dir, 'landmarks', f'{label}smoothed_landmarks.npz')
    emg_events_file_path = os.path.join(root_dir, 'events', f'{label}emg_events.txt')
    video_events_file_path = os.path.join(root_dir, 'events', f'{label}video_events.txt')
    sync_offset_file = os.path.join(root_dir, 'events', f'{label}sync_offset.txt')

    # Load the EMG data
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

    # Load the landmarks data
    print("Loading landmarks data...")
    landmarks_data = np.load(landmarks_file_path)
    lm = landmarks_data['landmarks']
    lm_fs = int(landmarks_data['sampling_rate'])
    lm_t = np.arange(lm.shape[0]) / lm_fs
    print("=== Landmarks data loaded ===")
    print(f"|   shape: {lm.shape}, sampling rate: {lm_fs} Hz, time vector: {lm_t.shape}")
    print(f"|   Landmarks time range: {lm_t[0]:.3f} to {lm_t[-1]:.3f} seconds\n")

    # Calculate sync offset if not provided
    if not os.path.exists(sync_offset_file):
        print(f"Sync offset file not found. Calculating sync offset...")
        sync_offset = calculate_sync_offset(root_dir, SYNC_LABEL, emg_fs, lm_fs, save_file=True, label_name=label_folder)
    else:
        with open(sync_offset_file, 'r') as f:
            sync_offset = float(f.readlines()[0].split(': ')[1].strip().split(' ')[0])
    print(f"|  Sync offset loaded: {sync_offset:.3f} seconds\n")

    # Shift landmark timestamp vector by sync offset
    lm_t += sync_offset

    # Interpolate landmarks to EMG time vector
    print("Interpolating landmarks to EMG time vector...")
    lm_interp = np.zeros((emg.shape[1], lm.shape[1], lm.shape[2]))
    for joint_idx in range(lm.shape[1]):
        for dim in range(lm.shape[2]):
            f = interp1d(lm_t, lm[:, joint_idx, dim], kind='linear', fill_value='extrapolate')
            lm_interp[:, joint_idx, dim] = f(emg_t)
    print(f"|  Shape of Interpolated landmarks: {lm_interp.shape}\n")

    # Attempt to parse event files to determine start and end indices
    print("Searching for Start/End events...")
    start_idx, end_idx = None, None
    for source_type, source_path in [('EMG', emg_events_file_path), ('Video', video_events_file_path)]:
        if os.path.exists(source_path):
            s_idx, e_idx = parse_events_file(source_path)
            if s_idx is not None and e_idx is not None:
                if source_type == 'Video':
                    # Convert video indices to EMG indices
                    start_idx = int(s_idx * emg_fs / lm_fs)
                    end_idx = int(e_idx * emg_fs / lm_fs)
                    print("Converted video indices to EMG indices.")
                else:
                    start_idx = s_idx
                    end_idx = e_idx
                print(f"|  Received Start index: {start_idx}, End index: {end_idx} from {source_type} events file.")
                break
            elif s_idx is not None and e_idx is None:
                # Use the end of the file as the end index
                start_idx = s_idx
                end_idx = emg.shape[1]
                print(f"{source_type} file found but End index not specified. Using end of {source_type} data as End index.")
            else:
                print(f"{source_type} file found but could not parse start/end indices.")
        else:
            print(f"[WARNING] {source_type} events file not found. Using full range.")

    if start_idx is None or end_idx is None:
        raise ValueError("Could not determine start and end indices from event files.")

    # === Add buffer to Start ===
    buffer_samples = int(PRE_EVENT_BUFFER_SEC * emg_fs)
    start_idx_adj = max(0, start_idx - buffer_samples)
    print(f"|  Adjusted Start index (with {PRE_EVENT_BUFFER_SEC:.2f}s buffer): {start_idx_adj}\n")

    # Preprocess the EMG data before extracting features
    emg_processed = apply_filters(emg, emg_fs)
    print(f"|  Processed EMG shape: {emg_processed.shape} with  {emg.shape[0]} channels\n")

    # Apply rolling window feature extraction, implement a rolling window of 250ms with 50ms step
    print("Performing feature extraction...")
    window_size = int(WINDOW_MS * emg_fs / 1000)
    step_size = int(STEP_MS * emg_fs / 1000)
    print(f"|  window size: {window_size} samples, step size: {step_size} samples")

    emg_features = []
    landmark_labels = []
    for start in tqdm(range(start_idx_adj, emg.shape[1] - window_size + 1, step_size), desc="Extracting features"):

        # Extract features from the EMG window across channels
        end = start + window_size
        features = extract_features(emg_processed[:, start:end])
        emg_features.append(features)

        # Corresponding landmarks for this window
        avg_lm = np.mean(lm_interp[start:end, :, :], axis=0)  # Average landmarks over the window, shape (21, 3)
        landmark_labels.append(avg_lm.flatten())  # Flatten landmarks for 63-dim vector

    # Convert to numpy arrays
    emg_features = np.array(emg_features)  # Shape (n_windows, n_features)
    landmark_labels = np.array(landmark_labels)  # Shape (n_windows, 63) if 21 landmarks with 3 dimensions each

    # Save the dataset to a npz file
    save_path = os.path.join(root_dir, f'{label_folder}_feature_dataset.npz' if label else 'feature_dataset.npz')
    np.savez(save_path, emg_features=emg_features, landmark_labels=landmark_labels, emg_fs=emg_fs, lm_fs=lm_fs)
    print(f"Dataset saved to {save_path} with shapes {emg_features.shape}, {landmark_labels.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create EMG-to-landmark dataset")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing data")
    parser.add_argument("--label", type=str, required=True, help="Label used for EMG and landmark files (e.g., Dynamic5kHz)")
    args = parser.parse_args()
    create_dataset(args.root_dir, args.label)
