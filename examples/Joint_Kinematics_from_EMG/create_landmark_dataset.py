from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d
from src.handtrack.processing import notch_filter, bandpass_filter, lowpass_filter, rectify, extract_features, compute_finger_angles


def apply_filters(emg, fs=5000):
    emg = notch_filter(emg, fs, 60)
    emg = bandpass_filter(emg, 20, 450, fs)
    emg = rectify(emg)
    return lowpass_filter(emg, 5, fs)


EMG_FILE = '../../data/emg_data.npz'
LANDMARKS_FILE = '../../data/HandDynamic_smoothed_landmarks.npz'
SYNC_FILE = '../../data/sync_offset.txt'

# Load the EMG data
result = np.load(EMG_FILE)
print(f"EMG data loaded with contents: {list(result.keys())} of shape {result['emg_data'].shape}")

# Load the landmarks data
landmarks_data = np.load(LANDMARKS_FILE)
print(f"Landmark file contents: {list(landmarks_data.keys())} of data size {landmarks_data['landmarks'].shape}")

# Load the sync offset
with open(SYNC_FILE, 'r') as f:
    sync_offset = float(f.readlines()[0].split(': ')[1].strip().split(' ')[0])
print(f"Sync offset loaded: {sync_offset:.3f} seconds")

# Extract landmarks and metadata
lm = landmarks_data['landmarks']
lm_fs = int(landmarks_data['sampling_rate'])
lm_t = np.arange(lm.shape[0]) / lm_fs
lm_t += + sync_offset  # Adjust time vector with sync offset

# Extract EMG data and metadata
emg = result['emg_data']  # Assuming EMG data is stored under 'emg' key
emg_fs = int(result['sampling_rate'])  # Assuming EMG sampling rate is 5000 Hz
emg_t = result['time_vector']

# Check shapes
print(f"Time vector (video): {lm_t.shape}")
print(f"Time vector (EMG): {emg_t.shape}")

# Interpolate landmarks to EMG time vector
lm_interp = np.zeros((emg.shape[1], lm.shape[1], lm.shape[2]))
for joint_idx in range(lm.shape[1]):
    for dim in range(lm.shape[2]):
        f = interp1d(lm_t, lm[:, joint_idx, dim],
                     kind='linear', fill_value='extrapolate')
        lm_interp[:, joint_idx, dim] = f(emg_t)

print("Shape of interpolated landmarks:", lm_interp.shape)

# Preprocess the EMG data before extracting features
print(f"Shape of EMG data: {emg.shape}, processing {emg.shape[0]} channels")
emg_processed = np.zeros_like(emg)
emg_processed = apply_filters(emg, emg_fs)
print(f"Processed EMG shape: {emg_processed.shape}")

#for i in tqdm(range(emg.shape[0]), desc="Processing EMG channels"):
#    data = emg[i, :]  # Get the i-th channel
#    print(f"Shape of channel {i}: {data.shape}")
#    emg_processed[i, :] = apply_filters(data, emg_fs, axis=0)

# Apply rolling window feature extraction, implement a rolling window of 250ms with 50ms step

WINDOW_MS = 50  # Window size in milliseconds
STEP_MS = 10  # Step size in milliseconds

window_size = int(WINDOW_MS * emg_fs / 1000)
step_size = int(STEP_MS * emg_fs / 1000)
print(f"Extracting features with window size: {window_size} samples, step size: {step_size} samples")

n_windows = (emg.shape[1] - window_size) // step_size + 1
emg_features = []
landmark_labels = []

for start in tqdm(range(0, emg.shape[1] - window_size + 1, step_size), desc="Extracting features"):
    end = start + window_size
    emg_window = emg_processed[:, start:end]

    # Extract features from the EMG window across channels
    features = extract_features(emg_window)
    emg_features.append(features)

    # Corresponding landmarks for this window
    lm_window = lm_interp[start:end, :, :]
    avg_lm = np.mean(lm_window, axis=0)  # Average landmarks over the window, shape (21, 3)
    landmark_labels.append(avg_lm.flatten())  # Flatten landmarks for 63-dim vector

# Convert to numpy arrays
emg_features = np.array(emg_features)  # Shape (n_windows, n_features)
landmark_labels = np.array(landmark_labels)  # Shape (n_windows, 63) if 21 landmarks with 3 dimensions each

print("Feature shape:", emg_features.shape)
print("Label shape:", landmark_labels.shape)

# Save the dataset to a npz file
np.savez('../../data/hand_dynamic_dataset.npz',
         emg_features=emg_features,
         landmark_labels=landmark_labels,
         )


