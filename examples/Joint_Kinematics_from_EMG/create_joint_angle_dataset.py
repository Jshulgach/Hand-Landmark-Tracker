from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d
from handtrack.processing import notch_filter, bandpass_filter, lowpass_filter, rectify, extract_features, compute_finger_angles


def apply_filters(emg, fs=5000):
    emg = notch_filter(emg, fs, 60)
    emg = bandpass_filter(emg, 20, 450, fs)
    emg = rectify(emg)
    return lowpass_filter(emg, 5, fs)


# ---------- CONFIG ----------
EMG_FILE = '../../data/emg_data.npz'
LANDMARKS_FILE = '../../data/HandDynamic_smoothed_landmarks.npz'
SYNC_FILE = '../../data/sync_offset.txt'

WINDOW_MS = 50  # Window size in milliseconds
STEP_MS = 10    # Step size in milliseconds

# ---------- LOAD DATA ----------
result = np.load(EMG_FILE)
print(f"EMG data loaded with contents: {list(result.keys())}, shape: {result['emg_data'].shape}")

landmarks_data = np.load(LANDMARKS_FILE)
print(f"Landmark file contents: {list(landmarks_data.keys())}, shape: {landmarks_data['landmarks'].shape}")

with open(SYNC_FILE, 'r') as f:
    sync_offset = float(f.readlines()[0].split(': ')[1].strip().split(' ')[0])
print(f"Sync offset loaded: {sync_offset:.3f} seconds")

# ---------- EXTRACT LANDMARKS AND TIME VECTORS ----------
lm = landmarks_data['landmarks']
lm_fs = int(landmarks_data['sampling_rate'])
lm_t = np.arange(lm.shape[0]) / lm_fs
lm_t += sync_offset

emg = result['emg_data']
emg_fs = int(result['sampling_rate'])
emg_t = result['time_vector']

print(f"Time vector shapes — Video: {lm_t.shape}, EMG: {emg_t.shape}")

# ---------- INTERPOLATE LANDMARKS ----------
lm_interp = np.zeros((emg.shape[1], lm.shape[1], lm.shape[2]))
for joint_idx in range(lm.shape[1]):
    for dim in range(lm.shape[2]):
        f = interp1d(lm_t, lm[:, joint_idx, dim], kind='linear', fill_value='extrapolate')
        lm_interp[:, joint_idx, dim] = f(emg_t)

print("Interpolated landmarks shape:", lm_interp.shape)

# ---------- FILTER EMG ----------
print(f"Filtering EMG data: {emg.shape}")
emg_processed = apply_filters(emg, emg_fs)
print(f"Filtered EMG shape: {emg_processed.shape}")

# ---------- EXTRACT FEATURES ----------
window_size = int(WINDOW_MS * emg_fs / 1000)
step_size = int(STEP_MS * emg_fs / 1000)
print(f"Window size: {window_size} samples, Step size: {step_size} samples")

n_windows = (emg.shape[1] - window_size) // step_size + 1
emg_features = []
joint_angle_labels = []

print("Extracting features and computing joint angles...")
for start in tqdm(range(0, emg.shape[1] - window_size + 1, step_size)):
    end = start + window_size
    emg_window = emg_processed[:, start:end]
    features = extract_features(emg_window)
    emg_features.append(features)

    # Average landmarks in this window
    lm_window = lm_interp[start:end, :, :]
    avg_lm = np.mean(lm_window, axis=0)  # shape: (21, 3)

    # Convert avg_lm to joint angles
    angles = compute_finger_angles(avg_lm)
    joint_angle_labels.append(angles)

# ---------- CONVERT TO NP ARRAYS ----------
emg_features = np.array(emg_features)
joint_angle_labels = np.array(joint_angle_labels)

print("Feature shape:", emg_features.shape)
print("Joint angle label shape:", joint_angle_labels.shape)

# ---------- SAVE ----------
np.savez('../../data/hand_dynamic_joint_angle_dataset.npz',
         emg_features=emg_features,
         joint_angles=joint_angle_labels)

print("Saved dataset with EMG features and joint angles.")
