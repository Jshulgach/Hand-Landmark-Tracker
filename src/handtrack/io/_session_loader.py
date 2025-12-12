import os
import logging
import numpy as np
from handtrack.io import extract_event_indices, get_sync_offset
from handtrack.tracker import HandTracker


def setup_logger():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


setup_logger()


class SessionLoader:
    def __init__(self, root_dir, label, verbose=False):
        self.root_dir = root_dir
        self.label = label
        self.label_folder = label.rstrip('_')
        self.verbose = verbose
        self.emg_shape = None
        self.emg_fs = None

    def load_emg(self):
        """
        Load EMG data from the specified session. The data is expected to be in a .npz file with keys 'emg_data' or
        'amplifier_data', 'sampling_rate' or 'sample_rate', and 't_amplifier' or 'time_vector'.

        Returns
        -------
            emg (np.ndarray): The EMG data array.
            fs (int):  The sampling frequency of the EMG data.
            t (np.ndarray): The time vector corresponding to the EMG data.

        """
        if self.verbose:
            print("\n================== EMG =================")
            print(f"| Searching for EMG data...")
        if not os.path.exists(os.path.join(self.root_dir, 'raw', self.label_folder)):
            raise FileNotFoundError(f"Raw data folder for label '{self.label}' not found in {self.root_dir}/raw/{self.label_folder}")
        if not os.path.exists(os.path.join(self.root_dir, 'raw', self.label_folder, f'{self.label}_emg_data.npz')):
            raise FileNotFoundError(f"EMG data file '{self.label}_emg_data.npz' not found in {self.root_dir}/raw/{self.label_folder}")

        # Load EMG data from the .npz file
        path = os.path.join(self.root_dir, 'raw', self.label_folder, f'{self.label}_emg_data.npz')
        if self.verbose:
            print(f"| Loading EMG data from {path}")
        data = np.load(path, allow_pickle=True)
        emg = next((data[k] for k in ['emg_data', 'amplifier_data'] if k in data), None)
        fs = int(next((data[k] for k in ['sampling_rate', 'sample_rate'] if k in data), None))
        t = next((data[k] for k in ['t_amplifier', 'time_vector'] if k in data), None)
        if emg is None or fs is None or t is None:
            raise KeyError("Missing EMG data, fs, or time vector in .npz")

        self.emg_shape = emg.shape
        self.emg_fs = fs
        if self.verbose:
            print(f"| EMG data loaded")
            print(f"|   shape: {emg.shape}, sampling rate: {fs} Hz, time vector: {t.shape}")
            print(f"|   EMG time range: {t[0]:.3f} to {t[-1]:.3f} seconds\n")

        return emg, fs, t

    def load_landmarks(self):
        """
        Load smoothed landmarks from the specified session. The data is expected to be in a .npz file with keys 'landmarks',
        'sampling_rate', and 'time_vector'.
        Returns
        -------
            landmarks (np.ndarray): The smoothed landmarks array.
            fs (int): The sampling frequency of the landmarks.
            t (np.ndarray): The time vector corresponding to the landmarks.

        """
        landmarks_file = f"{self.label}_landmarks.npz" if self.label else "landmarks.npz"
        landmarks_path = os.path.join(self.root_dir, 'landmarks')
        if self.verbose:
            print(f"\n================== Landmarks =================")
            print(f"|  Searching for landmarks in {self.root_dir}")

        if not os.path.exists(landmarks_path):
            if self.verbose:
                print(f"|  Landmarks folder not found in {landmarks_path}, searching root directory...")

            if not os.path.exists(os.path.join(self.root_dir, landmarks_file)):
                if self.verbose:
                    print(f"|  Landmarks file '{landmarks_file}' not found in root directory or landmarks folder.")

                return None, None, None

        if self.verbose:
            print(f"|  Loading smoothed landmarks from {landmarks_path}/{landmarks_file}")

        # Load smoothed landmarks from the .npz file
        data = np.load(os.path.join(landmarks_path, landmarks_file), allow_pickle=True)
        landmarks = data['landmarks']
        fs = int(data.get('sampling_rate', 30))  # Default to 30 Hz if not specified
        t = data.get('time_vector', np.arange(landmarks.shape[0]) / fs)
        if self.verbose:
            print(f"| Loaded landmarks: {landmarks.shape} landmarks at {fs} Hz.")
        return landmarks, fs, t

    def get_landmarks_from_video(self, video_path=None, apply_kalman=True, show_video=False, save_video=False, verbose=False):
        if self.verbose:
            print(f"\n================== Video Landmark =================")
            print(f"|  Extracting landmarks from video...")

        if video_path is None:
            video_path = os.path.join(self.root_dir, 'media', f"{self.label}.mp4") if self.label else os.path.join(self.root_dir, 'media', 'video.mp4')

        if not os.path.exists(video_path):
            if self.verbose:
                print(f"|  Video file '{video_path}' not found.")

        tracker = HandTracker(source=video_path, apply_kalman=apply_kalman, verbose=verbose)
        landmarks, metadata = tracker.extract_landmarks(visualize=show_video, save_video=save_video)
        return landmarks, metadata['sampling_rate'], metadata['time_vector']

    def load_event_indices(self):
        if self.verbose:
            print(f"\n================== Event Indices =================")
            print(f"|  Searching for event indices in {self.root_dir}/events...")

        # Extract start and end indices from the event files
        indices = extract_event_indices(os.path.join(
            self.root_dir, 'events'),
            label=self.label,
            verbose=self.verbose
        )
        if not indices['start_idx'] or not indices['end_idx']:
            print(f"|  No valid start/end indices found for session: {self.label}")
            return None, None

        # For now just returning the first result
        start_idx = indices['start_idx'][0]
        end_idx = indices['end_idx'][0]

        # Adjust slicing
        if start_idx is None or end_idx is None:
            start_idx, end_idx = 0, self.emg_shape[1]
        else:
            buffer = int(0.5 * self.emg_fs)
            start_idx = max(0, start_idx - buffer)

        return start_idx, end_idx

    def load_sync_offset(self):
        print("\n================== Sync Offset =================")
        print(f"|  Searching for sync offset file in {self.root_dir}/events...")
        events_dir = os.path.join(self.root_dir, 'events')
        if not os.path.isdir(events_dir):
            if self.verbose:
                print("|  Directory not found.")
            return

        filename = f"{self.label}_sync_offset.txt" if self.label else "sync_offset.txt"
        filepath = os.path.join(events_dir, filename)
        if os.path.exists(filepath):
            print(f"|  Sync file found, loading: '{filepath}'")
            with open(filepath, 'r') as f:
                line = f.readline()
                sync_offset = float(line.split(': ')[1].split(' ')[0])
            print(f"|  Sync offset loaded: {sync_offset:.3f} seconds")
            return sync_offset

        print(f"|  Sync offset file not found for label '{self.label}'. Attempting to compute...")
        try:
            sync_offset = get_sync_offset(
                root_dir=self.root_dir,
                label=self.label,
                save_file=True,
                emg_fs=self.emg_fs or 5000  # fallback if not yet loaded
            )
            return sync_offset
        except Exception as e:
            print(f"[ERROR] Failed to compute sync offset: {e}")
            raise e
