"""
Create training dataset from Open Ephys EMG data and video landmarks.

This script adapts the original create_feature_dataset.py to work with Open Ephys data
sources instead of Intan RHD files. It uses the OEphysSessionLoader to load EMG data
and creates a training dataset compatible with the EMGRegressor model.

Usage:
    python oephys_create_dataset.py --root_dir /path/to/data --label MySession --oebin_path /path/to/recording.oebin

Author: NML (Neuro-Mechatronics Lab)
Created: 2026-02-16
"""

import argparse
import os

import numpy as np

# Import handtrack modules
from handtrack.io import interpolate_array_to_timebase
from handtrack.processing import EMGPreprocessor, compute_3point_finger_angles

# Import Open Ephys session loader
from oephys_session_loader import OEphysSessionLoader
from scipy.signal import resample


def get_angles_from_landmarks(
    landmarks_interp: np.ndarray,
    emg_fs: float,
    start_idx: int,
    end_idx: int,
    window_ms: int = 250,
    step_ms: int = 50,
) -> np.ndarray:
    """
    Extract joint angles from interpolated landmarks using windowed averaging.

    Parameters
    ----------
    landmarks_interp : np.ndarray
        Interpolated landmarks aligned to EMG timebase, shape (n_samples, 63)
    emg_fs : float
        EMG sampling frequency in Hz
    start_idx : int
        Start index for windowing
    end_idx : int
        End index for windowing
    window_ms : int
        Window size in milliseconds
    step_ms : int
        Step size in milliseconds

    Returns
    -------
    angles : np.ndarray
        Joint angles array, shape (n_windows, 5)
        Columns: [thumb, index, middle, ring, pinky]
    """
    print("\n=================== Extracting Angles from Landmarks =================")
    win_size = int(window_ms * emg_fs / 1000)
    step_size = int(step_ms * emg_fs / 1000)

    labels = []
    for i in range(start_idx, end_idx - win_size + 1, step_size):
        # Average landmarks over window
        avg_landmarks = np.mean(landmarks_interp[i : i + win_size], axis=0)

        # Compute angles (returns dict)
        angles_dict = compute_3point_finger_angles(avg_landmarks)

        # Convert to array [thumb, index, middle, ring, pinky]
        angles_list = [
            angles_dict["thumb"],
            angles_dict["index"],
            angles_dict["middle"],
            angles_dict["ring"],
            angles_dict["pinky"],
        ]
        labels.append(angles_list)

    angles = np.array(labels)
    print(f"[INFO] Extracted {angles.shape[0]} angle windows, shape: {angles.shape}")

    return angles


def create_dataset(
    root_dir: str,
    label: str,
    oebin_path: str = None,
    target_emg_fs: int = None,
    window_ms: int = 250,
    step_ms: int = 50,
    selected_channels: list = None,
    show_video: bool = False,
    overwrite: bool = False,
    verbose: bool = False,
) -> tuple:
    """
    Load or create the feature dataset for EMG and joint angles using Open Ephys data.

    Parameters
    ----------
    root_dir : str
        Root directory containing session data
    label : str
        Session label identifier
    oebin_path : str, optional
        Explicit path to .oebin file or folder
    target_emg_fs : int, optional
        Target EMG sampling frequency for resampling. If None, use original fs
    window_ms : int
        Feature extraction window size in milliseconds
    step_ms : int
        Window step size in milliseconds
    selected_channels : list of int, optional
        Specific channels to use. If None, use all channels
    show_video : bool
        Display video during landmark extraction
    overwrite : bool
        Regenerate dataset even if it exists
    verbose : bool
        Enable verbose output

    Returns
    -------
    X : np.ndarray
        Feature matrix, shape (n_windows, n_channels × n_features)
    y : np.ndarray
        Label matrix (joint angles), shape (n_windows, 5)
    emg_fs : float
        EMG sampling frequency (after resampling if applicable)
    lm_fs : int
        Landmark sampling frequency
    """
    # Check if dataset already exists
    save_path = os.path.join(
        root_dir, f"{label}_training_dataset.npz" if label else "training_dataset.npz"
    )
    if os.path.exists(save_path) and not overwrite:
        print(
            f"[INFO] Feature dataset already exists at {save_path}. Use --overwrite to regenerate."
        )
        data = np.load(save_path, allow_pickle=True)
        return data["features"], data["labels"], data["emg_fs"], data["lm_fs"]

    # Load session data using Open Ephys loader
    loader = OEphysSessionLoader(root_dir, label, oebin_path, verbose)

    # Get EMG data
    emg, emg_fs, emg_t = loader.load_emg()

    # Select specific channels if requested
    if selected_channels is not None:
        print(f"[INFO] Using only selected EMG channels: {selected_channels}")
        emg = emg[selected_channels, :]

    original_emg_fs = emg_fs

    # Resample EMG if target_emg_fs is specified
    if target_emg_fs is not None and emg_fs != target_emg_fs:
        print(f"[INFO] Resampling EMG from {emg_fs} Hz to {target_emg_fs} Hz")
        num_samples = int(len(emg_t) * target_emg_fs / emg_fs)
        emg = resample(emg, num_samples, axis=1)
        emg_fs = target_emg_fs
        emg_t = np.linspace(emg_t[0], emg_t[-1], num_samples)
    else:
        print(f"[INFO] Using original EMG sampling frequency: {emg_fs} Hz")

    # Get start/end indices from events
    start_idx, end_idx = loader.load_event_indices()
    if start_idx is None or end_idx is None:
        print("[WARNING] Could not determine start and end indices from event files.")
        print("[WARNING] Using full EMG recording.")
        start_idx, end_idx = 0, emg.shape[1]

    # Rescale event indices if resampled
    if target_emg_fs is not None and original_emg_fs != target_emg_fs:
        scale = target_emg_fs / original_emg_fs
        start_idx = int(start_idx * scale)
        end_idx = int(end_idx * scale)
        print(
            f"[INFO] Rescaled event indices for resampled data: start={start_idx}, end={end_idx}"
        )

    # Load landmarks
    landmarks, lm_fs, lm_t = loader.load_landmarks()
    if landmarks is None:
        raise FileNotFoundError(
            "Landmark data not found. Please extract landmarks first using MediaPipe.\n"
            "See examples/02_data_extraction/ for landmark extraction tools."
        )

    # Time alignment: apply sync offset
    sync_offset = loader.load_sync_offset()
    lm_t = lm_t + sync_offset
    emg_t = emg_t[: emg.shape[1]]  # Ensure same length if clipped

    if verbose:
        print("\n[INFO] Time alignment:")
        print(f"|   EMG time range: {emg_t[0]:.3f} to {emg_t[-1]:.3f} seconds")
        print(
            f"|   Landmark time range (after sync): {lm_t[0]:.3f} to {lm_t[-1]:.3f} seconds"
        )
        print(f"|   Sync offset applied: {sync_offset:.3f} seconds")

    # Interpolate landmarks to EMG timebase
    landmarks_interp = interpolate_array_to_timebase(lm_t, emg_t, landmarks)
    if verbose:
        print(
            f"[INFO] Interpolated landmarks to EMG time vector. New shape: {landmarks_interp.shape}"
        )

    # Extract joint angles from landmarks
    y = get_angles_from_landmarks(
        landmarks_interp,
        emg_fs,
        start_idx,
        end_idx,
        window_ms=window_ms,
        step_ms=step_ms,
    )
    print(f"[INFO] Angle data shape: {y.shape} (windows, 5 fingers)")

    # Preprocess EMG data
    preprocessor = EMGPreprocessor(emg_fs, verbose=verbose)
    emg_filtered = preprocessor.preprocess(emg)

    # Extract EMG features
    X = preprocessor.extract_emg_features(
        emg_filtered, start_idx, end_idx, window_ms=window_ms, step_ms=step_ms
    )
    print(f"[INFO] Feature data shape: {X.shape} (windows, features)")

    # Verify shapes match
    if X.shape[0] != y.shape[0]:
        print(
            f"[WARNING] Feature and label counts don't match: {X.shape[0]} vs {y.shape[0]}"
        )
        min_len = min(X.shape[0], y.shape[0])
        X = X[:min_len]
        y = y[:min_len]
        print(f"[WARNING] Truncated to {min_len} samples")

    # Save dataset
    print(f"\n[INFO] Saving dataset to: {save_path}")
    np.savez(save_path, features=X, labels=y, emg_fs=emg_fs, lm_fs=lm_fs)
    print("[INFO] ✓ Dataset saved successfully")
    print(f"[INFO]   Features: {X.shape}")
    print(f"[INFO]   Labels: {y.shape}")

    return X, y, emg_fs, lm_fs


def merge_datasets(
    root_dir: str,
    labels: list,
    output_file: str = "combined_training_dataset.npz",
    **kwargs,
) -> tuple:
    """
    Create and merge multiple training datasets.

    Parameters
    ----------
    root_dir : str
        Root directory for all sessions
    labels : list of str
        List of session labels to merge
    output_file : str
        Output filename for merged dataset
    **kwargs : dict
        Additional arguments passed to create_dataset()

    Returns
    -------
    X_merged : np.ndarray
        Merged feature matrix
    y_merged : np.ndarray
        Merged label matrix
    """
    all_X, all_y = [], []

    for label in labels:
        print(f"\n{'=' * 60}")
        print(f"Processing session: {label}")
        print(f"{'=' * 60}")

        X, y, emg_fs, lm_fs = create_dataset(root_dir=root_dir, label=label, **kwargs)

        all_X.append(X)
        all_y.append(y)

    # Merge all datasets
    X_merged = np.vstack(all_X)
    y_merged = np.vstack(all_y)

    print(f"\n{'=' * 60}")
    print("[INFO] Merged dataset shapes:")
    print(f"  X: {X_merged.shape}")
    print(f"  y: {y_merged.shape}")
    print(f"{'=' * 60}")

    # Save merged dataset
    save_path = os.path.join(root_dir, output_file)
    np.savez(save_path, features=X_merged, labels=y_merged, emg_fs=emg_fs, lm_fs=lm_fs)
    print(f"\n✅ Merged dataset saved to: {save_path}")

    return X_merged, y_merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create training dataset from Open Ephys EMG data and video landmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--root_dir",
        required=True,
        help="Root directory containing session data (landmarks, events, etc.)",
    )
    parser.add_argument("--label", required=True, help="Session label identifier")

    # Data source
    parser.add_argument(
        "--oebin_path",
        default=None,
        help="Explicit path to .oebin file or folder. If not provided, will search in root_dir/raw/{label}/",
    )

    # Processing parameters
    parser.add_argument(
        "--target_emg_fs",
        type=int,
        default=None,
        help="Target EMG sampling frequency for resampling. If not specified, uses original fs",
    )
    parser.add_argument(
        "--window_ms",
        type=int,
        default=250,
        help="Window size in milliseconds for feature extraction",
    )
    parser.add_argument(
        "--step_ms",
        type=int,
        default=50,
        help="Step size in milliseconds for feature extraction",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        type=int,
        default=None,
        help="Specific channels to use (e.g., --channels 0 1 2 3). If not specified, uses all channels",
    )

    # Flags
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate dataset even if it already exists",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--show_video",
        action="store_true",
        help="Display video during landmark extraction (if landmarks need to be generated)",
    )

    # Merge multiple sessions
    parser.add_argument(
        "--merge_labels",
        nargs="+",
        default=None,
        help="Merge multiple sessions (e.g., --merge_labels Session1 Session2 Session3)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="combined_training_dataset.npz",
        help="Output filename for merged dataset",
    )

    args = parser.parse_args()

    # Check if merging multiple datasets
    if args.merge_labels:
        print(
            f"\n[INFO] Merging {len(args.merge_labels)} sessions: {args.merge_labels}"
        )
        X, y = merge_datasets(
            root_dir=args.root_dir,
            labels=args.merge_labels,
            output_file=args.output_file,
            oebin_path=args.oebin_path,
            target_emg_fs=args.target_emg_fs,
            window_ms=args.window_ms,
            step_ms=args.step_ms,
            selected_channels=args.channels,
            show_video=args.show_video,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
    else:
        # Single session
        print(f"\n[INFO] Creating dataset for session: {args.label}")
        X, y, emg_fs, lm_fs = create_dataset(
            root_dir=args.root_dir,
            label=args.label,
            oebin_path=args.oebin_path,
            target_emg_fs=args.target_emg_fs,
            window_ms=args.window_ms,
            step_ms=args.step_ms,
            selected_channels=args.channels,
            show_video=args.show_video,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )

        print("\n✅ Dataset creation complete!")
        print(f"   Features: {X.shape}")
        print(f"   Labels: {y.shape}")
        print(f"   EMG fs: {emg_fs} Hz")
        print(f"   Landmark fs: {lm_fs} Hz")
