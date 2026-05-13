"""
SessionLoader adapter for Open Ephys data sources.

This module provides a SessionLoader class compatible with the handtrack.io.SessionLoader
interface, but designed to work with Open Ephys data formats (.oebin, .npz from pyoephys).

Key differences from original SessionLoader:
- Uses pyoephys.io.load_open_ephys_session() instead of loading Intan .rhd files
- Supports .oebin files and Open Ephys-formatted .npz files
- Can work with ZMQ streaming (for real-time applications, see oephys_realtime_predict.py)
- Returns data in the same format expected by handtrack.processing.EMGPreprocessor

Author: NML (Neuro-Mechatronics Lab)
Created: 2026-02-16
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from handtrack.io import extract_event_indices, get_sync_offset
from handtrack.tracker import HandTracker

# Import Open Ephys loader
try:
    from pyoephys.io import load_open_ephys_session
except ImportError:
    raise ImportError(
        "python-open-ephys (pyoephys) is required for this module. "
        "Install with: pip install --index-url https://test.pypi.org/simple/ --no-deps python-oephys"
    )


def setup_logger():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


setup_logger()


class OEphysSessionLoader:
    """
    SessionLoader for Open Ephys data, compatible with handtrack.io.SessionLoader interface.

    This class loads EMG data from Open Ephys recordings (.oebin) and provides the same
    interface as handtrack.io.SessionLoader for seamless integration with the Joint_Kinematics_from_EMG
    pipeline.

    Parameters
    ----------
    root_dir : str
        Root directory containing session data (landmarks, events, etc.)
    label : str
        Session label identifier
    oebin_path : str, optional
        Explicit path to .oebin file or folder containing it. If None, will search
        in root_dir/raw/{label}/
    verbose : bool, default=False
        Enable verbose logging

    Examples
    --------
    >>> from oephys_session_loader import OEphysSessionLoader
    >>> loader = OEphysSessionLoader('/path/to/data', 'MySession',
    ...                               oebin_path='/path/to/recording.oebin')
    >>> emg, fs, t = loader.load_emg()
    >>> print(f"Loaded {emg.shape[0]} channels at {fs} Hz")
    """

    def __init__(
        self,
        root_dir: str,
        label: str,
        oebin_path: Optional[str] = None,
        verbose: bool = False,
    ):
        self.root_dir = str(root_dir)
        self.label = str(label)
        self.label_folder = label.rstrip("_")
        self.oebin_path = oebin_path
        self.verbose = bool(verbose)
        self.emg_shape = None
        self.emg_fs = None
        self._session_data = None  # Cache loaded session data

    def _find_oebin_or_npz(self) -> str:
        """
        Find .oebin file or pre-converted .npz file for this session.

        Search order:
        1. Explicit oebin_path if provided
        2. {root_dir}/raw/{label}/{label}_emg_data.npz (pre-converted)
        3. {root_dir}/raw/{label}/Record_Node_*/experiment*/recording*/structure.oebin

        Returns
        -------
        path : str
            Path to .oebin file, folder containing it, or .npz file

        Raises
        ------
        FileNotFoundError
            If no valid data source is found
        """
        # Case 1: Explicit path provided
        if self.oebin_path is not None:
            p = Path(self.oebin_path)
            if p.exists():
                if self.verbose:
                    print(f"| Using explicit path: {p}")
                return str(p)
            else:
                raise FileNotFoundError(f"Provided oebin_path does not exist: {p}")

        # Case 2: Pre-converted NPZ in expected location
        npz_path = (
            Path(self.root_dir)
            / "raw"
            / self.label_folder
            / f"{self.label}_emg_data.npz"
        )
        if npz_path.exists():
            if self.verbose:
                print(f"| Found pre-converted NPZ: {npz_path}")
            return str(npz_path)

        # Case 3: Search for structure.oebin in raw folder
        raw_folder = Path(self.root_dir) / "raw" / self.label_folder
        if not raw_folder.exists():
            raise FileNotFoundError(
                f"Raw data folder not found: {raw_folder}\n"
                f"Expected structure: {self.root_dir}/raw/{self.label_folder}/"
            )

        # Search for structure.oebin recursively
        oebin_files = list(raw_folder.rglob("structure.oebin"))
        if not oebin_files:
            raise FileNotFoundError(
                f"No structure.oebin file found in {raw_folder}\n"
                f"Expected Open Ephys Binary Format recording."
            )

        if len(oebin_files) > 1:
            print(
                f"[WARNING] Multiple .oebin files found. Using first: {oebin_files[0]}"
            )

        oebin_path = oebin_files[0]
        if self.verbose:
            print(f"| Found .oebin file: {oebin_path}")

        return str(oebin_path)

    def load_emg(self) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Load EMG data from Open Ephys recording.

        Returns
        -------
        emg : np.ndarray
            EMG data array of shape (n_channels, n_samples)
        fs : float
            Sampling frequency in Hz
        t : np.ndarray
            Time vector in seconds, shape (n_samples,)

        Examples
        --------
        >>> loader = OEphysSessionLoader('/path/to/data', 'Session1')
        >>> emg, fs, t = loader.load_emg()
        >>> print(f"EMG shape: {emg.shape}, fs: {fs} Hz")
        """
        if self.verbose:
            print("\n================== EMG (Open Ephys) =================")
            print(f"| Searching for EMG data for session '{self.label}'...")

        # Find data source
        data_path = self._find_oebin_or_npz()

        # Load using pyoephys
        if self.verbose:
            print(f"| Loading data from: {data_path}")

        try:
            session = load_open_ephys_session(data_path)
            self._session_data = session  # Cache for potential reuse
        except Exception as e:
            raise RuntimeError(f"Failed to load Open Ephys data from {data_path}: {e}")

        # Extract data in handtrack-compatible format
        emg = session["amplifier_data"]  # shape (C, S)
        fs = float(session["sample_rate"])
        t = session["t_amplifier"]  # shape (S,)

        # Validate shapes
        if emg.ndim != 2:
            raise ValueError(
                f"Expected 2D EMG array (channels, samples), got shape {emg.shape}"
            )

        if t.shape[0] != emg.shape[1]:
            raise ValueError(
                f"Time vector length {t.shape[0]} does not match EMG samples {emg.shape[1]}"
            )

        self.emg_shape = emg.shape
        self.emg_fs = fs

        if self.verbose:
            print("| EMG data loaded successfully")
            print(f"|   Shape: {emg.shape} (channels, samples)")
            print(f"|   Sampling rate: {fs} Hz")
            print(f"|   Duration: {t[-1] - t[0]:.2f} seconds")
            print(f"|   Time range: {t[0]:.3f} to {t[-1]:.3f} seconds")
            print(f"|   Channel names: {session.get('channel_names', 'N/A')}")

        return emg, fs, t

    def load_landmarks(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[int], Optional[np.ndarray]]:
        """
        Load smoothed landmarks from the session.

        This method is identical to handtrack.io.SessionLoader.load_landmarks()
        and is included for interface compatibility.

        Returns
        -------
        landmarks : np.ndarray or None
            Smoothed landmarks array of shape (n_frames, 63) or None if not found
        fs : int or None
            Sampling frequency of landmarks (typically 30 Hz for video)
        t : np.ndarray or None
            Time vector corresponding to landmarks
        """
        landmarks_file = (
            f"{self.label}_landmarks.npz" if self.label else "landmarks.npz"
        )
        landmarks_path = Path(self.root_dir) / "landmarks"

        if self.verbose:
            print("\n================== Landmarks =================")
            print(f"|  Searching for landmarks in {self.root_dir}")

        # Search in landmarks folder first
        full_path = landmarks_path / landmarks_file
        if not full_path.exists():
            # Try root directory
            full_path = Path(self.root_dir) / landmarks_file
            if not full_path.exists():
                if self.verbose:
                    print(f"|  Landmarks file '{landmarks_file}' not found.")
                return None, None, None

        if self.verbose:
            print(f"|  Loading smoothed landmarks from {full_path}")

        # Load landmarks
        data = np.load(full_path, allow_pickle=True)
        landmarks = data["landmarks"]
        fs = int(data.get("sampling_rate", 30))
        t = data.get("time_vector", np.arange(landmarks.shape[0]) / fs)

        if self.verbose:
            print(f"| Loaded landmarks: {landmarks.shape} at {fs} Hz")

        return landmarks, fs, t

    def get_landmarks_from_video(
        self,
        video_path: Optional[str] = None,
        apply_kalman: bool = True,
        show_video: bool = False,
        save_video: bool = False,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Extract landmarks from video file using MediaPipe HandTracker.

        Parameters
        ----------
        video_path : str, optional
            Path to video file. If None, searches in {root_dir}/media/{label}.mp4
        apply_kalman : bool, default=True
            Apply Kalman filtering to smooth landmarks
        show_video : bool, default=False
            Display video during processing
        save_video : bool, default=False
            Save annotated video
        verbose : bool, default=False
            Enable verbose output

        Returns
        -------
        landmarks : np.ndarray
            Extracted landmarks, shape (n_frames, 63)
        fs : int
            Sampling frequency (video FPS)
        t : np.ndarray
            Time vector in seconds
        """
        if self.verbose or verbose:
            print("\n================== Video Landmark Extraction =================")
            print("|  Extracting landmarks from video...")

        # Determine video path
        if video_path is None:
            video_path = Path(self.root_dir) / "media" / f"{self.label}.mp4"
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")

        video_path = str(video_path)
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Use HandTracker from handtrack
        tracker = HandTracker(
            source=video_path, apply_kalman=apply_kalman, verbose=verbose
        )
        landmarks, metadata = tracker.extract_landmarks(
            visualize=show_video, save_video=save_video
        )

        return landmarks, metadata["sampling_rate"], metadata["time_vector"]

    def load_event_indices(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Load start and end event indices from event files.

        Returns
        -------
        start_idx : int or None
            Start index in EMG samples
        end_idx : int or None
            End index in EMG samples
        """
        if self.verbose:
            print("\n================== Event Indices =================")
            print(f"|  Searching for event indices in {self.root_dir}/events...")

        events_dir = Path(self.root_dir) / "events"

        # Extract start and end indices
        indices = extract_event_indices(
            str(events_dir), label=self.label, verbose=self.verbose
        )

        if not indices.get("start_idx") or not indices.get("end_idx"):
            if self.verbose:
                print(f"|  No valid start/end indices found for session: {self.label}")
            return None, None

        # For now, use the first result
        start_idx = indices["start_idx"][0]
        end_idx = indices["end_idx"][0]

        # Apply buffer
        if start_idx is not None and end_idx is not None:
            buffer = int(0.5 * self.emg_fs) if self.emg_fs else 2500
            start_idx = max(0, start_idx - buffer)
            if self.verbose:
                print(
                    f"|  Event indices: start={start_idx}, end={end_idx} (with buffer)"
                )
        else:
            # Use full recording
            if self.emg_shape is not None:
                start_idx, end_idx = 0, self.emg_shape[1]
                if self.verbose:
                    print(f"|  Using full recording: 0 to {end_idx}")

        return start_idx, end_idx

    def load_sync_offset(self) -> float:
        """
        Load or compute synchronization offset between EMG and video.

        Returns
        -------
        sync_offset : float
            Time offset in seconds to add to landmark timestamps
        """
        if self.verbose:
            print("\n================== Sync Offset =================")
            print(f"|  Searching for sync offset file in {self.root_dir}/events...")

        events_dir = Path(self.root_dir) / "events"
        if not events_dir.is_dir():
            if self.verbose:
                print("|  Events directory not found.")
            return 0.0

        # Check for existing sync offset file
        filename = f"{self.label}_sync_offset.txt" if self.label else "sync_offset.txt"
        filepath = events_dir / filename

        if filepath.exists():
            if self.verbose:
                print(f"|  Sync file found: '{filepath}'")
            with open(filepath, "r") as f:
                line = f.readline()
                sync_offset = float(line.split(": ")[1].split(" ")[0])
            if self.verbose:
                print(f"|  Sync offset loaded: {sync_offset:.3f} seconds")
            return sync_offset

        # Attempt to compute sync offset
        if self.verbose:
            print("|  Sync offset file not found. Attempting to compute...")

        try:
            sync_offset = get_sync_offset(
                root_dir=self.root_dir,
                label=self.label,
                save_file=True,
                emg_fs=self.emg_fs or 5000,  # fallback
            )
            if self.verbose:
                print(f"|  Computed sync offset: {sync_offset:.3f} seconds")
            return sync_offset
        except Exception as e:
            print(f"[ERROR] Failed to compute sync offset: {e}")
            print("[WARNING] Returning zero offset. Results may be inaccurate.")
            return 0.0

    def get_session_info(self) -> Dict[str, Any]:
        """
        Get summary information about the loaded session.

        Returns
        -------
        info : dict
            Dictionary containing session metadata
        """
        if self._session_data is None:
            self.load_emg()  # Ensure data is loaded

        return {
            "label": self.label,
            "root_dir": self.root_dir,
            "emg_shape": self.emg_shape,
            "emg_fs": self.emg_fs,
            "channel_names": self._session_data.get("channel_names")
            if self._session_data
            else None,
            "data_source": self.oebin_path or "auto-detected",
        }


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Test OEphysSessionLoader")
    parser.add_argument("--root_dir", required=True, help="Root data directory")
    parser.add_argument("--label", required=True, help="Session label")
    parser.add_argument(
        "--oebin_path", default=None, help="Explicit path to .oebin file"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Create loader
    loader = OEphysSessionLoader(
        root_dir=args.root_dir,
        label=args.label,
        oebin_path=args.oebin_path,
        verbose=args.verbose,
    )

    # Test loading
    print("\n=== Testing EMG load ===")
    emg, fs, t = loader.load_emg()
    print(f"✓ EMG: {emg.shape}, {fs} Hz, {t[-1]:.2f} s")

    print("\n=== Testing landmarks load ===")
    landmarks, lm_fs, lm_t = loader.load_landmarks()
    if landmarks is not None:
        print(f"✓ Landmarks: {landmarks.shape}, {lm_fs} Hz")
    else:
        print("✗ No landmarks found")

    print("\n=== Testing event indices ===")
    start, end = loader.load_event_indices()
    print(f"✓ Events: start={start}, end={end}")

    print("\n=== Testing sync offset ===")
    offset = loader.load_sync_offset()
    print(f"✓ Sync offset: {offset:.3f} s")

    print("\n=== Session info ===")
    info = loader.get_session_info()
    for k, v in info.items():
        print(f"  {k}: {v}")
