"""
Real-time joint angle prediction from Open Ephys ZMQ streaming EMG data.

This script demonstrates real-time kinematic decoding by:
1. Connecting to Open Ephys GUI via ZMQ interface
2. Buffering incoming EMG data
3. Extracting features in sliding windows
4. Predicting joint angles using trained PyTorch model
5. (Optional) Streaming predictions via LSL for downstream applications

Usage:
    # Basic usage
    python oephys_realtime_predict.py --model_path /path/to/model.pth --zmq_host 127.0.0.1 --zmq_port 5556

    # With LSL output
    python oephys_realtime_predict.py --model_path /path/to/model.pth --lsl_outlet

    # With specific channels
    python oephys_realtime_predict.py --model_path /path/to/model.pth --channels 0 1 2 3 4 5 6 7

Requirements:
    - Open Ephys GUI running with ZMQ Interface plugin enabled
    - Trained EMGRegressor model and scaler
    - python-open-ephys (pyoephys) installed

Author: NML (Neuro-Mechatronics Lab)
Created: 2026-02-16
"""

import argparse
import time
from pathlib import Path

import torch

# Import Open Ephys ZMQ client
try:
    from pyoephys.interface import ZMQClient
except ImportError:
    raise ImportError(
        "python-open-ephys is required. Install with:\n"
        "pip install --index-url https://test.pypi.org/simple/ --no-deps python-oephys"
    )

# Import handtrack modules
from handtrack.ml import EMGRegressor
from handtrack.processing import (
    EMGPreprocessor,
    KalmanAngle,
    bandpass_filter,
    extract_features,
    lowpass_filter,
    notch_filter,
    rectify,
)

# Optional: LSL for streaming predictions
try:
    import pylsl

    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False
    print("[WARNING] pylsl not available. LSL streaming disabled.")


class RealtimeEMGPredictor:
    """
    Real-time EMG-to-joint-angles predictor using Open Ephys ZMQ streaming.

    Parameters
    ----------
    model_path : str
        Path to trained PyTorch model weights (.pth)
    scaler_path : str
        Path to saved scaler (.pkl)
    zmq_host : str
        ZMQ host IP address (default: '127.0.0.1')
    zmq_port : int
        ZMQ data port (default: 5556)
    emg_fs : float
        Expected EMG sampling frequency in Hz
    channels : list of int, optional
        Specific channels to use. If None, uses all available channels
    window_ms : int
        Feature extraction window size in milliseconds
    step_ms : int
        Window step size in milliseconds
    buffer_seconds : float
        ZMQ client buffer duration in seconds
    lsl_outlet_name : str, optional
        LSL outlet name for streaming predictions. If None, no LSL streaming
    apply_kalman : bool
        Apply Kalman filtering to smooth predictions
    verbose : bool
        Enable verbose output
    """

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        zmq_host: str = "127.0.0.1",
        zmq_port: int = 5556,
        emg_fs: float = 5000.0,
        channels: list = None,
        window_ms: int = 250,
        step_ms: int = 50,
        buffer_seconds: float = 30.0,
        lsl_outlet_name: str = None,
        apply_kalman: bool = True,
        verbose: bool = False,
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.zmq_host = zmq_host
        self.zmq_port = zmq_port
        self.emg_fs = emg_fs
        self.channels = channels
        self.window_ms = window_ms
        self.step_ms = step_ms
        self.verbose = verbose

        # Compute window/step sizes in samples
        self.window_size = int(window_ms * emg_fs / 1000)
        self.step_size = int(step_ms * emg_fs / 1000)

        # Initialize ZMQ client
        if self.verbose:
            print(f"[INFO] Connecting to Open Ephys ZMQ at {zmq_host}:{zmq_port}...")

        self.zmq_client = ZMQClient(
            host_ip=zmq_host,
            data_port=str(zmq_port),
            buffer_seconds=buffer_seconds,
            expected_channel_count=len(channels) if channels else None,
            auto_start=True,
            verbose=verbose,
        )

        # Wait for ZMQ client to be ready
        if not self.zmq_client.ready_event.wait(timeout=10.0):
            raise RuntimeError("ZMQ client failed to connect within 10 seconds")

        if self.verbose:
            print(f"[INFO] ZMQ client connected. Detected fs: {self.zmq_client.fs} Hz")

        # Update fs from detected value if different
        if abs(self.zmq_client.fs - emg_fs) > 1.0:
            print(
                f"[WARNING] Detected fs ({self.zmq_client.fs} Hz) differs from expected ({emg_fs} Hz)"
            )
            print(f"[WARNING] Using detected fs: {self.zmq_client.fs} Hz")
            self.emg_fs = self.zmq_client.fs
            self.window_size = int(window_ms * self.emg_fs / 1000)
            self.step_size = int(step_ms * self.emg_fs / 1000)

        # Load model
        if self.verbose:
            print(f"[INFO] Loading model from {self.model_path}...")

        # Need to determine input_dim from model file or scaler
        # For now, assume we can infer from scaler or load model state dict
        self.model, self.scaler = self._load_model_and_scaler()

        # Initialize preprocessing
        self.preprocessor = EMGPreprocessor(
            fs=self.emg_fs,
            band=(20, 450),
            notch_freq=60,
            envelope_cutoff=5,
            verbose=False,
        )

        # Kalman filters for smoothing (5 angles)
        self.kalman_filters = None
        if apply_kalman:
            self.kalman_filters = [
                KalmanAngle(process_noise=1e-4, measurement_noise=1e-3)
                for _ in range(5)
            ]

        # LSL outlet
        self.lsl_outlet = None
        if lsl_outlet_name and LSL_AVAILABLE:
            if self.verbose:
                print(f"[INFO] Creating LSL outlet: {lsl_outlet_name}")
            info = pylsl.StreamInfo(
                name=lsl_outlet_name,
                type="JointAngles",
                channel_count=5,
                nominal_srate=1000.0 / step_ms,  # Approximate prediction rate
                channel_format=pylsl.cf_float32,
                source_id="oephys_emg_regressor",
            )
            # Add channel labels
            chns = info.desc().append_child("channels")
            for label in ["thumb", "index", "middle", "ring", "pinky"]:
                chns.append_child("channel").append_child_value("label", label)
            self.lsl_outlet = pylsl.StreamOutlet(info)
            if self.verbose:
                print("[INFO] LSL outlet created")

        # Internal state
        self.running = False
        self.prediction_count = 0
        self.start_time = None
        self.last_prediction_time = 0

    def _load_model_and_scaler(self):
        """Load PyTorch model and scaler."""
        import pickle

        # Load scaler
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")

        with open(self.scaler_path, "rb") as f:
            scaler = pickle.load(f)

        if self.verbose:
            print(f"[INFO] Loaded scaler from {self.scaler_path}")

        # Determine input_dim from scaler
        # Scaler transforms (n_samples, n_features) data
        # For StandardScaler, scaler.n_features_in_ gives the feature count
        input_dim = scaler.n_features_in_ if hasattr(scaler, "n_features_in_") else None

        if input_dim is None:
            raise ValueError(
                "Could not determine input_dim from scaler. Check scaler file."
            )

        # Create model
        output_dim = 5  # 5 joint angles
        model = EMGRegressor(input_dim=input_dim, output_dim=output_dim)

        # Load weights
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        state_dict = torch.load(self.model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        if self.verbose:
            print(f"[INFO] Loaded model from {self.model_path}")
            print(
                f"[INFO] Model architecture: input_dim={input_dim}, output_dim={output_dim}"
            )

        return model, scaler

    def _get_emg_window(self):
        """
        Get latest EMG window from ZMQ client buffer.

        Returns
        -------
        emg_window : np.ndarray or None
            EMG data for current window, shape (n_channels, window_size)
            Returns None if insufficient data available
        """
        # Get latest data from ZMQ client
        # We need window_size samples
        try:
            data = self.zmq_client.drain_new(self.window_size)

            if data is None or len(data) == 0:
                return None

            # data is (C, S) where S <= window_size
            if data.shape[1] < self.window_size:
                return None  # Not enough data yet

            # Take last window_size samples
            emg_window = data[:, -self.window_size :]

            # Channel selection
            if self.channels is not None:
                emg_window = emg_window[self.channels, :]

            return emg_window

        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Failed to get EMG window: {e}")
            return None

    def _preprocess_window(self, emg_window):
        """Apply preprocessing to EMG window."""
        # Apply filtering pipeline manually (faster than EMGPreprocessor for single window)
        emg = notch_filter(emg_window, self.emg_fs, 60)
        emg = bandpass_filter(emg, 20, 450, self.emg_fs)
        emg = rectify(emg)
        emg = lowpass_filter(emg, 5, self.emg_fs)
        return emg

    def _extract_features_from_window(self, emg_window):
        """Extract features from single EMG window."""
        features = extract_features(emg_window)
        return features

    def _predict_angles(self, features):
        """
        Predict joint angles from features.

        Parameters
        ----------
        features : np.ndarray
            Feature vector, shape (n_features,)

        Returns
        -------
        angles : np.ndarray
            Predicted joint angles, shape (5,)
        """
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Predict
        with torch.no_grad():
            features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
            angles_tensor = self.model(features_tensor)
            angles = angles_tensor.numpy().flatten()

        # Apply Kalman filtering
        if self.kalman_filters is not None:
            for i in range(len(angles)):
                angles[i] = self.kalman_filters[i].update(angles[i])

        return angles

    def predict_once(self):
        """
        Perform one prediction cycle.

        Returns
        -------
        angles : np.ndarray or None
            Predicted joint angles, shape (5,), or None if insufficient data
        """
        # Get EMG window
        emg_window = self._get_emg_window()
        if emg_window is None:
            return None

        # Preprocess
        emg_filtered = self._preprocess_window(emg_window)

        # Extract features
        features = self._extract_features_from_window(emg_filtered)

        # Predict
        angles = self._predict_angles(features)

        # Update stats
        self.prediction_count += 1
        self.last_prediction_time = time.time()

        # Stream via LSL
        if self.lsl_outlet is not None:
            self.lsl_outlet.push_sample(angles.tolist())

        return angles

    def run(self, duration: float = None, display_rate: float = 1.0):
        """
        Run real-time prediction loop.

        Parameters
        ----------
        duration : float, optional
            Run duration in seconds. If None, runs indefinitely
        display_rate : float
            Display update rate in Hz
        """
        self.running = True
        self.start_time = time.time()
        last_display = time.time()
        display_interval = 1.0 / display_rate

        print("\n" + "=" * 60)
        print("Real-time Joint Angle Prediction")
        print("=" * 60)
        print(f"Model: {self.model_path.name}")
        print(f"ZMQ: {self.zmq_host}:{self.zmq_port}")
        print(f"Sampling rate: {self.emg_fs} Hz")
        print(f"Window: {self.window_ms} ms, Step: {self.step_ms} ms")
        if self.channels:
            print(f"Channels: {self.channels}")
        print(f"LSL outlet: {'Enabled' if self.lsl_outlet else 'Disabled'}")
        print("=" * 60)
        print("\nPress Ctrl+C to stop\n")

        try:
            while self.running:
                # Check duration
                if duration is not None:
                    if time.time() - self.start_time > duration:
                        print("\n[INFO] Reached target duration. Stopping...")
                        break

                # Predict
                angles = self.predict_once()

                # Display periodically
                if (
                    angles is not None
                    and (time.time() - last_display) >= display_interval
                ):
                    elapsed = time.time() - self.start_time
                    pred_rate = self.prediction_count / elapsed if elapsed > 0 else 0

                    print(
                        f"\r[{elapsed:6.1f}s] Rate: {pred_rate:5.1f} Hz | "
                        f"Angles: [{', '.join([f'{a:6.2f}' for a in angles])}] deg",
                        end="",
                        flush=True,
                    )
                    last_display = time.time()

                # Small sleep to prevent CPU spinning (adjust as needed)
                time.sleep(0.001)

        except KeyboardInterrupt:
            print("\n\n[INFO] Interrupted by user")

        finally:
            self.stop()

    def stop(self):
        """Stop prediction and cleanup."""
        self.running = False

        if self.zmq_client:
            self.zmq_client.stop()

        elapsed = time.time() - self.start_time if self.start_time else 0
        avg_rate = self.prediction_count / elapsed if elapsed > 0 else 0

        print("\n" + "=" * 60)
        print("Session Summary")
        print("=" * 60)
        print(f"Total predictions: {self.prediction_count}")
        print(f"Duration: {elapsed:.1f} seconds")
        print(f"Average rate: {avg_rate:.1f} Hz")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time joint angle prediction from Open Ephys ZMQ streaming",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model_path", required=True, help="Path to trained PyTorch model (.pth)"
    )
    parser.add_argument(
        "--scaler_path", required=True, help="Path to saved scaler (.pkl)"
    )

    # ZMQ connection
    parser.add_argument("--zmq_host", default="127.0.0.1", help="ZMQ host IP address")
    parser.add_argument("--zmq_port", type=int, default=5556, help="ZMQ data port")

    # EMG parameters
    parser.add_argument(
        "--emg_fs",
        type=float,
        default=5000.0,
        help="Expected EMG sampling frequency (Hz)",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        type=int,
        default=None,
        help="Specific channels to use (e.g., --channels 0 1 2 3)",
    )
    parser.add_argument(
        "--window_ms", type=int, default=250, help="Feature extraction window size (ms)"
    )
    parser.add_argument("--step_ms", type=int, default=50, help="Window step size (ms)")

    # Behavior
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Run duration in seconds (default: indefinite)",
    )
    parser.add_argument(
        "--display_rate", type=float, default=10.0, help="Display update rate (Hz)"
    )
    parser.add_argument(
        "--no_kalman", action="store_true", help="Disable Kalman filtering"
    )

    # LSL
    parser.add_argument(
        "--lsl_outlet",
        action="store_true",
        help="Enable LSL outlet for streaming predictions",
    )
    parser.add_argument("--lsl_name", default="JointAngles", help="LSL outlet name")

    # Flags
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Check dependencies
    if args.lsl_outlet and not LSL_AVAILABLE:
        parser.error(
            "LSL outlet requested but pylsl is not installed. Install with: pip install pylsl"
        )

    # Create predictor
    predictor = RealtimeEMGPredictor(
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        zmq_host=args.zmq_host,
        zmq_port=args.zmq_port,
        emg_fs=args.emg_fs,
        channels=args.channels,
        window_ms=args.window_ms,
        step_ms=args.step_ms,
        lsl_outlet_name=args.lsl_name if args.lsl_outlet else None,
        apply_kalman=not args.no_kalman,
        verbose=args.verbose,
    )

    # Run
    predictor.run(duration=args.duration, display_rate=args.display_rate)
