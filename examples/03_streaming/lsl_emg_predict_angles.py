"""
Predict joint angles from EMG signals streamed via LSL.

This example demonstrates how to:
1. Receive EMG data from an LSL stream
2. Apply a trained model to predict joint angles
3. Visualize or stream the predicted angles

NOTE: This is a placeholder script. Full implementation requires:
- A trained model file (e.g., from examples/Joint_Kinematics_from_EMG/)
- An active LSL stream of EMG data with compatible channel configuration

Usage:
    python lsl_emg_predict_angles.py --emg_stream EMGStream --model path/to/model.pkl

Requirements:
    pip install pylsl scikit-learn joblib
"""

import argparse
import sys
import os
import numpy as np

# Add parent directory to path if handtrack is not installed
src_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
sys.path.insert(0, os.path.abspath(src_path))

try:
    from pylsl import StreamInlet, resolve_stream
    HAS_LSL = True
except ImportError:
    HAS_LSL = False
    print("Warning: pylsl not installed. Install with: pip install pylsl")

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    print("Warning: joblib not installed. Install with: pip install joblib")


def find_emg_stream(stream_name=None, timeout=5.0):
    """
    Find and connect to an EMG LSL stream.
    
    Args:
        stream_name (str): Name of the stream to find. If None, finds first EMG stream.
        timeout (float): Timeout in seconds to wait for stream.
        
    Returns:
        StreamInlet: Connected LSL inlet, or None if not found.
    """
    print(f"Looking for EMG stream{f' named {stream_name}' if stream_name else ''}...")
    
    if stream_name:
        streams = resolve_stream('name', stream_name, timeout=timeout)
    else:
        # Try to find any EMG-type stream
        streams = resolve_stream('type', 'EMG', timeout=timeout)
    
    if not streams:
        print(f"No EMG stream found within {timeout} seconds.")
        return None
    
    inlet = StreamInlet(streams[0])
    info = inlet.info()
    print(f"Connected to stream: {info.name()}")
    print(f"  Channels: {info.channel_count()}")
    print(f"  Sample rate: {info.nominal_srate()} Hz")
    
    return inlet


def load_model(model_path):
    """
    Load a trained prediction model.
    
    Args:
        model_path (str): Path to the saved model file (.pkl or .joblib)
        
    Returns:
        model: Loaded model object, or None if failed.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return None
    
    try:
        model = joblib.load(model_path)
        print(f"Loaded model from: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def predict_angles(inlet, model, window_size=50, visualize=True):
    """
    Predict joint angles from streaming EMG data.
    
    Args:
        inlet: LSL StreamInlet for EMG data
        model: Trained prediction model
        window_size (int): Number of samples to buffer before prediction
        visualize (bool): Whether to print predictions
    """
    print("\nStarting prediction loop (Ctrl+C to stop)...")
    print("=" * 60)
    
    buffer = []
    sample_count = 0
    
    try:
        while True:
            # Pull sample from LSL
            sample, timestamp = inlet.pull_sample(timeout=1.0)
            
            if sample is None:
                continue
            
            buffer.append(sample)
            sample_count += 1
            
            # Make prediction when buffer is full
            if len(buffer) >= window_size:
                # Prepare input features
                window = np.array(buffer[-window_size:])
                
                # Extract features (example: mean and std of each channel)
                features = np.concatenate([
                    window.mean(axis=0),
                    window.std(axis=0)
                ]).reshape(1, -1)
                
                # Predict
                try:
                    prediction = model.predict(features)
                    
                    if visualize:
                        print(f"\rSample {sample_count:6d} | Predicted angles: {prediction[0]}", end="")
                        
                except Exception as e:
                    print(f"\nPrediction error: {e}")
                
                # Slide window
                buffer = buffer[-window_size:]
    
    except KeyboardInterrupt:
        print("\n\nPrediction stopped by user.")
    
    print("=" * 60)
    print(f"Processed {sample_count} samples")


def main():
    parser = argparse.ArgumentParser(
        description="Predict joint angles from EMG signals streamed via LSL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Connect to default EMG stream with a trained model
  python lsl_emg_predict_angles.py --model ./models/emg_to_angles.pkl
  
  # Connect to a specific named stream
  python lsl_emg_predict_angles.py --emg_stream MyEMGStream --model ./model.pkl
  
  # Adjust prediction window size
  python lsl_emg_predict_angles.py --model ./model.pkl --window_size 100

Note: This requires a trained model file. See examples/Joint_Kinematics_from_EMG/
for training scripts.
        """
    )
    
    parser.add_argument("--emg_stream", type=str, default=None,
                       help="Name of the EMG LSL stream (default: auto-detect)")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model file (.pkl or .joblib)")
    parser.add_argument("--window_size", type=int, default=50,
                       help="Number of samples to buffer before prediction (default: 50)")
    parser.add_argument("--timeout", type=float, default=10.0,
                       help="Timeout to wait for stream (default: 10 seconds)")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not HAS_LSL:
        print("Error: pylsl is required. Install with: pip install pylsl")
        sys.exit(1)
    
    if not HAS_JOBLIB:
        print("Error: joblib is required. Install with: pip install joblib")
        sys.exit(1)
    
    # Load model
    model = load_model(args.model)
    if model is None:
        sys.exit(1)
    
    # Find and connect to EMG stream
    inlet = find_emg_stream(args.emg_stream, timeout=args.timeout)
    if inlet is None:
        sys.exit(1)
    
    # Run prediction loop
    predict_angles(inlet, model, window_size=args.window_size)


if __name__ == "__main__":
    main()
