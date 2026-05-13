"""
Offline prediction of joint angles from saved EMG datasets.

This script loads a training dataset (created by oephys_create_dataset.py) and
a trained model to predict joint angles and visualize the results.

Usage:
    python oephys_predict_angles.py --root_dir /path/to/data --label MySession --model_path /path/to/model.pth

Author: NML (Neuro-Mechatronics Lab)
Created: 2026-02-16
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Import handtrack modules
from handtrack.ml import EMGRegressor, ModelManager
from handtrack.processing import KalmanAngle


def predict_angles(
    root_dir: str,
    label: str,
    model_path: str,
    scaler_path: str = None,
    apply_kalman: bool = True,
    visualize: bool = True,
    save_predictions: bool = True,
    verbose: bool = False,
):
    """
    Predict joint angles from saved dataset.

    Parameters
    ----------
    root_dir : str
        Root directory containing dataset
    label : str
        Session label
    model_path : str
        Path to trained model (.pth file)
    scaler_path : str, optional
        Path to scaler (.pkl file). If None, searches in model directory
    apply_kalman : bool
        Apply Kalman filtering to smooth predictions
    visualize : bool
        Display prediction plots
    save_predictions : bool
        Save predictions to NPY file
    verbose : bool
        Enable verbose output

    Returns
    -------
    y_pred : np.ndarray
        Predicted joint angles, shape (n_windows, 5)
    y_true : np.ndarray
        Ground truth joint angles, shape (n_windows, 5)
    """
    # Load dataset
    data_path = (
        Path(root_dir) / f"{label}_training_dataset.npz"
        if label
        else Path(root_dir) / "training_dataset.npz"
    )

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {data_path}")

    if verbose:
        print(f"Loading dataset from {data_path}")

    data = np.load(data_path)
    X = data["features"]
    y_true = data["labels"]
    emg_fs = float(data["emg_fs"])
    lm_fs = float(data["lm_fs"])

    print("Dataset loaded:")
    print(f"  Features: {X.shape}")
    print(f"  Labels: {y_true.shape}")
    print(f"  EMG fs: {emg_fs} Hz")
    print(f"  Landmark fs: {lm_fs} Hz")

    # Determine scaler path
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if scaler_path is None:
        # Try to find scaler in same directory as model
        scaler_path = model_path.parent / f"{label}_scaler.pkl"
        if not scaler_path.exists():
            # Try with model name
            scaler_path = model_path.with_suffix(".pkl").parent / (
                model_path.stem.replace("_model", "_scaler") + ".pkl"
            )
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler file not found. Please specify --scaler_path explicitly.\n"
                f"Searched: {scaler_path}"
            )

    scaler_path = Path(scaler_path)
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    if verbose:
        print(f"\nLoading model from {model_path}")
        print(f"Loading scaler from {scaler_path}")

    # Load model using ModelManager
    manager = ModelManager(root_dir=str(root_dir), label=label, verbose=verbose)

    # Load model weights
    input_dim = X.shape[1]
    output_dim = y_true.shape[1]

    model = EMGRegressor(input_dim=input_dim, output_dim=output_dim)
    manager.load_model(model=model, weights=str(model_path))
    manager.scalar_path = str(scaler_path)
    manager.load_scalar()

    if not manager.model_exists:
        raise ValueError("Failed to load model")

    print("\nModel loaded successfully:")
    print(f"  Input dim: {input_dim}")
    print(f"  Output dim: {output_dim}")

    # Predict angles
    if verbose:
        print("\nPredicting joint angles...")

    y_pred = manager.predict(X)

    print(f"Prediction complete: {y_pred.shape}")

    # Apply Kalman filtering
    if apply_kalman:
        if verbose:
            print("Applying Kalman filtering...")

        kalman_filters = [
            KalmanAngle(process_noise=1e-4, measurement_noise=1e-3)
            for _ in range(y_pred.shape[1])
        ]

        for i in range(y_pred.shape[0]):
            for j in range(y_pred.shape[1]):
                y_pred[i, j] = kalman_filters[j].update(y_pred[i, j])

    # Compute metrics
    mse = np.mean((y_pred - y_true) ** 2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_true), axis=0)
    r2 = 1 - np.sum((y_pred - y_true) ** 2, axis=0) / np.sum(
        (y_true - np.mean(y_true, axis=0)) ** 2, axis=0
    )

    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    print("\nPrediction Metrics:")
    print(f"{'Finger':<10} {'RMSE (deg)':>12} {'MAE (deg)':>12} {'R²':>8}")
    print("-" * 45)
    for i, name in enumerate(finger_names):
        print(f"{name:<10} {rmse[i]:>12.3f} {mae[i]:>12.3f} {r2[i]:>8.3f}")
    print(
        f"{'Mean':<10} {np.mean(rmse):>12.3f} {np.mean(mae):>12.3f} {np.mean(r2):>8.3f}"
    )

    # Visualize
    if visualize:
        plot_predictions(y_pred, y_true, finger_names)

    # Save predictions
    if save_predictions:
        output_file = Path(root_dir) / f"{label}_predicted_angles.npy"
        np.save(output_file, y_pred)
        print(f"\nPredictions saved to: {output_file}")

    return y_pred, y_true


def plot_predictions(y_pred, y_true, finger_names, step_ms=50):
    """
    Plot predicted vs ground truth joint angles.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted angles, shape (n_windows, 5)
    y_true : np.ndarray
        Ground truth angles, shape (n_windows, 5)
    finger_names : list of str
        Finger names for subplot titles
    step_ms : float
        Window step size in milliseconds (for time axis)
    """
    # Create time axis
    time_axis = np.arange(y_pred.shape[0]) * step_ms / 1000.0  # Convert to seconds

    # Create figure with subplots
    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        "Predicted vs Ground Truth Joint Angles", fontsize=14, fontweight="bold"
    )

    for i, (ax, name) in enumerate(zip(axes, finger_names)):
        # Plot ground truth and prediction
        ax.plot(
            time_axis,
            y_true[:, i],
            label="Ground Truth",
            color="blue",
            linewidth=1.5,
            alpha=0.7,
        )
        ax.plot(
            time_axis,
            y_pred[:, i],
            label="Prediction",
            color="orange",
            linewidth=1.5,
            alpha=0.7,
        )

        # Compute error metrics for this finger
        mse = np.mean((y_pred[:, i] - y_true[:, i]) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum((y_pred[:, i] - y_true[:, i]) ** 2) / np.sum(
            (y_true[:, i] - np.mean(y_true[:, i])) ** 2
        )

        # Styling
        ax.set_ylabel("Angle (degrees)", fontsize=10)
        ax.set_title(f"{name} (RMSE: {rmse:.2f}°, R²: {r2:.3f})", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (seconds)", fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_error_analysis(y_pred, y_true, finger_names):
    """
    Plot error analysis (residuals, distributions).

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted angles
    y_true : np.ndarray
        Ground truth angles
    finger_names : list of str
        Finger names
    """
    errors = y_pred - y_true

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Error Analysis", fontsize=14, fontweight="bold")

    # Plot error distributions
    for i, name in enumerate(finger_names):
        ax = axes[i // 3, i % 3]
        ax.hist(errors[:, i], bins=50, alpha=0.7, color="steelblue", edgecolor="black")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_title(f"{name}", fontsize=11)
        ax.set_xlabel("Error (degrees)", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_error = np.mean(errors[:, i])
        std_error = np.std(errors[:, i])
        ax.text(
            0.05,
            0.95,
            f"μ={mean_error:.2f}°\nσ={std_error:.2f}°",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Remove last subplot (we have 5 fingers, not 6)
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict joint angles from saved EMG dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--root_dir", required=True, help="Root directory containing dataset"
    )
    parser.add_argument("--label", required=True, help="Session label")
    parser.add_argument(
        "--model_path", required=True, help="Path to trained model (.pth file)"
    )

    # Optional
    parser.add_argument(
        "--scaler_path",
        default=None,
        help="Path to scaler (.pkl file). If not specified, searches in model directory",
    )

    # Flags
    parser.add_argument(
        "--no_kalman", action="store_true", help="Disable Kalman filtering"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Display prediction plots",
    )
    parser.add_argument(
        "--no_visualize",
        action="store_false",
        dest="visualize",
        help="Disable visualization",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        default=True,
        help="Save predictions to file",
    )
    parser.add_argument(
        "--error_analysis", action="store_true", help="Show error analysis plots"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Run prediction
    y_pred, y_true = predict_angles(
        root_dir=args.root_dir,
        label=args.label,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        apply_kalman=not args.no_kalman,
        visualize=args.visualize,
        save_predictions=args.save_predictions,
        verbose=args.verbose,
    )

    # Error analysis
    if args.error_analysis:
        finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        plot_error_analysis(y_pred, y_true, finger_names)
