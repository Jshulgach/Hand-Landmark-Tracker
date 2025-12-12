import argparse
import matplotlib.pyplot as plt
import numpy as np
from handtrack.processing import run_pipeline


ANGLE_NAMES = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

def plot_joint_angle_predictions(time_vector, ground_truth, predictions):
    plt.figure(figsize=(12, 8))
    for i in range(5):
        plt.subplot(5, 1, i + 1)
        plt.plot(time_vector, np.degrees(ground_truth[:, i]), label='Ground Truth', color='blue')
        plt.plot(time_vector, np.degrees(predictions[:, i]), label='Prediction', color='orange')
        plt.ylabel(ANGLE_NAMES[i])
        plt.legend(loc='upper right')
        plt.grid(True)
    plt.xlabel('Time (s)')
    plt.suptitle('Predicted vs Ground Truth Finger Joint Angles')
    plt.tight_layout()
    plt.show()


def main(root_dir, label, train_model):

    emg_features, joint_labels, predicted_angles, metrics = run_pipeline(
        root_dir=root_dir,
        label=label,
        window_ms=250,
        step_ms=50,
        train_model=train_model,
        verbose=True,
    )

    # Create time vector assuming 50ms step size (~20Hz sampling rate)
    time_vector = np.arange(len(joint_labels)) * 0.05

    print("Evaluation Metrics:", metrics)
    plot_joint_angle_predictions(time_vector, joint_labels, predicted_angles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EMG angle prediction pipeline on a session")
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of data')
    parser.add_argument('--label', type=str, required=True, help='Session label')
    parser.add_argument('--train', action='store_true', help='If set, train model from scratch')
    args = parser.parse_args()

    main(args.root_dir, args.label, args.train)
