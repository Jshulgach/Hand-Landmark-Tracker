"""
Extract facial landmarks from video files using MediaPipe Face Mesh.

Applies optional Kalman filtering to smooth landmark trajectories and saves
the results to NPZ format for offline analysis.

Author: Jonathan Shulgach
Date Created: June 10th, 2025
Updated: January 6th, 2026

"""

import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm
import mediapipe as mp
import sys

# Add parent directory to path if handtrack is not installed
src_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
sys.path.insert(0, os.path.abspath(src_path))

try:
    from handtrack.processing import Kalman3D
    from handtrack.io import check_video_path
except ImportError:
    # Fallback: inline Kalman3D if handtrack not available
    class Kalman3D:
        def __init__(self, dt=1/30, process_noise=1e-3, measurement_noise=1e-2):
            self.x = np.zeros((6, 1))
            self.F = np.eye(6)
            for i in range(3): self.F[i, i+3] = dt
            self.H = np.hstack((np.eye(3), np.zeros((3, 3))))
            self.P = np.eye(6)
            self.Q = np.eye(6) * process_noise
            self.R = np.eye(3) * measurement_noise

        def update(self, z):
            z = np.reshape(z, (3, 1))
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q
            y = z - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x += K @ y
            self.P = (np.eye(6) - K @ self.H) @ self.P
            return self.x[:3].flatten()

    def check_video_path(path):
        """Simple fallback for video path checking."""
        if path and os.path.isfile(path):
            return path
        return None

# MediaPipe Face Mesh setup
mp_face_mesh = mp.solutions.face_mesh

# Number of face mesh landmarks (without iris refinement)
NUM_FACE_LANDMARKS = 468


def extract_landmarks(video_path, visualize=False):
    """
    Extract raw facial landmarks from a video file.

    Args:
        video_path (str): Path to the video file.
        visualize (bool): Whether to display the video with landmarks.

    Returns:
        raw_landmarks (np.ndarray): Array of shape (N, 468, 3) with landmark positions.
        metadata (dict): Dictionary with sampling_rate and total_frames.
    """
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    raw_landmarks = []

    with tqdm(total=total_frames, desc="Extracting landmarks") as pbar:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
                    raw_landmarks.append(landmarks)

                    if visualize:
                        for lm in landmarks:
                            x = int(lm[0] * frame.shape[1])
                            y = int(lm[1] * frame.shape[0])
                            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                        cv2.putText(frame, f"Frame: {len(raw_landmarks)}/{total_frames}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, "Press ESC to quit", (10, frame.shape[0] - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                        cv2.imshow("Face Landmark Extraction", frame)

                        if cv2.waitKey(1) & 0xFF == 27:
                            break
            else:
                # No face detected - append zeros
                raw_landmarks.append(np.zeros((NUM_FACE_LANDMARKS, 3)))

            pbar.update(1)

    face_mesh.close()
    cap.release()
    if visualize:
        cv2.destroyAllWindows()

    metadata = {
        'sampling_rate': fps,
        'total_frames': total_frames,
        'num_landmarks': NUM_FACE_LANDMARKS,
    }

    return np.array(raw_landmarks), metadata


def apply_kalman_filter(landmarks, video_path=None, visualize=False, save_video=False):
    """
    Apply Kalman filtering to smooth landmark trajectories.

    Args:
        landmarks (np.ndarray): Raw landmarks of shape (N, 468, 3).
        video_path (str): Optional path to video for visualization.
        visualize (bool): Whether to display the smoothed landmarks.
        save_video (bool): Whether to save a video with smoothed landmarks.

    Returns:
        smoothed (np.ndarray): Smoothed landmarks of shape (N, 468, 3).
    """
    num_landmarks = landmarks.shape[1]
    filters = [Kalman3D(process_noise=1e-3, measurement_noise=1e-4) for _ in range(num_landmarks)]
    smoothed = []

    cap = None
    out = None

    if visualize and video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Cannot open video file for visualization: {video_path}")
            cap = None
        else:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30

            if save_video:
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                out_path = os.path.join(os.path.dirname(video_path), f"{video_name}_face_labeled.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(out_path, fourcc, video_fps, (frame_width, frame_height))
                print(f"Saving labeled video to: {out_path}")

    with tqdm(total=landmarks.shape[0], desc="Applying Kalman filter") as pbar:
        for idx, frame_landmarks in enumerate(landmarks):
            filtered_frame = np.array([filters[i].update(frame_landmarks[i]) for i in range(num_landmarks)])
            smoothed.append(filtered_frame)

            if cap:
                ret, frame = cap.read()
                if not ret:
                    break

                # Draw smoothed landmarks
                for i in range(num_landmarks):
                    x_filt = int(filtered_frame[i][0] * frame.shape[1])
                    y_filt = int(filtered_frame[i][1] * frame.shape[0])
                    cv2.circle(frame, (x_filt, y_filt), 1, (255, 255, 255), -1)

                cv2.putText(frame, f"Frame: {idx + 1}/{landmarks.shape[0]}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Kalman Smoothed", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "Press ESC to quit", (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                if visualize:
                    cv2.imshow("Kalman Filtered Landmarks", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                if out:
                    out.write(frame)

            pbar.update(1)

    if cap:
        cap.release()
    if out:
        out.release()
    if visualize:
        cv2.destroyAllWindows()

    return np.array(smoothed)


def save_landmarks(landmarks, file_path, metadata):
    """
    Save landmarks to NPZ file.

    Args:
        landmarks (np.ndarray): Landmark data of shape (N, 468, 3).
        file_path (str): Output file path.
        metadata (dict): Metadata dictionary.
    """
    total_frames = len(landmarks)
    time_vector = np.arange(total_frames) / metadata['sampling_rate']

    np.savez(file_path,
             landmarks=landmarks,
             num_landmarks=metadata['num_landmarks'],
             sampling_rate=metadata['sampling_rate'],
             time_vector=time_vector)

    print(f"Saved landmarks to: {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract facial landmarks from video files with optional Kalman filtering."
    )
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to the video file containing faces.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save landmarks. Defaults to 'landmarks/' next to video.")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the landmark extraction process.")
    parser.add_argument("--save_video", action="store_true",
                        help="Save a video with landmarks overlaid.")
    parser.add_argument("--no_kalman", action="store_true",
                        help="Disable Kalman filtering (save raw landmarks only).")
    args = parser.parse_args()

    # Validate video path
    video_path = args.video_path.strip()
    if not os.path.isfile(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.dirname(video_path)

    # Determine output directory
    if args.output_dir:
        landmarks_dir = args.output_dir
    else:
        landmarks_dir = os.path.normpath(os.path.join(video_dir, '..', 'landmarks'))
    os.makedirs(landmarks_dir, exist_ok=True)

    # Step 1: Extract raw landmarks
    print(f"Processing video: {video_path}")
    raw_landmarks, metadata = extract_landmarks(video_path, visualize=args.visualize)
    print(f"Extracted {len(raw_landmarks)} frames with {metadata['num_landmarks']} landmarks each")

    # Step 2: Apply Kalman filtering (unless disabled)
    if args.no_kalman:
        final_landmarks = raw_landmarks
        suffix = "_raw_landmarks.npz"
    else:
        print("Applying Kalman filtering...")
        final_landmarks = apply_kalman_filter(
            raw_landmarks,
            video_path=video_path,
            visualize=args.visualize,
            save_video=args.save_video
        )
        suffix = "_landmarks.npz"

    # Step 3: Save landmarks
    save_path = os.path.join(landmarks_dir, f"{video_name}{suffix}")
    save_landmarks(final_landmarks, save_path, metadata)

    print("Done!")
