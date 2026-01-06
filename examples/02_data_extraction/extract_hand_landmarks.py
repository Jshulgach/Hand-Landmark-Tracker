"""
Extract hand landmarks from video files using MediaPipe Hands.

Applies optional Kalman filtering to smooth landmark trajectories and saves
the results to NPZ format for offline analysis.

Author: Jonathan Shulgach
Updated: January 6th, 2026

"""

import os
import argparse
import sys
import numpy as np

# Add parent directory to path if handtrack is not installed
src_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
sys.path.insert(0, os.path.abspath(src_path))

try:
    from handtrack.io import check_video_path
    from handtrack.tracker import HandTracker
except ImportError:
    print("Error: handtrack package not found. Please install it or run from the project root.")
    sys.exit(1)


if __name__ == "__main__":

    # ---------- Allow user to define video path ----------
    parser = argparse.ArgumentParser(
        description="Extract hand landmarks from video files with Kalman filtering."
    )
    parser.add_argument("--video_path", type=str, default="",
                        help="Path to the video file containing hand landmarks.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save landmarks. Defaults to 'landmarks/' next to video.")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the landmark extraction process.")
    parser.add_argument("--save_video", action="store_true",
                        help="Save the video with landmarks drawn on it.")
    parser.add_argument("--no_kalman", action="store_true",
                        help="Disable Kalman filtering (save raw landmarks only).")
    args = parser.parse_args()

    video_path = check_video_path(args.video_path.strip())
    if video_path is None:
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    video_name = os.path.basename(video_path).split('.')[0]
    video_dir = os.path.dirname(video_path)

    # Determine output directory
    if args.output_dir:
        landmarks_dir = args.output_dir
    else:
        # Assuming video is contained in a separate 'media' folder
        landmarks_dir = os.path.normpath(os.path.join(video_dir, '..', 'landmarks'))
    os.makedirs(landmarks_dir, exist_ok=True)

    suffix = "_raw_landmarks.npz" if args.no_kalman else "_landmarks.npz"
    save_path = os.path.join(landmarks_dir, f"{video_name}{suffix}")

    # Step 1: Create an instance of the HandTracker
    apply_kalman = not args.no_kalman
    tracker = HandTracker(source=video_path, apply_kalman=apply_kalman)

    # Step 2: Extract landmarks from the video
    print(f"Processing video: {video_path}")
    print(f"Kalman filtering: {'enabled' if apply_kalman else 'disabled'}")
    landmarks, metadata = tracker.extract_landmarks(
        visualize=args.visualize,
        save_video=args.save_video
    )

    if landmarks is None:
        print("Error: No landmarks extracted.")
        sys.exit(1)

    # Step 3: Save the landmarks and metadata
    np.savez(save_path,
             landmarks=landmarks,
             labels=metadata['landmark_labels'],
             sampling_rate=metadata['sampling_rate'],
             time_vector=metadata['time_vector'],
             )

    print(f"Extracted {len(landmarks)} frames with 21 landmarks each")
    print(f"Saved landmarks to: {save_path}")
    print("Done!")
