import os
import argparse
import numpy as np
from handtrack.io import check_video_path
from handtrack.tracker import HandTracker


if __name__ == "__main__":

    # ---------- Allow user to define video path ----------
    parser = argparse.ArgumentParser(description="Process video file to extract and smooth hand landmarks using Kalman filtering.")
    parser.add_argument("--video_path", type=str, default="", help="Path to the video file containing hand landmarks.")
    parser.add_argument("--visualize", action="store_true", help="Visualize the landmark extraction process.")
    parser.add_argument("--save_video", action="store_true", help="Save the video with landmarks drawn on it.")
    args = parser.parse_args()

    video_path = check_video_path(args.video_path.strip())
    if video_path is not None:
        video_name = os.path.basename(video_path).split('.')[0]
        video_dir = os.path.dirname(video_path)

        # Assuming video is contained in a separate 'media' folder
        landmarks_dir = os.path.normpath(os.path.join(video_dir, '..', 'landmarks'))
        os.makedirs(landmarks_dir, exist_ok=True)

        save_path = os.path.join(landmarks_dir, f"{video_name}_landmarks.npz")
        #save_path = os.path.splitext(video_path)[0] + "_landmarks.npz"

        # Step 1: Create an instance of the HandTracker and enable Kalman filtering for landmarks
        #tracker = HandTracker(source=0, apply_kalman=True)
        # tracker.run()
        tracker = HandTracker(source=video_path, apply_kalman=True)

        # Step 2: Extract landmarks from the video
        print("Extracting landmarks...")
        landmarks, metadata = tracker.extract_landmarks(visualize=args.visualize, save_video=args.save_video)

        #Step 3: Save the smoothed landmarks and metadata
        np.savez(save_path,
                 landmarks=landmarks,
                 labels=metadata['landmark_labels'],
                 sampling_rate=metadata['sampling_rate'],
                 time_vector=metadata['time_vector'],
                 )

        print(f"Saved smoothed landmarks to: {save_path}")
