""" 
A script that extracts the facial landmakrs using the Mediapipe face solution and 
implements a Klaman filter to smoothen out the positions.

Author: Jonathan Shulgach
Date Created: June 10th, 2025

"""

import os
import numpy as np
import cv2
from tqdm import tqdm
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh


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


def apply_kalman_filter(landmarks, video_path=None, visualize=False, save_video=False):
    filters = [Kalman3D() for _ in range(landmarks.shape[1])]
    smoothed = []

    cap = None
    out = None
    if visualize and video_path:
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        assert cap.isOpened(), "Cannot open video file for visualization"

        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_path = os.path.join(os.path.dirname(video_path), f"{os.path.basename(video_path).split('.')[0]}_labeled.mp4")
            out = cv2.VideoWriter(out_path, fourcc, video_fps, (frame_width, frame_height))

    with tqdm(total=landmarks.shape[0], desc="Processing frames") as pbar:
        for idx, frame_landmarks in enumerate(landmarks):
            filtered_frame = [filters[i].update(frame_landmarks[i]) for i in range(landmarks.shape[1])]
            smoothed.append(filtered_frame)

            if visualize and cap:
                ret, frame = cap.read()
                if not ret:
                    break

                if visualize:
                    for i in range(landmarks.shape[1]):
                        x_filt, y_filt = int(filtered_frame[i][0] * frame.shape[1]), int(filtered_frame[i][1] * frame.shape[0])
                        cv2.circle(frame, (x_filt, y_filt), 2, (255, 255, 255), -1)
                    cv2.imshow("Filtered Landmark Tracking", frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

                if save_video:
                    out.write(frame)

            pbar.update(1)

    if cap:
        cap.release()
        if save_video and out:
            out.release()
        cv2.destroyAllWindows()

    return np.array(smoothed)


def extract_landmarks(video_path, visualize=False):
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    raw_landmarks = []
    landmarks = None
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
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
                        cv2.imshow("Landmark Tracking", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            pbar.update(1)

    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()

    metadata = {
        'sampling_rate': fps,
        'total_frames': total_frames,
        #'landmark_labels': [''] * landmarks.shape[1] if landmarks else 1 # Placeholder for landmark labels
    }

    return np.array(raw_landmarks), metadata


def save_landmarks(data, file_path, metadata):
    total_frames = len(data)
    time_vector = np.arange(total_frames) / metadata['sampling_rate']
    np.savez(file_path,
             landmarks=data,
             landmark_labels=metadata['landmark_labels'],
             sampling_rate=metadata['sampling_rate'],
             time_vector=time_vector)
    print(f"Saved smoothed landmarks to: {file_path}")


if __name__ == "__main__":
    root_path = r"G:/Shared drives/NML_shared/DataShare/HDEMG_Face/Data/Jack/060525_Pilot/raw/"
    # file_path = os.path.join(root_path, "Angry-20250605_123811.poly5")
    # file_path = os.path.join(root_path, "Disgust-20250605_124503.poly5")
    file_path = os.path.join(root_path, "Fear.mp4")
    # file_path = os.path.join(root_path, "Frown-20250605_123513.poly5")
    # file_path = os.path.join(root_path, "Grin-20250605_123132.poly5")
    # file_path = os.path.join(root_path, "Rest-20250605_122936.poly5")
    # file_path = os.path.join(root_path, "Silly-20250605_124651.poly5")
    # file_path = os.path.join(root_path, "Surprise-20250605_123641.poly5")

    SAVE_VIDEO = False

    VIDEO_PATH = file_path
    print("Extracting landmarks...")
    raw, metadata = extract_landmarks(VIDEO_PATH, visualize=False)
    print(f"Shape of raw landmarks: {raw.shape} (T, 468, 3)")

    print("Applying Kalman filtering...")
    smooth = apply_kalman_filter(raw, VIDEO_PATH, visualize=True, save_video=SAVE_VIDEO)
    print(f"Shape of smoothed landmarks: {smooth.shape} (T, 468, 3)")

    # Save example (uncomment if needed)
    output_path = os.path.join(os.path.dirname(VIDEO_PATH), f"{os.path.basename(VIDEO_PATH).split('.')[0]}_smoothed_landmarks.npz")
    save_landmarks(smooth, output_path, metadata)
