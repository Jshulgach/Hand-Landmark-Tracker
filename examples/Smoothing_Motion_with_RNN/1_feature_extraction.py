import os
import cv2
from tqdm import tqdm
import numpy as np
import mediapipe as mp
from kalman_filter import Kalman3D


def extract_landmarks(video_path, visualize=False):
    """
    Extracts 3D hand landmarks from a video file using MediaPipe Hands.
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    raw_landmarks = []

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
                if visualize:
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            else:
                landmarks = np.zeros((21, 3))  # 21 landmarks, padded if hand not found

            raw_landmarks.append(landmarks)

            if visualize:
                cv2.imshow("Hand Tracking", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            pbar.update(1)

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

    return np.array(raw_landmarks)  # shape: [T, 21, 3]


def apply_kalman_filter(landmarks, video_path=None, visualize=False):
    filters = [Kalman3D() for _ in range(21)]
    smoothed = []

    cap = None
    if visualize and video_path:
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "Cannot open video file for visualization"

    n_landmarks = len(landmarks)
    with tqdm(total=n_landmarks, desc="Processing frames") as pbar:
        for idx, frame_landmarks in enumerate(landmarks):
            filtered_frame = [filters[i].update(frame_landmarks[i]) for i in range(21)]
            smoothed.append(filtered_frame)

            if visualize and cap:
                ret, frame = cap.read()
                if not ret:
                    break

                # Draw filtered landmarks (white) and raw landmarks (green)
                for i in range(21):
                    x_raw, y_raw = int(frame_landmarks[i][0] * frame.shape[1]), int(frame_landmarks[i][1] * frame.shape[0])
                    x_filt, y_filt = int(filtered_frame[i][0] * frame.shape[1]), int(filtered_frame[i][1] * frame.shape[0])

                    cv2.circle(frame, (x_raw, y_raw), 4, (0, 255, 0), -1)     # Green = raw
                    cv2.circle(frame, (x_filt, y_filt), 4, (255, 255, 255), -1) # White = filtered

                cv2.imshow("Kalman Filtered Landmarks", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            pbar.update(1)

    if cap:
        cap.release()
        cv2.destroyAllWindows()

    return np.array(smoothed)


def create_sequences(data, window_size=10, step=1):
    sequences = []
    total = (len(data) - window_size + 1) // step

    for i in tqdm(range(0, total * step, step), desc="Creating sequences", total=total):
        seq = data[i:i + window_size]
        sequences.append(seq)
    return np.array(sequences)  # shape: [N, window, 21, 3]


def save_dataset(X, Y, save_dir):
    np.save(os.path.join(save_dir, "X_raw.npy"), X)
    np.save(os.path.join(save_dir, "Y_smooth.npy"), Y)
    print(f"Saved raw input to {save_dir}/X_raw.npy")
    print(f"Saved smoothed target to {save_dir}/Y_smooth.npy")




if __name__ == "__main__":

    # ---------- CONFIG ----------
    VIDEO_PATH = r"G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\Open_Ephys\Jonathan\2025_05_07\media\HandDynamic.mp4"
    SEQUENCE_LENGTH = 10  # Length of each sequence
    STEP_SIZE = 1  # Step size for creating sequences
    SAVE_DIR = "data"  # Directory to save the dataset

    print("Extracting landmarks...")
    raw = extract_landmarks(VIDEO_PATH, visualize=False)
    print(f"Shape of raw landmarks: {raw.shape} (T, 21, 3)")

    print("Applying Kalman filtering...")
    smooth = apply_kalman_filter(raw, VIDEO_PATH, visualize=False)
    print(f"Shape of smoothed landmarks: {smooth.shape} (T, 21, 3)")

    print("Creating sequences...")
    X = create_sequences(raw, window_size=SEQUENCE_LENGTH, step=STEP_SIZE)
    Y = create_sequences(smooth, window_size=SEQUENCE_LENGTH, step=STEP_SIZE)
    print(f"Created dataset: {len(X)} samples of {SEQUENCE_LENGTH} frames each")

    print("💾 Saving dataset...")
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_dataset(X, Y, SAVE_DIR)

