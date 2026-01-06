import os
import cv2
import csv
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import time
from handtrack.processing import Kalman3D, compute_all_joint_angles


class HandTracker:
    """
    Tracks hand landmarks using MediaPipe from either a webcam or video file

    Supports Kalman filtering for smoothing landmark trajectories and joint angle computation.
    """
    def __init__(self, source=0, img_size=(1080, 720), video_fps=30, max_hands=1, confidence=0.8, apply_kalman=True,
                 save_angles=False, out_path='angles.csv', verbose=False):
        """
        Initializes the HandTracker.

        Args:
            source (int or str): Video source, either an integer for webcam or a string for video file path.
            img_size (tuple): Size of the output image (width, height).
            video_fps (int): Frames per second for video capture.
            max_hands (int): Maximum number of hands to detect.
            confidence (float): Minimum confidence threshold for hand detection.
            apply_kalman (bool): Whether to apply Kalman filtering to the landmarks.
            save_angles (bool): Whether to save joint angles to a CSV file.
            out_path (str): Path to save the joint angles CSV file.
            verbose (bool): If True, prints additional information during processing.

        """
        self.source = source
        self.img_size = img_size
        self.apply_kalman = apply_kalman
        self.save_angles = save_angles
        self.out_path = out_path
        self.frame_count = 0
        self.mode = 'video' if isinstance(source, str) else 'realtime'
        self.verbose = verbose

        # Initialize video capture, and set it up
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_size[1])
        #self.cap.set(cv2.CAP_PROP_FPS, video_fps)  # Set FPS to 30 for consistency
        if self.verbose:
            print(f"Using video source: {source} (mode: {self.mode})")

        # Initialize MediaPipe Hands
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=confidence,
            min_tracking_confidence=0.8
        )

        # Initialize filters and state
        self.kalman_filters = [Kalman3D(process_noise=1e-3, measurement_noise=1e-4) for _ in range(21)]
        self.joint_log = []  # to store angle data
        self.last_angles = None  # For histogram
        self.landmarks_filtered = None  # Store filtered landmarks for visualization

    def extract_landmarks(self, visualize=False, save_video=False, flip_frame=False):
        """
        Extracts hand landmarks from the video source.

        Args:
            visualize (bool): If True, displays the video with landmarks in real-time.
            save_video (bool): If True, saves the video with landmarks overlayed.

        Returns:
            landmarks_np (np.ndarray): Array of shape (N, 21, 3) containing the 3D landmarks.
            metadata (dict): Metadata including sampling rate, total frames, landmark labels, and time vector.

        """
        if self.mode == 'realtime':
            self.run()
            return None, None

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        pbar = tqdm(total=total_frames if total_frames > 0 else None,
                    desc="Processing frames", unit="frame")

        # Optional video writer
        if save_video:
            base_path = os.path.splitext(os.path.basename(self.source))[0]
            out_path = f"{base_path}_labeled.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))
            print(f"[INFO] Saving labeled video to {out_path}")
        else:
            out_writer = None

        if visualize:
            print(F"visualize: {visualize}, save_video: {save_video}")

        raw_landmarks = []
        smooth_landmarks = []
        frame_idx = 0
        while self.cap.isOpened():
            frame = self.get_image(flip_frame=flip_frame)
            if frame is None:
                print("No more frames to read.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            if results and results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
            else:
                landmark_array = np.zeros((21, 3))

            raw_landmarks.append(landmark_array)

            # Apply Kalman filtering if enabled
            if self.apply_kalman:
                filtered_frame = [self.kalman_filters[i].update(landmark_array[i]) for i in range(21)]
                smooth_landmarks.append(filtered_frame)
            else:
                smooth_landmarks.append(landmark_array)

            # Compute joint angles
            #angles = compute_all_joint_angles(self._wrap_to_hand(self.landmarks_filtered))
            #print(f"shape of angles: {len(angles)}")
            #self.last_angles = angles

            if visualize or save_video:
                # Draw landmarks on the frame
                if results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, results.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS
                    )

                if self.apply_kalman:
                    for x, y, _ in filtered_frame:
                        cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 4, (255, 255, 255), -1)

                # Show the frame index on the frame
                cv2.putText(frame, f"Frame: {frame_idx + 1}/{total_frames}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if visualize:
                    cv2.imshow("Raw Hand Landmarks", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

            if save_video and out_writer is not None:
                out_writer.write(frame)

            frame_idx += 1
            pbar.update(1)  # NEW: advance progress

        self.cap.release()
        pbar.close()
        if out_writer:
            out_writer.release()
        if visualize:
            cv2.destroyAllWindows()

        #landmarks_np = np.array(raw_landmarks)
        landmarks_np = smooth_landmarks
        metadata = {
            'sampling_rate': fps,
            'total_frames': total_frames,
            'landmark_labels': [lm.name for lm in mp.solutions.hands.HandLandmark],
            'time_vector': np.arange(total_frames) / fps
        }

        return landmarks_np, metadata

    def reset_filters(self):
        """
        Reset the Kalman filters for all 21 landmarks. This is useful if you want to clear the state of the filters

        Args:
            None

        Returns:
            None

        """
        self.kalman_filters = [Kalman3D(process_noise=1e-3, measurement_noise=1e-4) for _ in range(21)]

    def get_image(self, flip_frame=False):
        success, frame = self.cap.read()
        if not success:
            return None
        return cv2.flip(frame, 1) if flip_frame else frame

    def run(self):
        """
        Run real-time hand tracking from webcam

        Args:
            None

        Returns:
            None

        """
        print("Running live hand tracking (ESC to quit)...")
        frame_idx = 0
        while self.cap.isOpened():
            frame = self.get_image(flip_frame=True)
            if frame is None:
                break

            landmarks = self.detect_hands(frame)
            if landmarks:
                landmarks_raw = np.array([[lm.x, lm.y, lm.z] for lm in landmarks[0].landmark])
                self.landmarks_filtered = np.array([
                    self.kalman_filters[i].update(landmarks_raw[i]) for i in range(21)
                ])

                # Compute joint angles
                #angles = compute_all_joint_angles(self._wrap_to_hand(self.landmarks_filtered))
                #print(f"shape of angles: {len(angles)}")
                #self.last_angles = angles

                #if self.verbose:
                #    print(angles)
                #if self.save_angles:
                #    self.joint_log.append([time.time()] + list(angles.values()))

            # Show frame index on frame
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show quit instructions
            cv2.putText(frame, "Press ESC to quit", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            frame_idx += 1

            # Display OpenCV image with 2D landmarks
            self._visualize(frame, landmarks)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()
        if self.save_angles:
            self.export_joint_log()

    def _visualize(self, frame, landmarks):
        """
        Helper function to visualize landmarks and angles on the frame.

        Args:
            frame (np.ndarray): The current video frame.
            landmarks (list): List of detected hand landmarks.

        Returns:
            None

        """
        if self.landmarks_filtered is not None and landmarks is not None:
            for hand in landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)

            for x, y, _ in self.landmarks_filtered:
                cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 4, (255, 255, 255), -1)

        if self.last_angles:
            print("drawing histogram")
            self._draw_joint_angle_histogram(frame, self.last_angles)
        cv2.imshow("Hand Tracker", frame)

    def _wrap_to_hand(self, coords):
        """
        Wraps the filtered coordinates into a dummy hand object for compatibility with joint angle computation.

        Args:
            coords (np.ndarray): Filtered coordinates of shape (21, 3).

        Returns:
            DummyHand: A dummy hand object containing the landmarks.

        """
        class DummyLandmark:
            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z

        class DummyHand:
            def __init__(self, coords):
                self.landmark = [DummyLandmark(x, y, z) for x, y, z in coords]

        return DummyHand(coords)

    def _draw_joint_angle_histogram(self, frame, angles, base_x=10, base_y=30, max_bar_width=150, bar_height=15, spacing=5):
        """
        Draws a histogram of joint angles on the given frame.
        This function visualizes the angles of each joint as horizontal bars.

        Args:
            frame (np.ndarray): The current video frame.
            angles (dict): Dictionary of joint angles where keys are joint names and values are angles in degrees.
            base_x (int): X-coordinate for the base of the histogram.
            base_y (int): Y-coordinate for the base of the histogram.
            max_bar_width (int): Maximum width of the bars representing angles.
            bar_height (int): Height of each bar in the histogram.
            spacing (int): Spacing between each bar in the histogram.

        Returns
            None

        """
        for i, (joint_name, angle) in enumerate(angles.items()):
            y = base_y + i * (bar_height + spacing)
            bar_len = int((angle / 180) * max_bar_width)

            # Draw background bar
            cv2.rectangle(frame, (base_x, y), (base_x + max_bar_width, y + bar_height), (50, 50, 50), -1)
            # Draw active bar
            cv2.rectangle(frame, (base_x, y), (base_x + bar_len, y + bar_height), (0, 255, 0), -1)
            # Draw text
            cv2.putText(frame, f"{joint_name} ({int(angle)} deg)",
                        (base_x + max_bar_width + 10, y + bar_height - 2),
                        cv2.FONT_HERSHEY_PLAIN, 1, (200, 255, 200), 1)

    def detect_hands(self, image):
        """
        Detects hands in the given image using MediaPipe Hands.

        Args:
            image (np.ndarray): Input image in BGR format.

        Returns:
            list: List of detected hand landmarks, or None if no hands are detected.

        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        return results.multi_hand_landmarks if results.multi_hand_landmarks else None

    def _process_frame(self, frame):
        """
        Process a single frame: detect hands, apply Kalman filter, compute angles.

        Args:
            frame (np.ndarray): Input image frame.

        Returns:
            tuple: landmarks, filtered_landmarks, angles, results

        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        landmarks = results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None

        if landmarks:
            landmarks_raw = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        else:
            landmarks_raw = np.zeros((21, 3))

        if self.apply_kalman:
            self.landmarks_filtered = np.array([
                self.kalman_filters[i].update(landmarks_raw[i]) for i in range(21)
            ])
        else:
            self.landmarks_filtered = landmarks_raw

        angles = compute_all_joint_angles(self._wrap_to_hand(self.landmarks_filtered))
        self.last_angles = angles

        return landmarks, self.landmarks_filtered, angles, results

    def export_joint_log(self):
        """
        Exports the joint angle log to a CSV file.

        Returns:
            None

        """
        header = ["timestamp"] + [
            "Thumb_MCP", "Thumb_IP",
            "Index_MCP", "Index_PIP", "Index_DIP",
            "Middle_MCP", "Middle_PIP", "Middle_DIP",
            "Ring_MCP", "Ring_PIP", "Ring_DIP",
            "Pinky_MCP", "Pinky_PIP", "Pinky_DIP"
        ]
        with open(self.out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self.joint_log)
        print(f"Saved joint angle log to {self.out_path}")

    def get_landmarks(self):
        """
        Returns the last detected landmarks. If Kalman filtering was applied, returns the filtered landmarks.
        If no landmarks were detected, returns None.

        Returns:
            np.ndarray or None: Filtered landmarks of shape (N, 21, 3) or None if no landmarks were detected.

        """
        return self.landmarks_filtered if self.landmarks_filtered is not None else None

    def get_angles(self):
        """
        Returns the last computed joint angles.

        Returns:
            dict or None: Dictionary of joint angles where keys are joint names and values are angles in degrees.

        """
        return self.last_angles if self.last_angles is not None else None
