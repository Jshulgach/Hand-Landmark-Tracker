import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm


class FaceTracker:
    """
    Tracks face landmarks using MediaPipe Face Mesh from either a webcam or video file.

    Supports real-time visualization with optional Kalman filtering for smoothing.
    """

    # Key facial landmark indices for visualization
    # These are commonly used landmarks for face analysis
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    
    LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    
    LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
    RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

    def __init__(self, source=0, img_size=(1080, 720), max_faces=1, confidence=0.5,
                 refine_landmarks=True, verbose=False):
        """
        Initializes the FaceTracker.

        Args:
            source (int or str): Video source, either an integer for webcam or a string for video file path.
            img_size (tuple): Size of the output image (width, height).
            max_faces (int): Maximum number of faces to detect.
            confidence (float): Minimum confidence threshold for face detection.
            refine_landmarks (bool): Whether to refine landmarks around eyes and lips.
            verbose (bool): If True, prints additional information during processing.
        """
        self.source = source
        self.img_size = img_size
        self.frame_count = 0
        self.mode = 'video' if isinstance(source, str) else 'realtime'
        self.verbose = verbose

        # Initialize video capture
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_size[1])
        if self.verbose:
            print(f"Using video source: {source} (mode: {self.mode})")

        # Initialize MediaPipe Face Mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=confidence,
            min_tracking_confidence=0.5
        )

        # Drawing specs
        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
            thickness=1, circle_radius=1, color=(0, 255, 0)
        )
        self.connection_spec = mp.solutions.drawing_utils.DrawingSpec(
            thickness=1, color=(0, 200, 0)
        )

    def get_image(self, flip_frame=False):
        """Read a frame from the video source."""
        success, frame = self.cap.read()
        if not success:
            return None
        return cv2.flip(frame, 1) if flip_frame else frame

    def detect_faces(self, frame):
        """
        Detect face landmarks in a frame.

        Args:
            frame (np.ndarray): BGR image frame.

        Returns:
            results: MediaPipe face mesh results object.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        return results

    def run(self):
        """
        Run real-time face tracking from webcam.
        """
        print("Running live face tracking (ESC to quit)...")
        frame_idx = 0

        while self.cap.isOpened():
            frame = self.get_image(flip_frame=True)
            if frame is None:
                break

            results = self.detect_faces(frame)

            # Draw landmarks
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw the full mesh
                    mp.solutions.drawing_utils.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    # Draw contours (face oval, eyes, eyebrows, lips)
                    mp.solutions.drawing_utils.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
                    )
                    # Draw iris if refined landmarks are enabled
                    mp.solutions.drawing_utils.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style()
                    )

                    if self.verbose:
                        # Print number of landmarks detected
                        num_landmarks = len(face_landmarks.landmark)
                        print(f"Frame {frame_idx}: Detected {num_landmarks} landmarks")

            # Show frame info
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            num_faces = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0
            cv2.putText(frame, f"Faces: {num_faces}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show quit instructions
            cv2.putText(frame, "Press ESC to quit", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            frame_idx += 1

            cv2.imshow("Face Tracker", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def extract_landmarks(self, visualize=False, save_video=False, flip_frame=False):
        """
        Extracts face landmarks from the video source.

        Args:
            visualize (bool): If True, displays the video with landmarks in real-time.
            save_video (bool): If True, saves the video with landmarks overlayed.
            flip_frame (bool): If True, flips the frame horizontally.

        Returns:
            landmarks_np (list): List of landmark arrays, one per frame.
            metadata (dict): Metadata including sampling rate, total frames, etc.
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
            out_path = f"{base_path}_face_labeled.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))
            print(f"[INFO] Saving labeled video to {out_path}")
        else:
            out_writer = None

        all_landmarks = []
        frame_idx = 0

        while self.cap.isOpened():
            frame = self.get_image(flip_frame=flip_frame)
            if frame is None:
                break

            results = self.detect_faces(frame)

            if results.multi_face_landmarks:
                # Get first face landmarks as numpy array (468 landmarks x 3 coords)
                face_landmarks = results.multi_face_landmarks[0]
                landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
            else:
                # No face detected - use zeros (468 landmarks for standard, 478 with refinement)
                landmark_array = np.zeros((478, 3))

            all_landmarks.append(landmark_array)

            if visualize or save_video:
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
                        )

                cv2.putText(frame, f"Frame: {frame_idx + 1}/{total_frames}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if visualize:
                    cv2.imshow("Face Landmarks", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                if save_video and out_writer is not None:
                    out_writer.write(frame)

            frame_idx += 1
            pbar.update(1)

        self.cap.release()
        pbar.close()
        if out_writer:
            out_writer.release()
        if visualize:
            cv2.destroyAllWindows()

        metadata = {
            'sampling_rate': fps,
            'total_frames': total_frames,
            'num_landmarks': 478,  # 468 base + 10 iris landmarks with refinement
            'time_vector': np.arange(len(all_landmarks)) / fps
        }

        return all_landmarks, metadata

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
