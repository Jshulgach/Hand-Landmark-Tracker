"""
Stereo (Multi-Camera) Tracker module.
"""

import cv2
import numpy as np
import mediapipe as mp
import os

class MultiCameraTracker:
    """Tracks hands across multiple calibrated cameras."""
    
    def __init__(self, 
                 camera_ids, 
                 calibration_file,
                 width=640,
                 height=480,
                 num_landmarks=21,
                 max_hands=2,
                 triangulation_method='simple_average',
                 min_cameras=2,
                 match_threshold=0.1):
        
        self.camera_ids = camera_ids
        self.num_cameras = len(camera_ids)
        self.calibration_file = calibration_file
        self.width = width
        self.height = height
        self.num_landmarks = num_landmarks
        self.max_hands = max_hands
        self.triangulation_method = triangulation_method
        self.min_cameras = min_cameras
        self.match_threshold = match_threshold
        
        # Camera captures
        self.captures = []
        
        # MediaPipe Hands instances (one per camera)
        self.hands_detectors = []
        
        # Calibration data
        self.camera_matrices = []
        self.dist_coeffs = []
        self.R_matrices = []
        self.T_vectors = []
        
        # Projection matrices for triangulation
        self.projection_matrices = []
        
        # Load calibration
        self.load_calibration()
        
    def load_calibration(self):
        """Load camera calibration data."""
        if not os.path.exists(self.calibration_file):
            raise FileNotFoundError(f"Calibration file not found: {self.calibration_file}")

        try:
            calib_data = np.load(self.calibration_file, allow_pickle=True)
            
            for idx in range(self.num_cameras):
                self.camera_matrices.append(calib_data[f'camera_matrix_{idx}'])
                self.dist_coeffs.append(calib_data[f'dist_coeffs_{idx}'])
                self.R_matrices.append(calib_data[f'R_{idx}'])
                self.T_vectors.append(calib_data[f'T_{idx}'])
            
            # Compute projection matrices
            self._compute_projection_matrices()
            
            print(f" Loaded calibration for {self.num_cameras} cameras")
            
        except Exception as e:
            print(f" Error loading calibration: {e}")
            raise
    
    def _compute_projection_matrices(self):
        """Compute projection matrices for triangulation."""
        for idx in range(self.num_cameras):
            # Create [R|T] matrix
            RT = np.hstack([self.R_matrices[idx], self.T_vectors[idx]])
            # Projection matrix P = K * [R|T]
            P = self.camera_matrices[idx] @ RT
            self.projection_matrices.append(P)
    
    def initialize_cameras(self):
        """Open all cameras and create MediaPipe detectors."""
        print(f"Initializing {self.num_cameras} cameras...")
        
        for cam_id in self.camera_ids:
            # Open camera
            cap = cv2.VideoCapture(cam_id)
            if not cap.isOpened():
                print(f"Failed to open camera {cam_id}")
                return False
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Check actual resolution
            actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            if actual_width == 0 or actual_height == 0:
                print(f" Camera {cam_id} reported 0x0 resolution. Retrying without MJPG force...", end=" ")
                cap.release()
                
                # Re-initialize
                if os.name == 'nt':
                    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(cam_id)
                
                if not cap.isOpened():
                    print("FAILED ")
                    return False
                
                # Set properties again
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                print("OK")

            self.captures.append(cap)
            
            # Create MediaPipe Hands detector
            hands = mp.solutions.hands.Hands(
                max_num_hands=self.max_hands,
                model_complexity=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.hands_detectors.append(hands)
            
            print(f" Camera {cam_id} initialized")
        
        return True

    def reconnect_camera(self, cam_idx):
        """Attempt to reconnect a failed camera."""
        cam_id = self.camera_ids[cam_idx]
        print(f"Attempting to reconnect Camera {cam_id}...", end=" ")
        
        # Release old capture
        if self.captures[cam_idx] is not None:
            self.captures[cam_idx].release()
        
         # Re-initialize
        if os.name == 'nt':
            cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(cam_id)
        
        if not cap.isOpened():
            print("FAILED")
            self.captures[cam_idx] = None
            return False
            
        # Set properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Check resolution
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        if w == 0 or h == 0:
            print("Retry (0x0)...", end=" ")
            cap.release()
            
            # Retry without specific backend
            cap = cv2.VideoCapture(cam_id)
            if not cap.isOpened():
                 print("FAILED")
                 self.captures[cam_idx] = None
                 return False
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        self.captures[cam_idx] = cap
        print("SUCCESS")
        return True
    
    def capture_frames(self):
        """Capture frames from all cameras."""
        frames = []
        for i, cap in enumerate(self.captures):
            frame = None
            if cap is not None and cap.isOpened():
                try:
                    ret, frame = cap.read()
                    if not ret or frame is None or frame.size == 0:
                        print(f"Warning: Failed to read from camera {self.camera_ids[i]}")
                        frame = None
                except Exception as e:
                    print(f"Error reading from camera {self.camera_ids[i]}: {e}")
                    frame = None
            
            # Auto-reconnection attempt logic
            if frame is None:
                if self.reconnect_camera(i):
                    try:
                        ret, frame = self.captures[i].read()
                        if not ret: frame = None
                    except:
                        frame = None
            
            frames.append(frame)
        return frames
    
    def detect_hands_all_cameras(self, frames):
        """
        Detect hands in all camera frames.
        Returns: list of detection results, one per camera
        """
        all_results = []
        
        for idx, frame in enumerate(frames):
            if frame is None:
                all_results.append(None)
                continue
            
            # Convert to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect hands
            results = self.hands_detectors[idx].process(rgb)
            all_results.append(results)
        
        return all_results
    
    def extract_landmarks_2d(self, results):
        """
        Extract 2D landmarks from MediaPipe results.
        Returns: list of (N_hands, 21, 2) arrays for each camera
        """
        all_landmarks_2d = []
        
        for cam_idx, result in enumerate(results):
            if result is None or not result.multi_hand_landmarks:
                all_landmarks_2d.append([])
                continue
            
            camera_hands = []
            for hand_landmarks in result.multi_hand_landmarks:
                # Extract normalized coordinates
                landmarks_norm = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
                
                # Convert to pixel coordinates
                landmarks_px = landmarks_norm * [self.width, self.height]
                
                # Undistort points
                landmarks_px = landmarks_px.reshape(-1, 1, 2)
                landmarks_undist = cv2.undistortPoints(
                    landmarks_px,
                    self.camera_matrices[cam_idx],
                    self.dist_coeffs[cam_idx],
                    P=self.camera_matrices[cam_idx]
                )
                landmarks_undist = landmarks_undist.reshape(-1, 2)
                
                camera_hands.append(landmarks_undist)
            
            all_landmarks_2d.append(camera_hands)
        
        return all_landmarks_2d
    
    def match_hands_across_cameras(self, all_landmarks_2d):
        """
        Match hands across cameras using position similarity.
        Returns: list of matched hand groups, each group is dict {cam_idx: hand_idx}
        """
        if not all_landmarks_2d[0]:
            return []
        
        matched_groups = []
        
        for hand_idx_0, landmarks_0 in enumerate(all_landmarks_2d[0]):
            group = {0: hand_idx_0}
            centroid_0 = np.mean(landmarks_0, axis=0)
            
            for cam_idx in range(1, self.num_cameras):
                if not all_landmarks_2d[cam_idx]:
                    continue
                
                best_match_idx = None
                best_distance = float('inf')
                
                for hand_idx, landmarks in enumerate(all_landmarks_2d[cam_idx]):
                    centroid = np.mean(landmarks, axis=0)
                    dist = np.linalg.norm(centroid - centroid_0) / self.width
                    
                    if dist < best_distance and dist < self.match_threshold:
                        best_distance = dist
                        best_match_idx = hand_idx
                
                if best_match_idx is not None:
                    group[cam_idx] = best_match_idx
            
            if len(group) >= self.min_cameras:
                matched_groups.append(group)
        
        return matched_groups
    
    def triangulate_landmark(self, landmark_points_2d, camera_indices):
        """
        Triangulate a single landmark from multiple views.
        """
        if len(landmark_points_2d) < 2:
            return None
        
        if self.triangulation_method == 'simple_average':
            triangulated_points = []
            
            ref_idx = camera_indices[0]
            ref_point = landmark_points_2d[0].reshape(2, 1)
            
            for i in range(1, len(landmark_points_2d)):
                other_idx = camera_indices[i]
                other_point = landmark_points_2d[i].reshape(2, 1)
                
                point_4d = cv2.triangulatePoints(
                    self.projection_matrices[ref_idx],
                    self.projection_matrices[other_idx],
                    ref_point,
                    other_point
                )
                
                point_3d = point_4d[:3] / point_4d[3]
                triangulated_points.append(point_3d.flatten())
            
            avg_point = np.mean(triangulated_points, axis=0)
            return avg_point
        
        else:
            point_4d = cv2.triangulatePoints(
                self.projection_matrices[camera_indices[0]],
                self.projection_matrices[camera_indices[1]],
                landmark_points_2d[0].reshape(2, 1),
                landmark_points_2d[1].reshape(2, 1)
            )
            point_3d = point_4d[:3] / point_4d[3]
            return point_3d.flatten()
    
    def triangulate_hand(self, all_landmarks_2d, matched_group):
        """
        Triangulate all landmarks for a matched hand.
        """
        landmarks_3d = []
        cam_indices = list(matched_group.keys())
        
        for lm_idx in range(self.num_landmarks):
            points_2d = []
            valid_cam_indices = []
            
            for cam_idx in cam_indices:
                hand_idx = matched_group[cam_idx]
                landmark_2d = all_landmarks_2d[cam_idx][hand_idx][lm_idx]
                points_2d.append(landmark_2d)
                valid_cam_indices.append(cam_idx)
            
            point_3d = self.triangulate_landmark(points_2d, valid_cam_indices)
            
            if point_3d is not None:
                landmarks_3d.append(point_3d)
            else:
                landmarks_3d.append(np.zeros(3))
        
        return np.array(landmarks_3d)
    
    def process_frame(self):
        """
        Main processing pipeline: capture, detect, match, triangulate.
        """
        frames = self.capture_frames()
        all_results = self.detect_hands_all_cameras(frames)
        all_landmarks_2d = self.extract_landmarks_2d(all_results)
        matched_groups = self.match_hands_across_cameras(all_landmarks_2d)
        
        triangulated_hands = []
        for group in matched_groups:
            landmarks_3d = self.triangulate_hand(all_landmarks_2d, group)
            triangulated_hands.append(landmarks_3d)
        
        return frames, triangulated_hands, all_results
    
    def cleanup(self):
        """Release all resources."""
        for cap in self.captures:
            cap.release()
        
        for hands in self.hands_detectors:
            hands.close()
        
        print("Multi-camera tracker cleanup complete")
