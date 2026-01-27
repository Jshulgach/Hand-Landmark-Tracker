"""
Multi-camera hand tracking with 3D triangulation.
Handles camera capture, hand detection, matching, and triangulation.
"""

import cv2
import numpy as np
import mediapipe as mp
from config import (
    CAMERA_IDS, NUM_CAMERAS, CAMERA_WIDTH, CAMERA_HEIGHT,
    CALIBRATION_FILE, NUM_LANDMARKS, MAX_HANDS,
    MIN_CAMERAS_FOR_TRIANGULATION, HAND_MATCH_THRESHOLD,
    WORLD_COORDINATE_SYSTEM, TRIANGULATION_METHOD
)


class MultiCameraTracker:
    """Tracks hands across multiple calibrated cameras."""
    
    def __init__(self):
        self.num_cameras = NUM_CAMERAS
        self.camera_ids = CAMERA_IDS
        self.num_landmarks = NUM_LANDMARKS
        
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
        try:
            calib_data = np.load(CALIBRATION_FILE, allow_pickle=True)
            
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
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.captures.append(cap)
            
            # Create MediaPipe Hands detector
            hands = mp.solutions.hands.Hands(
                max_num_hands=MAX_HANDS,
                model_complexity=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.hands_detectors.append(hands)
            
            print(f" Camera {cam_id} initialized")
        
        return True
    
    def capture_frames(self):
        """Capture frames from all cameras."""
        frames = []
        for cap in self.captures:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                frames.append(None)
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
                h, w = CAMERA_HEIGHT, CAMERA_WIDTH
                landmarks_px = landmarks_norm * [w, h]
                
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
        # Simple greedy matching based on centroid distance
        # For each hand in camera 0, find closest hand in other cameras
        
        if not all_landmarks_2d[0]:
            return []
        
        matched_groups = []
        
        # For each hand in camera 0 (reference camera)
        for hand_idx_0, landmarks_0 in enumerate(all_landmarks_2d[0]):
            group = {0: hand_idx_0}
            centroid_0 = np.mean(landmarks_0, axis=0)
            
            # Find matching hands in other cameras
            for cam_idx in range(1, self.num_cameras):
                if not all_landmarks_2d[cam_idx]:
                    continue
                
                best_match_idx = None
                best_distance = float('inf')
                
                for hand_idx, landmarks in enumerate(all_landmarks_2d[cam_idx]):
                    centroid = np.mean(landmarks, axis=0)
                    
                    # Normalize distance by image size
                    dist = np.linalg.norm(centroid - centroid_0) / CAMERA_WIDTH
                    
                    if dist < best_distance and dist < HAND_MATCH_THRESHOLD:
                        best_distance = dist
                        best_match_idx = hand_idx
                
                if best_match_idx is not None:
                    group[cam_idx] = best_match_idx
            
            # Only keep groups visible in at least MIN_CAMERAS_FOR_TRIANGULATION cameras
            if len(group) >= MIN_CAMERAS_FOR_TRIANGULATION:
                matched_groups.append(group)
        
        return matched_groups
    
    def triangulate_landmark(self, landmark_points_2d, camera_indices):
        """
        Triangulate a single landmark from multiple views.
        landmark_points_2d: list of 2D points (one per camera)
        camera_indices: list of camera indices corresponding to the points
        Returns: 3D point in world coordinates
        """
        if len(landmark_points_2d) < 2:
            return None
        
        # Use cv2.triangulatePoints for pairs, then average
        if TRIANGULATION_METHOD == 'simple_average':
            triangulated_points = []
            
            # Triangulate using camera 0 as reference with each other camera
            ref_idx = camera_indices[0]
            ref_point = landmark_points_2d[0].reshape(2, 1)
            
            for i in range(1, len(landmark_points_2d)):
                other_idx = camera_indices[i]
                other_point = landmark_points_2d[i].reshape(2, 1)
                
                # Triangulate
                point_4d = cv2.triangulatePoints(
                    self.projection_matrices[ref_idx],
                    self.projection_matrices[other_idx],
                    ref_point,
                    other_point
                )
                
                # Convert from homogeneous to 3D
                point_3d = point_4d[:3] / point_4d[3]
                triangulated_points.append(point_3d.flatten())
            
            # Average all triangulated points
            avg_point = np.mean(triangulated_points, axis=0)
            return avg_point
        
        else:
            # Default: use first pair only
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
        Returns: (21, 3) array of 3D landmarks
        """
        landmarks_3d = []
        
        # Get camera indices and corresponding landmarks
        cam_indices = list(matched_group.keys())
        
        # For each landmark point
        for lm_idx in range(self.num_landmarks):
            points_2d = []
            valid_cam_indices = []
            
            for cam_idx in cam_indices:
                hand_idx = matched_group[cam_idx]
                landmark_2d = all_landmarks_2d[cam_idx][hand_idx][lm_idx]
                points_2d.append(landmark_2d)
                valid_cam_indices.append(cam_idx)
            
            # Triangulate this landmark
            point_3d = self.triangulate_landmark(points_2d, valid_cam_indices)
            
            if point_3d is not None:
                landmarks_3d.append(point_3d)
            else:
                # Fallback: use zeros
                landmarks_3d.append(np.zeros(3))
        
        return np.array(landmarks_3d)
    
    def process_frame(self):
        """
        Main processing pipeline: capture, detect, match, triangulate.
        Returns: (frames, triangulated_hands, all_results)
            - frames: list of camera frames
            - triangulated_hands: list of (21, 3) landmark arrays
            - all_results: MediaPipe results for visualization
        """
        # Capture frames
        frames = self.capture_frames()
        
        # Detect hands in all cameras
        all_results = self.detect_hands_all_cameras(frames)
        
        # Extract 2D landmarks
        all_landmarks_2d = self.extract_landmarks_2d(all_results)
        
        # Match hands across cameras
        matched_groups = self.match_hands_across_cameras(all_landmarks_2d)
        
        # Triangulate each matched hand
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