"""
Stereo (Multi-Camera) Tracker module with parallel processing and FPS monitoring.
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading

class MultiCameraTracker:
    """Tracks hands across multiple calibrated cameras with parallel processing."""
    
    def __init__(self, 
                 camera_ids, 
                 calibration_file,
                 width=640,
                 height=480,
                 num_landmarks=21,
                 max_hands=2,
                 triangulation_method='simple_average',
                 min_cameras=2,
                 match_threshold=0.1,
                 enable_parallel=True,
                 num_workers=4):
        
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
        
        # Parallel processing settings
        self.enable_parallel = enable_parallel
        self.num_workers = num_workers
        # Use more efficient thread pool sizing (2 per camera min)
        optimal_workers = max(self.num_cameras, min(num_workers, self.num_cameras * 2))
        self.executor = ThreadPoolExecutor(max_workers=optimal_workers) if enable_parallel else None
        
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
        
        # FPS monitoring
        self.frame_times = []
        self.capture_times = []
        self.detection_times = []
        self.triangulation_times = []
        self.fps_window_size = 30
        self.last_frame_time = time.time()
        self.frame_count = 0
        
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
    
    # ==================== FPS MONITORING ====================
    
    def get_fps_stats(self):
        """Return FPS statistics."""
        if len(self.frame_times) < 2:
            return {"fps": 0, "capture_ms": 0, "detection_ms": 0, "triangulation_ms": 0}
        
        # Calculate average of last N frames
        recent_frame_times = self.frame_times[-self.fps_window_size:]
        recent_capture_times = self.capture_times[-self.fps_window_size:]
        recent_detection_times = self.detection_times[-self.fps_window_size:]
        recent_tri_times = self.triangulation_times[-self.fps_window_size:]
        
        if len(recent_frame_times) > 1:
            frame_deltas = np.diff(recent_frame_times)
            avg_fps = 1.0 / np.mean(frame_deltas) if np.mean(frame_deltas) > 0 else 0
        else:
            avg_fps = 0
        
        return {
            "fps": avg_fps,
            "capture_ms": np.mean(recent_capture_times) * 1000 if recent_capture_times else 0,
            "detection_ms": np.mean(recent_detection_times) * 1000 if recent_detection_times else 0,
            "triangulation_ms": np.mean(recent_tri_times) * 1000 if recent_tri_times else 0,
            "total_ms": np.mean([t2 - t1 for t1, t2 in zip(recent_frame_times[:-1], recent_frame_times[1:])]) * 1000
        }
    
    # ==================== PARALLEL CAPTURE ====================
    
    def _capture_single_camera(self, cam_idx):
        """Capture a single frame from a camera (for threading)."""
        frame = None
        cap = self.captures[cam_idx]
        
        if cap is not None and cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    frame = None
            except Exception as e:
                print(f"Error reading from camera {self.camera_ids[cam_idx]}: {e}")
                frame = None
        
        # Auto-reconnection
        if frame is None:
            if self.reconnect_camera(cam_idx):
                try:
                    ret, frame = self.captures[cam_idx].read()
                    if not ret: frame = None
                except:
                    frame = None
        
        return cam_idx, frame
    
    def capture_frames(self):
        """
        Capture frames from all cameras in parallel.
        Returns: list of frames indexed by camera index
        """
        t0 = time.time()
        
        if self.enable_parallel and self.executor:
            # Submit all capture tasks
            future_to_idx = {}
            for cam_idx in range(self.num_cameras):
                future = self.executor.submit(self._capture_single_camera, cam_idx)
                future_to_idx[future] = cam_idx
            
            # Collect results AS THEY COMPLETE (not in submission order!)
            frames = [None] * self.num_cameras
            for future in as_completed(future_to_idx.keys(), timeout=0.15):
                try:
                    cam_idx, frame = future.result()
                    frames[cam_idx] = frame
                except Exception as e:
                    cam_idx = future_to_idx[future]
                    print(f"Capture failed for camera {self.camera_ids[cam_idx]}: {e}")
                    frames[cam_idx] = None
            
            capture_time = time.time() - t0
            self.capture_times.append(capture_time)
            return frames
        else:
            # Fallback to sequential capture
            frames = []
            for cam_idx in range(self.num_cameras):
                _, frame = self._capture_single_camera(cam_idx)
                frames.append(frame)
            
            capture_time = time.time() - t0
            self.capture_times.append(capture_time)
            return frames
    
    # ==================== PARALLEL DETECTION ====================
    
    def _detect_single_camera(self, cam_idx, frame):
        """Detect hands in a single camera frame (for threading)."""
        if frame is None:
            return cam_idx, None
        
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands_detectors[cam_idx].process(rgb)
            return cam_idx, results
        except Exception as e:
            print(f"Detection error camera {cam_idx}: {e}")
            return cam_idx, None
    
    def detect_hands_all_cameras(self, frames):
        """
        Detect hands in all camera frames in parallel.
        Returns: list of detection results, one per camera
        """
        t0 = time.time()
        
        if self.enable_parallel and self.executor:
            # Submit all detection tasks
            future_to_idx = {}
            for cam_idx, frame in enumerate(frames):
                future = self.executor.submit(self._detect_single_camera, cam_idx, frame)
                future_to_idx[future] = cam_idx
            
            # Collect results AS THEY COMPLETE (not in submission order!)
            all_results = [None] * self.num_cameras
            for future in as_completed(future_to_idx.keys(), timeout=0.15):
                try:
                    cam_idx, results = future.result()
                    all_results[cam_idx] = results
                except Exception as e:
                    cam_idx = future_to_idx[future]
                    print(f"Detection failed for camera {self.camera_ids[cam_idx]}: {e}")
                    all_results[cam_idx] = None
            
            detection_time = time.time() - t0
            self.detection_times.append(detection_time)
            return all_results
        else:
            # Fallback to sequential detection
            all_results = []
            for cam_idx, frame in enumerate(frames):
                _, results = self._detect_single_camera(cam_idx, frame)
                all_results.append(results)
            
            detection_time = time.time() - t0
            self.detection_times.append(detection_time)
            return all_results
    
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
        Main processing pipeline: capture, detect, match, triangulate with timing.
        """
        frame_start_time = time.time()
        self.frame_times.append(frame_start_time)
        self.frame_count += 1
        
        # Trim history if too long
        if len(self.frame_times) > self.fps_window_size * 2:
            self.frame_times = self.frame_times[-self.fps_window_size:]
            self.capture_times = self.capture_times[-self.fps_window_size:]
            self.detection_times = self.detection_times[-self.fps_window_size:]
            self.triangulation_times = self.triangulation_times[-self.fps_window_size:]
        
        # Capture frames (parallel if enabled)
        frames = self.capture_frames()
        
        # Detect hands (parallel if enabled)
        all_results = self.detect_hands_all_cameras(frames)
        
        # Extract, match, triangulate
        t_tri_start = time.time()
        all_landmarks_2d = self.extract_landmarks_2d(all_results)
        matched_groups = self.match_hands_across_cameras(all_landmarks_2d)
        
        triangulated_hands = []
        for group in matched_groups:
            landmarks_3d = self.triangulate_hand(all_landmarks_2d, group)
            triangulated_hands.append(landmarks_3d)
        
        tri_time = time.time() - t_tri_start
        self.triangulation_times.append(tri_time)
        
        return frames, triangulated_hands, all_results
    
    def cleanup(self):
        """Release all resources."""
        for cap in self.captures:
            cap.release()
        
        for hands in self.hands_detectors:
            hands.close()
        
        if self.executor:
            self.executor.shutdown(wait=True)
        
        print("Multi-camera tracker cleanup complete")
