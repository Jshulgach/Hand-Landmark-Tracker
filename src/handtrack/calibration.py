"""
Multi-camera calibration module.
"""

import cv2
import numpy as np
import os
import sys
from datetime import datetime

class MultiCameraCalibrator:
    """Handles calibration of multiple cameras."""
    
    def __init__(self, 
                 num_cameras=2, 
                 camera_ids=[0, 1],
                 width=640,
                 height=480,
                 rows=5, 
                 cols=7, 
                 square_size=32.0,
                 output_dir="calibration_data",
                 target_captures=20):
        
        self.num_cameras = num_cameras
        self.camera_ids = camera_ids
        self.checkerboard_size = (cols, rows)
        self.square_size = square_size
        self.img_size = (width, height)
        self.output_dir = output_dir
        self.output_file = os.path.join(output_dir, "multi_camera_calib.npz")
        
        # Initialize cameras
        self.captures = []
        self.camera_matrices = [None] * self.num_cameras
        self.dist_coeffs = [None] * self.num_cameras
        
        # Collected calibration data
        self.obj_points_all = [[] for _ in range(self.num_cameras)]  # 3D points
        self.img_points_all = [[] for _ in range(self.num_cameras)]  # 2D points
        
        # Prepare object points (3D coordinates of checkerboard corners)
        self.obj_points_template = np.zeros(
            (cols * rows, 3), np.float32
        )
        self.obj_points_template[:, :2] = np.mgrid[
            0:cols, 0:rows
        ].T.reshape(-1, 2)
        self.obj_points_template *= self.square_size
        
        # Capture counter
        self.num_captured = 0
        self.target_captures = target_captures
    
    def initialize_cameras(self):
        """Open all cameras."""
        print(f"Initializing {self.num_cameras} camera(s)...")
        
        for cam_id in self.camera_ids:
            print(f"  Camera {cam_id}...", end=" ")
            
            # Use DirectShow on Windows for better multi-camera support
            if os.name == 'nt':
                cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(cam_id)
            
            if not cap.isOpened():
                print("FAILED ")
                return False
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size[1])
            
            # Check actual resolution
            actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            if actual_width == 0 or actual_height == 0:
                print("FAILED (0x0). Retrying...", end=" ")
                cap.release()
                
                # Re-initialize without specific backend if needed, or just default
                if os.name == 'nt':
                    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(cam_id)
                
                if not cap.isOpened():
                    print("FAILED ")
                    return False
                
                # Set properties again
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size[1])
            
            self.captures.append(cap)
            print("OK [OK]")
        
        return True
    
    def find_checkerboard_corners(self, frame):
        """Find checkerboard corners in frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(
            gray, self.checkerboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        return ret, corners
    
    def run_capture_loop(self):
        """Interactively capture calibration images from all cameras."""
        print("\n" + "=" * 60)
        print("CALIBRATION IMAGE CAPTURE")
        print("=" * 60)
        print(f"Target: {self.target_captures} valid captures")
        print("\nInstructions:")
        print("  - Move the checkerboard to different positions and angles")
        print("  - Ensure the board is visible in ALL cameras")
        print("  - Press SPACE when ready to capture")
        print("  - Press 'q' or ESC to finish early")
        print("=" * 60)
        
        window_name = "Calibration - All Cameras"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        while self.num_captured < self.target_captures:
            frames = []
            valid_detections = [False] * self.num_cameras
            all_corners = [None] * self.num_cameras
            
            # Capture from all cameras
            for idx, cap in enumerate(self.captures):
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to read from camera {self.camera_ids[idx]}")
                    continue
                
                # Try to find checkerboard
                found, corners = self.find_checkerboard_corners(frame)
                
                if found:
                    valid_detections[idx] = True
                    all_corners[idx] = corners
                    
                    # Draw corners
                    cv2.drawChessboardCorners(frame, self.checkerboard_size, corners, found)
                    
                    # Green border for valid detection
                    cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), 
                                (0, 255, 0), 10)
                else:
                    # Red border for no detection
                    cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), 
                                (0, 0, 255), 10)
                
                # Add camera label and status
                status = "READY" if found else "NO BOARD"
                color = (0, 255, 0) if found else (0, 0, 255)
                cv2.putText(frame, f"Camera {self.camera_ids[idx]}: {status}", 
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                
                frames.append(frame)
            
            # Create grid display
            cols = int(np.ceil(np.sqrt(self.num_cameras)))
            rows = int(np.ceil(self.num_cameras / cols))
            
            grid_frames = []
            for i in range(rows):
                row_frames = []
                for j in range(cols):
                    idx = i * cols + j
                    if idx < len(frames):
                        row_frames.append(frames[idx])
                    else:
                        row_frames.append(np.zeros_like(frames[0]))
                
                row_img = np.hstack(row_frames)
                grid_frames.append(row_img)
            
            grid_img = np.vstack(grid_frames)
            
            # Add overall status
            all_valid = all(valid_detections)
            status_text = f"Captured: {self.num_captured}/{self.target_captures}"
            if all_valid:
                status_text += " - Press SPACE to capture!"
                status_color = (0, 255, 0)
            else:
                status_text += " - Move board until all cameras see it"
                status_color = (0, 0, 255)
            
            cv2.putText(grid_img, status_text, (20, grid_img.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)
            
            cv2.imshow(window_name, grid_img)
            
            # Handle keypresses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and all_valid:  # SPACE - capture
                # Save corners for all cameras
                for idx in range(self.num_cameras):
                    self.obj_points_all[idx].append(self.obj_points_template)
                    self.img_points_all[idx].append(all_corners[idx])
                
                self.num_captured += 1
                print(f"[OK] Captured {self.num_captured}/{self.target_captures}")
            
            elif key == ord('q') or key == 27:  # q or ESC - quit early
                if self.num_captured >= 10:
                    print(f"\nFinishing with {self.num_captured} captures...")
                    break
                else:
                    print(f"\nNeed at least 10 captures. Currently have {self.num_captured}.")
        
        cv2.destroyAllWindows()
        
        print(f"\n[OK] Capture complete! Collected {self.num_captured} valid image sets.")
        return self.num_captured >= 10
    
    def calibrate(self):
        """Run the full calibration process."""
        # Calibrate individual cameras
        print("\n" + "=" * 60)
        print("INDIVIDUAL CAMERA CALIBRATION")
        print("=" * 60)
        
        for idx in range(self.num_cameras):
            print(f"\nCalibrating camera {self.camera_ids[idx]}...")
            
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                self.obj_points_all[idx],
                self.img_points_all[idx],
                self.img_size,
                None, None
            )
            
            if ret:
                self.camera_matrices[idx] = camera_matrix
                self.dist_coeffs[idx] = dist_coeffs
                
                # Calculate reprojection error
                mean_error = 0
                for i in range(len(self.obj_points_all[idx])):
                    img_points2, _ = cv2.projectPoints(
                        self.obj_points_all[idx][i], rvecs[i], tvecs[i],
                        camera_matrix, dist_coeffs
                    )
                    error = cv2.norm(self.img_points_all[idx][i], img_points2, cv2.NORM_L2) / len(img_points2)
                    mean_error += error
                
                mean_error /= len(self.obj_points_all[idx])
                print(f"  Reprojection error: {mean_error:.3f} pixels")
            else:
                print(f"  Calibration failed!")
                return False
        
        print("\n Individual calibrations complete!")
        
        # Calibrate stereo pairs
        print("\n" + "=" * 60)
        print("STEREO PAIR CALIBRATION")
        print("=" * 60)
        
        # Store R and T for each camera relative to camera 0
        self.R_matrices = [np.eye(3)] + [None] * (self.num_cameras - 1)
        self.T_vectors = [np.zeros((3, 1))] + [None] * (self.num_cameras - 1)
        
        # Calibrate each camera relative to camera 0 (primary)
        for idx in range(1, self.num_cameras):
            print(f"\nCalibrating camera {self.camera_ids[idx]} relative to camera {self.camera_ids[0]}...")
            
            # Stereo calibration
            ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
                self.obj_points_all[0],  # Use camera 0's object points
                self.img_points_all[0],  # Camera 0 image points
                self.img_points_all[idx],  # Camera idx image points
                self.camera_matrices[0],
                self.dist_coeffs[0],
                self.camera_matrices[idx],
                self.dist_coeffs[idx],
                self.img_size,
                flags=cv2.CALIB_FIX_INTRINSIC
            )
            
            if ret:
                self.R_matrices[idx] = R
                self.T_vectors[idx] = T
                
                # Calculate baseline distance
                baseline = np.linalg.norm(T)
                print(f"  Baseline distance: {baseline:.1f} mm")
                print(f"  Translation: [{T[0][0]:.1f}, {T[1][0]:.1f}, {T[2][0]:.1f}] mm")
            else:
                print(f"  Stereo calibration failed!")
                return False
        
        print(f"\n[OK] Stereo calibrations complete!")
        return True
    
    def save(self, filename=None):
        """Save calibration data to file."""
        print("\n" + "=" * 60)
        print("SAVING CALIBRATION DATA")
        print("=" * 60)
        
        # Create directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        if filename is None:
            filename = self.output_file
            
        # Prepare data dictionary
        calib_data = {
            'num_cameras': self.num_cameras,
            'camera_ids': self.camera_ids,
            'img_size': self.img_size,
            'num_captures': self.num_captured,
            'checkerboard_size': self.checkerboard_size,
            'square_size': self.square_size,
        }
        
        # Add individual camera calibrations
        for idx in range(self.num_cameras):
            calib_data[f'camera_matrix_{idx}'] = self.camera_matrices[idx]
            calib_data[f'dist_coeffs_{idx}'] = self.dist_coeffs[idx]
        
        # Add stereo calibrations
        for idx in range(self.num_cameras):
            calib_data[f'R_{idx}'] = self.R_matrices[idx]
            calib_data[f'T_{idx}'] = self.T_vectors[idx]
        
        # Save to file
        np.savez(filename, **calib_data)
        print(f"[OK] Calibration saved to: {filename}")
        
        return True
    
    def cleanup(self):
        """Release camera resources."""
        for cap in self.captures:
            if cap:
                cap.release()
        cv2.destroyAllWindows()
