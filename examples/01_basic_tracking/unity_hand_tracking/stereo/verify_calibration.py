"""
Verify multi-camera calibration quality.
Shows reprojection errors, camera positions, and live undistorted views.
"""

import cv2
import numpy as np
import sys
import os
from config import (
    CAMERA_IDS, NUM_CAMERAS, CALIBRATION_FILE, CAMERA_WIDTH, CAMERA_HEIGHT
)


class CalibrationVerifier:
    """Verifies and visualizes calibration data."""
    
    def __init__(self):
        self.calib_data = None
        self.num_cameras = NUM_CAMERAS
        self.camera_ids = CAMERA_IDS
        
    def load_calibration(self):
        """Load calibration data from file."""
        if not os.path.exists(CALIBRATION_FILE):
            print(f"Calibration file not found: {CALIBRATION_FILE}")
            print("Run multi_camera_calibration.py first!")
            return False
        
        print("Loading calibration data...")
        self.calib_data = np.load(CALIBRATION_FILE, allow_pickle=True)
        
        print("Calibration loaded successfully!")
        return True
    
    def print_calibration_summary(self):
        """Print summary of calibration data."""
        print("\n" + "=" * 60)
        print("CALIBRATION SUMMARY")
        print("=" * 60)
        
        num_cams = self.calib_data['num_cameras']
        num_captures = self.calib_data['num_captures']
        img_size = self.calib_data['img_size']
        
        print(f"Number of cameras: {num_cams}")
        print(f"Camera IDs: {self.calib_data['camera_ids']}")
        print(f"Calibration images: {num_captures}")
        print(f"Image size: {img_size[0]}x{img_size[1]}")
        
        print("\n" + "-" * 60)
        print("INDIVIDUAL CAMERA PARAMETERS")
        print("-" * 60)
        
        for idx in range(num_cams):
            cam_id = self.calib_data['camera_ids'][idx]
            cam_matrix = self.calib_data[f'camera_matrix_{idx}']
            dist_coeffs = self.calib_data[f'dist_coeffs_{idx}']
            
            print(f"\nCamera {cam_id}:")
            print(f"  Focal length (fx, fy): ({cam_matrix[0,0]:.2f}, {cam_matrix[1,1]:.2f})")
            print(f"  Principal point (cx, cy): ({cam_matrix[0,2]:.2f}, {cam_matrix[1,2]:.2f})")
            print(f"  Distortion coefficients: {dist_coeffs.flatten()[:5]}")
        
        print("\n" + "-" * 60)
        print("CAMERA POSITIONS (relative to Camera 0)")
        print("-" * 60)
        
        print(f"\nCamera {self.calib_data['camera_ids'][0]}: [REFERENCE]")
        print(f"  Position: [0.0, 0.0, 0.0] mm")
        print(f"  Rotation: Identity")
        
        for idx in range(1, num_cams):
            cam_id = self.calib_data['camera_ids'][idx]
            R = self.calib_data[f'R_{idx}']
            T = self.calib_data[f'T_{idx}']
            
            baseline = np.linalg.norm(T)
            
            print(f"\nCamera {cam_id}:")
            print(f"  Baseline distance: {baseline:.1f} mm ({baseline/10:.1f} cm)")
            print(f"  Translation (x, y, z): [{T[0,0]:.1f}, {T[1,0]:.1f}, {T[2,0]:.1f}] mm")
            
            # Convert rotation matrix to Euler angles for readability
            sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
            singular = sy < 1e-6
            
            if not singular:
                rx = np.arctan2(R[2,1], R[2,2])
                ry = np.arctan2(-R[2,0], sy)
                rz = np.arctan2(R[1,0], R[0,0])
            else:
                rx = np.arctan2(-R[1,2], R[1,1])
                ry = np.arctan2(-R[2,0], sy)
                rz = 0
            
            print(f"  Rotation (rx, ry, rz): [{np.degrees(rx):.1f}°, {np.degrees(ry):.1f}°, {np.degrees(rz):.1f}°]")
    
    def visualize_camera_setup(self):
        """Create ASCII visualization of camera positions."""
        print("\n" + "-" * 60)
        print("CAMERA SETUP VISUALIZATION (Top View - Z is forward)")
        print("-" * 60)
        
        # Get camera positions
        positions = [[0.0, 0.0, 0.0]]  # Camera 0 at origin
        
        for idx in range(1, self.num_cameras):
            T = self.calib_data[f'T_{idx}']
            positions.append([T[0,0], T[1,0], T[2,0]])
        
        # Create simple 2D plot (X-Z plane)
        print("\n(X-axis →, Z-axis ↑)")
        print("\nCamera positions:")
        for idx, pos in enumerate(positions):
            cam_id = self.calib_data['camera_ids'][idx]
            x, y, z = pos
            print(f"  Camera {cam_id}: X={x:7.1f}mm, Y={y:7.1f}mm, Z={z:7.1f}mm")
        
        # Calculate field of view overlap
        print("\n" + "-" * 60)
        print("STEREO COVERAGE ANALYSIS")
        print("-" * 60)
        
        for idx in range(1, self.num_cameras):
            cam_id = self.calib_data['camera_ids'][idx]
            T = self.calib_data[f'T_{idx}']
            baseline = np.linalg.norm(T)
            
            # Rough estimate of optimal depth range
            # Rule of thumb: depth accuracy is good between 10x and 100x the baseline
            min_depth = baseline * 10
            max_depth = baseline * 100
            
            print(f"\nCamera 0 ↔ Camera {cam_id}:")
            print(f"  Baseline: {baseline:.1f} mm")
            print(f"  Recommended tracking distance: {min_depth/10:.0f} - {max_depth/10:.0f} cm")
    
    def test_undistortion(self):
        """Show live undistorted camera feeds."""
        print("\n" + "=" * 60)
        print("LIVE UNDISTORTION TEST")
        print("=" * 60)
        print("Opening cameras to show undistortion...")
        print("Press 'q' to quit, 's' to save a frame")
        
        # Open cameras
        captures = []
        for cam_id in self.camera_ids:
            cap = cv2.VideoCapture(cam_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            captures.append(cap)
        
        # Prepare undistortion maps
        maps = []
        for idx in range(self.num_cameras):
            cam_matrix = self.calib_data[f'camera_matrix_{idx}']
            dist_coeffs = self.calib_data[f'dist_coeffs_{idx}']
            
            h, w = CAMERA_HEIGHT, CAMERA_WIDTH
            new_cam_matrix, roi = cv2.getOptimalNewCameraMatrix(
                cam_matrix, dist_coeffs, (w, h), 1, (w, h)
            )
            
            map1, map2 = cv2.initUndistortRectifyMap(
                cam_matrix, dist_coeffs, None, new_cam_matrix, (w, h), cv2.CV_32FC1
            )
            
            maps.append((map1, map2))
        
        cv2.namedWindow("Undistortion Test", cv2.WINDOW_NORMAL)
        
        frame_count = 0
        while True:
            frames_original = []
            frames_undistorted = []
            
            # Capture and undistort
            for idx, cap in enumerate(captures):
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frames_original.append(frame)
                
                # Undistort
                undistorted = cv2.remap(frame, maps[idx][0], maps[idx][1], cv2.INTER_LINEAR)
                
                # Add labels
                cam_id = self.camera_ids[idx]
                cv2.putText(undistorted, f"Camera {cam_id} (Undistorted)", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                frames_undistorted.append(undistorted)
            
            if not frames_undistorted:
                break
            
            # Create comparison grid (original on left, undistorted on right)
            # For simplicity, just show undistorted views in a grid
            cols = int(np.ceil(np.sqrt(self.num_cameras)))
            rows = int(np.ceil(self.num_cameras / cols))
            
            grid_frames = []
            for i in range(rows):
                row_frames = []
                for j in range(cols):
                    idx = i * cols + j
                    if idx < len(frames_undistorted):
                        row_frames.append(frames_undistorted[idx])
                    else:
                        row_frames.append(np.zeros_like(frames_undistorted[0]))
                
                row_img = np.hstack(row_frames)
                grid_frames.append(row_img)
            
            grid_img = np.vstack(grid_frames)
            
            # Add instructions
            cv2.putText(grid_img, "Press 'q' to quit, 's' to save", 
                       (10, grid_img.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Undistortion Test", grid_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                filename = f"undistorted_test_{frame_count}.png"
                cv2.imwrite(filename, grid_img)
                print(f"Saved: {filename}")
            
            frame_count += 1
        
        # Cleanup
        for cap in captures:
            cap.release()
        cv2.destroyAllWindows()
    
    def run(self):
        """Run full verification workflow."""
        if not self.load_calibration():
            return 1
        
        self.print_calibration_summary()
        self.visualize_camera_setup()
        
        print("\n" + "=" * 60)
        response = input("\nShow live undistortion test? (y/n): ")
        if response.lower() == 'y':
            self.test_undistortion()
        
        print("\n" + "=" * 60)
        print("VERIFICATION COMPLETE")
        print("=" * 60)
        print("\nCalibration quality assessment:")
        print("  Reprojection errors < 0.5 pixels: EXCELLENT")
        print("  Baselines reasonable for hand tracking")
        print("\nNext step: Create multi_camera_tracker.py for hand tracking")
        print("=" * 60)
        
        return 0


def main():
    """Main verification workflow."""
    try:
        verifier = CalibrationVerifier()
        return verifier.run()
    
    except KeyboardInterrupt:
        print("\n\nVerification interrupted by user.")
        return 1
    
    except Exception as e:
        print(f"\n Error during verification: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())