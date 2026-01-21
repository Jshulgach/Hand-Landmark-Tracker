"""
Test script to verify all cameras are working correctly.
Displays feeds from all cameras simultaneously in a grid layout.
Press 'q' to quit, 's' to save a test frame from each camera.
"""

import cv2
import numpy as np
import sys
from config import (
    CAMERA_IDS, NUM_CAMERAS, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS
)


def test_cameras():
    """Test all cameras and display their feeds."""
    
    print(f"Testing {NUM_CAMERAS} camera(s)...")
    print(f"Camera IDs: {CAMERA_IDS}")
    print("-" * 50)
    
    # Initialize all cameras
    captures = []
    failed_cameras = []
    
    for cam_id in CAMERA_IDS:
        print(f"Initializing camera {cam_id}...", end=" ")
        cap = cv2.VideoCapture(cam_id)
        
        if not cap.isOpened():
            print("FAILED ")
            failed_cameras.append(cam_id)
            captures.append(None)
            continue
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        
        # Read actual properties (may differ from requested)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"OK ({actual_width}x{actual_height} @ {actual_fps:.1f} FPS)")
        captures.append(cap)
    
    print("-" * 50)
    
    # Check if any cameras failed
    if failed_cameras:
        print(f"\n WARNING: Failed to open camera(s): {failed_cameras}")
        print("Please check:")
        print("  1. Camera is connected")
        print("  2. Camera ID is correct")
        print("  3. Camera is not being used by another application")
        print("  4. You have camera permissions")
        
        if len(failed_cameras) == NUM_CAMERAS:
            print("\n All cameras failed. Exiting.")
            return
    
    working_cameras = [cap for cap in captures if cap is not None]
    working_ids = [cam_id for cam_id, cap in zip(CAMERA_IDS, captures) if cap is not None]
    
    print(f"\n✓ {len(working_cameras)}/{NUM_CAMERAS} camera(s) working")
    print("\nControls:")
    print("  'q' or ESC - Quit")
    print("  's' - Save test frames")
    print("  'f' - Toggle fullscreen")
    print("-" * 50)
    
    # Calculate grid layout
    cols = int(np.ceil(np.sqrt(len(working_cameras))))
    rows = int(np.ceil(len(working_cameras) / cols))
    
    # Window setup
    window_name = f"Camera Test - {len(working_cameras)} Camera(s)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    frame_count = 0
    
    while True:
        frames = []
        
        # Capture frame from each working camera
        for cap in working_cameras:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                # Create blank frame if read fails
                frames.append(np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8))
        
        if not frames:
            print("No frames captured. Exiting.")
            break
        
        # Create grid layout
        grid_frames = []
        for i in range(rows):
            row_frames = []
            for j in range(cols):
                idx = i * cols + j
                if idx < len(frames):
                    frame = frames[idx].copy()
                    cam_id = working_ids[idx]
                    
                    # Add camera label
                    cv2.putText(frame, f"Camera {cam_id}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    
                    # Add frame counter
                    cv2.putText(frame, f"Frame: {frame_count}", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Add resolution
                    h, w = frame.shape[:2]
                    cv2.putText(frame, f"{w}x{h}", (10, frame.shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    
                    row_frames.append(frame)
                else:
                    # Blank frame for empty grid slot
                    row_frames.append(np.zeros_like(frames[0]))
            
            # Concatenate frames horizontally
            row_img = np.hstack(row_frames)
            grid_frames.append(row_img)
        
        # Concatenate rows vertically
        if grid_frames:
            grid_img = np.vstack(grid_frames)
            cv2.imshow(window_name, grid_img)
        
        frame_count += 1
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC
            print("\nQuitting...")
            break
        
        elif key == ord('s'):  # Save frames
            print("\nSaving test frames...")
            for idx, (frame, cam_id) in enumerate(zip(frames, working_ids)):
                filename = f"test_camera_{cam_id}_frame_{frame_count}.png"
                cv2.imwrite(filename, frame)
                print(f"  Saved: {filename}")
            print("Done!")
        
        elif key == ord('f'):  # Toggle fullscreen
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                 cv2.WINDOW_FULLSCREEN)
    
    # Cleanup
    print("\nCleaning up...")
    for cap in working_cameras:
        cap.release()
    cv2.destroyAllWindows()
    
    print("Test complete!")
    print("\nNext steps:")
    print("  1. If all cameras work, proceed to calibration")
    print("  2. If any cameras failed, fix issues and run this test again")
    print("  3. Update CAMERA_IDS in config.py if needed")


if __name__ == "__main__":
    try:
        test_cameras()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)