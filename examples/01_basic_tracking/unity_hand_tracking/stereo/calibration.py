"""
Multi-camera calibration script wrapper.
Uses the handtrack package for core logic.
"""

import sys
from config import (
    CAMERA_IDS, NUM_CAMERAS, CAMERA_WIDTH, CAMERA_HEIGHT,
    CHECKERBOARD_ROWS, CHECKERBOARD_COLS, CHECKERBOARD_SQUARE_SIZE,
    NUM_CALIBRATION_IMAGES, CALIBRATION_DIR
)
from handtrack.calibration import MultiCameraCalibrator


def main():
    """Main calibration workflow."""
    print("\n" + "=" * 60)
    print("MULTI-CAMERA CALIBRATION")
    print("=" * 60)
    print(f"Cameras: {NUM_CAMERAS}")
    print(f"Camera IDs: {CAMERA_IDS}")
    print(f"Checkerboard: {CHECKERBOARD_COLS}x{CHECKERBOARD_ROWS}")
    print(f"Square size: {CHECKERBOARD_SQUARE_SIZE} mm")
    print(f"Target images: {NUM_CALIBRATION_IMAGES}")
    print("=" * 60)
    
    calibrator = MultiCameraCalibrator(
        num_cameras=NUM_CAMERAS,
        camera_ids=CAMERA_IDS,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        rows=CHECKERBOARD_ROWS,
        cols=CHECKERBOARD_COLS,
        square_size=CHECKERBOARD_SQUARE_SIZE,
        output_dir=CALIBRATION_DIR,
        target_captures=NUM_CALIBRATION_IMAGES
    )
    
    try:
        # Initialize cameras
        if not calibrator.initialize_cameras():
            print("\n Failed to initialize cameras!")
            return 1
        
        # Capture calibration images
        if not calibrator.run_capture_loop():
            print("\n Not enough calibration images captured!")
            return 1
        
        # Calibrate
        if not calibrator.calibrate():
            print("\n Calibration failed!")
            return 1
        
        # Save calibration
        if not calibrator.save():
            print("\n Failed to save calibration!")
            return 1
        
        print("\n" + "=" * 60)
        print("[OK] CALIBRATION COMPLETE!")
        print("=" * 60)
        
        return 0
    
    except KeyboardInterrupt:
        print("\n\nCalibration interrupted by user.")
        return 1
    
    except Exception as e:
        print(f"\n Error during calibration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        calibrator.cleanup()


if __name__ == "__main__":
    sys.exit(main())