"""
Test script to verify all cameras are working correctly.
Displays feeds from all cameras simultaneously in a grid layout.
Press 'q' to quit, 's' to save a test frame from each camera.
"""

import argparse
import cv2
import numpy as np
import sys
import os
import time
from config import (
    CAMERA_IDS, NUM_CAMERAS, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS
)


COMMON_RESOLUTIONS = [
    (3840, 2160),
    (2560, 1440),
    (1920, 1080),
    (1600, 1200),
    (1280, 720),
    (1024, 768),
    (800, 600),
    (640, 480),
]


def _try_set_resolution(cap, width, height):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual_width == 0 or actual_height == 0:
        return False, actual_width, actual_height
    ret, frame = cap.read()
    if not ret or frame is None:
        return False, actual_width, actual_height
    h, w = frame.shape[:2]
    return (w == actual_width and h == actual_height), actual_width, actual_height


def _measure_fps(cap, seconds=1.5, warmup=15):
    for _ in range(warmup):
        cap.read()
    start = time.time()
    count = 0
    while time.time() - start < seconds:
        ret, _ = cap.read()
        if ret:
            count += 1
    elapsed = max(time.time() - start, 1e-6)
    return count / elapsed


def _select_best_resolution(cap, preferred=None, target_fps=None, verbose=False, cam_id=None):
    candidates = list(COMMON_RESOLUTIONS)
    if preferred:
        candidates = [preferred] + [r for r in candidates if r != preferred]
    best = None
    for w, h in candidates:
        ok, aw, ah = _try_set_resolution(cap, w, h)
        if ok:
            if target_fps is not None:
                fps = _measure_fps(cap)
                if fps >= target_fps:
                    if verbose:
                        print(f"  Cam {cam_id} OK {aw}x{ah} @ {fps:.1f} fps (target {target_fps})")
                    best = (aw, ah)
                    break
                if verbose:
                    print(f"  Cam {cam_id} LOW {aw}x{ah} @ {fps:.1f} fps (target {target_fps})")
            else:
                if verbose:
                    print(f"  Cam {cam_id} OK {aw}x{ah}")
                best = (aw, ah)
                break
        elif verbose:
            print(f"  Cam {cam_id} FAIL {w}x{h}")
    return best


def test_cameras(use_max=True, save_config=False, normalize_display=True, target_fps=None, verbose=False):
    """Test all cameras and display their feeds."""
    
    print(f"Testing {NUM_CAMERAS} camera(s)...")
    print(f"Camera IDs: {CAMERA_IDS}")
    print("-" * 50)
    
    # Initialize all cameras
    captures = []
    failed_cameras = []
    
    selected_resolutions = {}
    for cam_id in CAMERA_IDS:
        print(f"Initializing camera {cam_id}...", end=" ")
        
        # Use DirectShow on Windows for better multi-camera support
        if os.name == 'nt':
            cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(cam_id)
        
        if not cap.isOpened():
            print("FAILED ")
            failed_cameras.append(cam_id)
            captures.append(None)
            continue
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        if use_max:
            if verbose:
                print(f" Probing resolutions for camera {cam_id} (target fps: {target_fps})")
            best = _select_best_resolution(cap, target_fps=target_fps, verbose=verbose, cam_id=cam_id)
            if best:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, best[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, best[1])
                selected_resolutions[cam_id] = {"width": best[0], "height": best[1]}
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        
        # Read actual properties (may differ from requested)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Check if camera initialized correctly
        if actual_width == 0 or actual_height == 0:
            print(f"FAILED (0x0 resolution). Retrying without MJPG force...", end=" ")
            cap.release()
            
            # Re-initialize
            if os.name == 'nt':
                cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(cam_id)
            
            if not cap.isOpened():
                print("FAILED (Could not re-open)")
                failed_cameras.append(cam_id)
                captures.append(None)
                continue
                
            # Set properties without forcing MJPG
            if use_max:
                if verbose:
                    print(f" Probing resolutions for camera {cam_id} (target fps: {target_fps})")
                best = _select_best_resolution(cap, target_fps=target_fps, verbose=verbose, cam_id=cam_id)
                if best:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, best[0])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, best[1])
                    selected_resolutions[cam_id] = {"width": best[0], "height": best[1]}
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            
            # Read actual properties again
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        print(f"OK ({actual_width}x{actual_height} @ {actual_fps:.1f} FPS, {fourcc_str})")
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
    
    print(f"\n[OK] {len(working_cameras)}/{NUM_CAMERAS} camera(s) working")
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
    last_fps_time = time.time()
    last_fps_frames = 0
    measured_fps = 0.0
    
    display_size = None
    while True:
        frames = []
        native_sizes = []
        fps_values = []
        
        # Capture frame from each working camera
        for cap in working_cameras:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                h, w = frame.shape[:2]
                native_sizes.append((w, h))
                fps_values.append(cap.get(cv2.CAP_PROP_FPS))
            else:
                # Create blank frame if read fails
                frames.append(np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8))
                native_sizes.append((CAMERA_WIDTH, CAMERA_HEIGHT))
                fps_values.append(0.0)
        
        if not frames:
            print("No frames captured. Exiting.")
            break
        
        # Determine common display size (smallest native size)
        if normalize_display and native_sizes:
            min_w = min(w for w, _ in native_sizes)
            min_h = min(h for _, h in native_sizes)
            display_size = (min_w, min_h)

        # Create grid layout
        grid_frames = []
        for i in range(rows):
            row_frames = []
            for j in range(cols):
                idx = i * cols + j
                if idx < len(frames):
                    frame = frames[idx].copy()
                    cam_id = working_ids[idx]
                    native_w, native_h = native_sizes[idx]
                    if normalize_display and display_size:
                        frame = cv2.resize(frame, display_size, interpolation=cv2.INTER_AREA)

                    # Add camera label
                    cv2.putText(frame, f"Camera {cam_id}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    
                    # Add frame counter
                    cv2.putText(frame, f"Frame: {frame_count}", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Add resolution and target FPS
                    h, w = frame.shape[:2]
                    if normalize_display and display_size:
                        cv2.putText(
                            frame,
                            f"Native {native_w}x{native_h} | Display {w}x{h}",
                            (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (200, 200, 200),
                            1,
                        )
                    else:
                        cv2.putText(frame, f"{w}x{h}", (10, frame.shape[0] - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    if target_fps:
                        cv2.putText(frame, f"Target FPS: {target_fps:.1f}", (10, frame.shape[0] - 45),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    cam_fps = fps_values[idx]
                    if cam_fps:
                        cv2.putText(frame, f"Reported FPS: {cam_fps:.1f}", (10, frame.shape[0] - 65),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    if measured_fps:
                        cv2.putText(frame, f"Measured FPS: {measured_fps:.1f}", (10, frame.shape[0] - 85),
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
        last_fps_frames += 1
        now = time.time()
        if now - last_fps_time >= 1.0:
            measured_fps = last_fps_frames / max(now - last_fps_time, 1e-6)
            last_fps_time = now
            last_fps_frames = 0
        
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
    if save_config and use_max and selected_resolutions:
        import json
        out_path = "camera_max_resolutions.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"resolutions": selected_resolutions}, f, indent=2)
        print(f"  4. Saved max resolutions: {out_path}")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Test cameras and optionally use max resolution.")
        parser.add_argument("--no-max", action="store_true", help="Disable probing; use config resolution")
        parser.add_argument("--save-config", action="store_true", help="Save selected max resolutions to JSON")
        parser.add_argument("--no-normalize", action="store_true", help="Do not normalize display size across cameras")
        parser.add_argument("--target-fps", type=float, default=CAMERA_FPS, help="Minimum FPS target when probing max")
        parser.add_argument("--quiet", action="store_true", help="Disable verbose optimization output")
        args = parser.parse_args()
        test_cameras(
            use_max=not args.no_max,
            save_config=args.save_config,
            normalize_display=not args.no_normalize,
            target_fps=args.target_fps,
            verbose=not args.quiet,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
