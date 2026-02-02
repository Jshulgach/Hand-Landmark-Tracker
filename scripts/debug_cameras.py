import cv2
import time
import os

from examples.unity_hand_tracking.stereo.config import CAMERA_IDS, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS
# Or just hardcode for simplicity since path matching is annoying
CAMERA_IDS = [0, 1]
WIDTH = 1280
HEIGHT = 720
FPS = 30

def test_single_camera(cam_id):
    print(f"DEBUG: Testing Camera {cam_id} INDIVIDUALLY")
    if os.name == 'nt':
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(cam_id)
        
    if not cap.isOpened():
        print(f"DEBUG: Failed to open camera {cam_id}")
        return False
        
    # Check initial format
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    print(f"DEBUG: Camera {cam_id} initial FourCC: {fourcc_str}")
    
    # Try setting MJPG
    print(f"DEBUG: Setting MJPG for camera {cam_id}")
    ret_fourcc = cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    if not ret_fourcc:
        print(f"DEBUG: Failed to set MJPG for camera {cam_id}")
    else:
        print(f"DEBUG: Set MJPG success for camera {cam_id}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    
    ret, frame = cap.read()
    if ret:
        print(f"DEBUG: Successfully read frame from camera {cam_id}")
    else:
        print(f"DEBUG: Failed to read frame from camera {cam_id}")
        
    cap.release()
    return ret

def test_dual_cameras():
    print("\nDEBUG: Testing BOTH Cameras TOGETHER")
    caps = []
    
    for cam_id in CAMERA_IDS:
        print(f"DEBUG: Opening camera {cam_id}...")
        if os.name == 'nt':
            cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(cam_id)
            
        if not cap.isOpened():
            print(f"DEBUG: Failed to open camera {cam_id}")
            caps.append(None)
            continue
            
        # Set MJPG IMMEDIATELY
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        
        # Read a frame to verify
        ret, frame = cap.read()
        if ret:
            print(f"DEBUG: OK - Camera {cam_id}")
            caps.append(cap)
        else:
            print(f"DEBUG: Failed to read from camera {cam_id}")
            caps.append(None)
            
    print("-" * 20)
    success_count = sum(1 for c in caps if c is not None)
    print(f"DEBUG: {success_count}/{len(CAMERA_IDS)} cameras working together")
    
    for cap in caps:
        if cap:
            cap.release()

if __name__ == "__main__":
    try:
        print("=== SINGLE TESTS ===")
        c1 = test_single_camera(1)
        c2 = test_single_camera(2)
        
        if c1 and c2:
            test_dual_cameras()
        else:
            print("\nDEBUG: Skipping dual test because single tests failed.")
    except Exception as e:
        print(f"DEBUG: Exception: {e}")
