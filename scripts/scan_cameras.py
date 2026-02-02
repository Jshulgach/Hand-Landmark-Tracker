import cv2
import os

def scan_cameras(max_ids=10):
    print("Scanning for cameras...")
    available_cameras = []
    
    for i in range(max_ids):
        if os.name == 'nt':
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(i)
            
        if cap.isOpened():
            print(f"Camera {i}: FOUND")
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                print(f"  - Resolution: {w}x{h}")
                print(f"  - FPS: {cap.get(cv2.CAP_PROP_FPS)}")
                available_cameras.append(i)
            else:
                print(f"  - Failed to read frame")
            cap.release()
        else:
            print(f"Camera {i}: Not found")
            
    print("-" * 20)
    print(f"Available Camera IDs: {available_cameras}")

if __name__ == "__main__":
    scan_cameras()
