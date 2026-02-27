"""
Simple test script to debug checkerboard detection.
Shows live feed and tries multiple checkerboard size combinations.
"""

import cv2
import numpy as np
from config import CAMERA_IDS, CHECKERBOARD_ROWS, CHECKERBOARD_COLS

# Test different size combinations
test_sizes = [
    (CHECKERBOARD_COLS, CHECKERBOARD_ROWS),  # As configured
    (CHECKERBOARD_ROWS, CHECKERBOARD_COLS),  # Swapped
    (9, 11),  # Your board specs
    (11, 9),  # Swapped
    (8, 10),  # One less
    (10, 8),  # Swapped one less
]

print("Testing checkerboard detection...")
print(f"Camera: {CAMERA_IDS[0]}")
print(f"Configured size: {CHECKERBOARD_COLS}x{CHECKERBOARD_ROWS}")
print("\nTrying multiple size combinations...")

cap = cv2.VideoCapture(CAMERA_IDS[0])
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Failed to open camera!")
    exit(1)

cv2.namedWindow("Checkerboard Detection Test", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    display_frame = frame.copy()
    
    # Try each size combination
    detection_results = []
    for idx, size in enumerate(test_sizes):
        ret, corners = cv2.findChessboardCorners(
            gray, size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        status = "✓ FOUND" if ret else "✗ Not found"
        detection_results.append(f"{size[0]}x{size[1]}: {status}")
        
        # If found with this size, draw it
        if ret and idx == 0:  # Only draw first successful detection
            cv2.drawChessboardCorners(display_frame, size, corners, ret)
            cv2.rectangle(display_frame, (0, 0), 
                         (display_frame.shape[1]-1, display_frame.shape[0]-1), 
                         (0, 255, 0), 10)
    
    # Display results on frame
    y_offset = 30
    for result in detection_results:
        color = (0, 255, 0) if "FOUND" in result else (0, 0, 255)
        cv2.putText(display_frame, result, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 30
    
    cv2.putText(display_frame, "Press 'q' to quit", 
               (10, display_frame.shape[0] - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("Checkerboard Detection Test", display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("\nIf a size was detected, update config.py with:")
print("CHECKERBOARD_COLS = <first number>")
print("CHECKERBOARD_ROWS = <second number>")