import cv2
import numpy as np

# Load the video or image
cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide a video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=50)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Draw the circle and its center
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

    # Display the result
    cv2.imshow("Marker Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
