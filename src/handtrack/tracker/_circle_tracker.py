import cv2
import numpy as np


class CircularMarkerTracker:
    def __init__(self, dp=1.0, min_dist=30, param1=100, param2=40, min_radius=0, max_radius=0):
        self.dp = dp
        self.min_dist = min_dist
        self.param1 = param1  # Upper threshold for the Canny edge detector
        self.param2 = param2  # Threshold for center detection
        self.min_radius = min_radius
        self.max_radius = max_radius

    def detect(self, frame):
        output = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 15)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for x, y, r in circles[0]:
                cv2.circle(output, (x, y), r, (0, 255, 0), 2)
                cv2.circle(output, (x, y), 2, (0, 0, 255), 3)

        return output, circles
