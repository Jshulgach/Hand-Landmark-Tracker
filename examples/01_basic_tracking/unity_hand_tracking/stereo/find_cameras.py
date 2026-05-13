import cv2


def find_available_cameras(max_tested=10):
    available_cameras = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras


if __name__ == "__main__":
    print("Scanning for available cameras...")
    cameras = find_available_cameras()
    if cameras:
        print(f"Found cameras at indices: {cameras}")
    else:
        print("No cameras found. Please check your connections.")
