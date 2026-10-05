"""Run with Python 3.10–3.12 and the base MAVIS package installed."""
import cv2
from mavis_track import HandTracker

def main():
    capture = cv2.VideoCapture(0)
    try:
        if not capture.isOpened():
            raise RuntimeError("Camera unavailable; check its index and permissions")
        with HandTracker(max_hands=2) as tracker:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result = tracker.process(frame)
                for hand in result.hands:
                    x, y, _ = hand.landmarks[0]
                    point = (int(x * frame.shape[1]), int(y * frame.shape[0]))
                    cv2.circle(frame, point, 10, (0, 220, 255), 2)
                    cv2.putText(frame, f"{hand.handedness} #{hand.track_id}", point,
                                cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 220, 255), 2)
                cv2.imshow("MAVIS", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
