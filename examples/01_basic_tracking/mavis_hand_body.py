"""One camera, synchronized hand/body results, and visible quality flags.

Install MAVIS from the repository, then run this script. Press Q or Escape to
close. Use the record CLI with --mode hand-body to save synchronized data.
"""
import cv2
import numpy as np
from mavis_track import HandBodyTracker


def main():
    capture = cv2.VideoCapture(0)
    try:
        if not capture.isOpened():
            raise RuntimeError("Camera unavailable; check its index and permissions")
        with HandBodyTracker() as tracker:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result = tracker.process(frame)
                if result.body is not None:
                    for point, visibility in zip(result.body.landmarks, result.body.visibility):
                        if np.isfinite(visibility) and visibility >= tracker.visibility_threshold:
                            xy = tuple((np.clip(point[:2], 0, 1) *
                                        [frame.shape[1] - 1, frame.shape[0] - 1]).astype(int))
                            cv2.circle(frame, xy, 3, (255, 180, 40), -1)
                for hand in result.hands:
                    for point in hand.landmarks:
                        xy = tuple((np.clip(point[:2], 0, 1) *
                                    [frame.shape[1] - 1, frame.shape[0] - 1]).astype(int))
                        cv2.circle(frame, xy, 3, (0, 220, 255), -1)
                flags = result.quality
                text = (f"L: {flags.left_hand.name} | R: {flags.right_hand.name} | "
                        f"Body: {flags.body.name}")
                cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            .45, (255, 255, 255), 1)
                cv2.imshow("MAVIS hand/body", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
