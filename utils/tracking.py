import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class HandTracker:
    """ Hand Tracker class for CLI mode

    Parameters
    ----------
    camera_id  (int)  : Camera ID to capture video stream
    img_size  (tuple) : Image size to resize the input frame
    max_hands  (int)  : Maximum number of hands to detect
    visualize (bool)  : Enable/disable visualization
    verbose   (bool)  : Enable/disable verbose output
    """
    def __init__(self, camera_id=0, img_size=(1080,720), max_hands=2, confidence=0.5, visualize=False, verbose=False):
        # Initialize the camera object and set the image size
        self.cap = cv2.VideoCapture(camera_id)
        self.img_size = img_size
        self.visualize = visualize
        self.verbose = verbose
        self.hand_landmarks = None

        # Create the Hand Tracker object
        self.hands = mp_hands.Hands(max_num_hands=max_hands, min_detection_confidence=confidence, min_tracking_confidence=0.5)

    def run(self):
        """ Run the Hand Tracker in CLI mode"""
        try:
            print("Press ESC or Ctrl+C to stop the Hand Tracker ...")
            while self.cap.isOpened():
                # Get the input frame and resize
                image = self.get_image()
                image = cv2.resize(image, self.img_size)

                # Make the detection, h-flip to accurately mirror the image
                image = self.detect_hands(image)
                if self.hand_landmarks:
                    for hand_landmarks in self.hand_landmarks:
                        if self.visualize:
                            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if self.visualize:
                    cv2.imshow('Hand Tracking CLI', image)
                    if cv2.waitKey(5) & 0xFF == 27:
                        break
        except KeyboardInterrupt:
            self.logger("Stopping the Hand Tracker...")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

    def get_image(self, flip_image=False):
        """ Get the image from the camera """
        success, image = self.cap.read()
        if success:
            image = cv2.resize(image, self.img_size)
            return cv2.flip(image, 1) if flip_image else image
        return None

    def detect_hands(self, image):
        """ Detect hands in the input image and return any landmarks """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            # Store local copy of the hand landmarks
            self.hand_landmarks = results.multi_hand_landmarks
        else:
            self.hand_landmarks = None
        return image
