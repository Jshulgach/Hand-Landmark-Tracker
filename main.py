import sys
import time

import cv2
import math
import socket
import numpy as np
import mediapipe as mp
from scipy.spatial.transform import Rotation as R
from PyQt5 import QtCore, QtGui, QtWidgets
from utils import FrontUI, Landmarks

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
DEFAULT_AXIS_DRAWING_SPEC = mp_drawing.DrawingSpec()

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)


instructions = """Instructions:
 - Press the "Start Webcam" button to connect the webcam
 - Place hand in front of the camera to track hand landmarks
 - Input an IP address and enable the "Stream" checkbox to stream the landmark data
"""
print(instructions)


class MainWindow(QtWidgets.QDialog, FrontUI):
    """
        MainWindow object handles the camera acquisition, obtaining landmarks from mediapipe, and the entry point
        for the app

        Parameters
        ----------
        cap : Object
            object obtained from cv2, for capturing video frame.
        cam_height : int
            highet in pixels of obtained frame from camera.
        cam_width : int
            width in pixels of obtained frame from camera.
    """

    def __init__(self, name='Hand Tracker', cap=None, cam_height=480, cam_width=640, verbose=False):
        super().__init__()
        self.name = name
        self.verbose = verbose

        # Set up the GUI
        self.setup_ui(self)
        self.control_bt.clicked.connect(self.start_webcam)
        self.image_label.setScaledContents(True)
        self.enable_stream_bt.stateChanged.connect(self.enable_stream_cb)
        self._enable_stream = self.enable_stream_bt.isChecked()

        # Set up socket client attributes (IP address comes from GUI input)
        # self.HOST = self.ip_edit.text()
        self.toc = 0.0
        self.PORT = 5000
        self._server_connected = False
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Initialize the camera attribute, either inheriting or creating a new instance
        self.cap = cap
        self.cam_height = cam_height
        self.cam_width = cam_width
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)

        # Create timer to handle the frames captured per second
        self.timer = QtCore.QTimer(self, interval=5)
        self.timer.timeout.connect(self.update_frame)

        # Create the HandLandmarker object
        self.hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.connect_server()

        # Show the gui
        self.show()

    def logger(self, *argv):
        msg = ''.join(argv)
        print("[{:.3f}][{}] {}".format(time.monotonic(), self.name, msg))

    def connect_server(self):
        """Function that attempts to make a connection to the server """
        try:
            print("Attempting to connect to server at {}, port: {}".format(self.ip_edit.text(), self.PORT))
            self.s.connect((self.ip_edit.text(), self.PORT))
            self._server_connected = True
            print("Success")
        except:
            print("Error connecting to server, timeout")

    @QtCore.pyqtSlot()
    def enable_stream_cb(self):
        """ Callback function to adjust streaming property"""
        self._enable_stream = self.enable_stream_bt.isChecked()

    @QtCore.pyqtSlot()
    def start_webcam(self):
        """
            Entry point of whole programm,  'handmajor' and 'handminor' for
            controlling.
        """
        self.timer.start()

    @QtCore.pyqtSlot()
    def update_frame(self):
        """ Main function that captures a new video frame and passes it to the landmark detection model

        """
        if self.cap.isOpened():
            success, image = self.cap.read()  # Capture a video frame
            if success:

                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = self.hands.process(image)  # Run the landmark detection on the image
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # If landmarks have been detected, draw them on top of the image
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get the position and orientation of the wrist landmark
                    if results.multi_hand_landmarks:

                        wrist_lm = self.get_landmark_pos(results, "WRIST")
                        thumb_mcp = self.get_landmark_pos(results, "THUMB_MCP")
                        index_mcp = self.get_landmark_pos(results, "INDEX_FINGER_MCP")
                        index_tip = self.get_landmark_pos(results, "INDEX_FINGER_TIP")
                        thumb_tip = self.get_landmark_pos(results, "THUMB_TIP")
                        #pinky_mcp = self.get_landmark_pos(results, "PINKY_MCP")

                        roll, pitch, yaw = self.hand_orientation(thumb_mcp, index_mcp, wrist_lm)
                        pinch_dist = self.get_diff(thumb_tip, index_tip)
                        pinch_dist = round(pinch_dist, 2)

                        x = round(wrist_lm[0], 2)
                        y = round(wrist_lm[1], 2)
                        z = round(1000000*wrist_lm[2], 4)

                        tic = time.perf_counter()
                        if self._enable_stream:
                            if tic - self.toc > 0.1:
                                if results.multi_hand_landmarks:
                                    if self._server_connected:
                                        self.toc = time.perf_counter()
                                        self.logger("Streaming data: \n")
                                        i = "GRIPPER:{};POSITION:{},{},{};ROTATION:{},{},{};".format(pinch_dist, x, y, z, round(roll, 1), round(pitch, 1), round(yaw), 1)
                                        print(i)
                                        self.s.send(str(i).encode('utf-8'))

                self.display_image(image, True)  # Update the GUI with the webcam image

    def display_image(self, img, window=True):
        qformat = QtGui.QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QtGui.QImage.Format_RGBA8888
            else:
                qformat = QtGui.QImage.Format_RGB888
        outImage = QtGui.QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        if window:
            self.image_label.setPixmap(QtGui.QPixmap.fromImage(outImage))

    def get_landmark_pos(self, result, key):
        """ Function that returns the position of a landmark

        Args:
            result (mp_hands.HandLandmark): Result of the landmark detection model
            key (str): Name of the landmark

        Returns:
            list: List containing the x, y, and z coordinates of the landmark
        """
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                return [lm[Landmarks[key]].x, lm[Landmarks[key]].y, lm[Landmarks[key]].z]

    def hand_orientation(self, p1, p2, p3, unit='degrees'):
        """ Function that returns the roll, pitch, and yaw of the palm landmarks. Math steps from Matlab forums:
             https://www.mathworks.com/matlabcentral/answers/298940-how-to-calculate-roll-pitch-and-yaw-from-xyz-coordinates-of-3-planar-points
             Note that this function took a long time to figure out and the result is somewhat simple, but this only
             works using points that sit inside a plane which is parallel to the palm

        Parameters:
            p1 (list): List containing the x, y, and z coordinates of the wrist landmark
            p2 (list): List containing the x, y, and z coordinates of the thumb mcp landmark
            p3 (list): List containing the x, y, and z coordinates of the index mcp landmark
            unit (str): Unit of the angles returned. Can be 'degrees' or 'radians'

        Returns:
            roll (float): Roll angle in radians or degrees
            pitch (float): Pitch angle in radians or degrees
            yaw (float): Yaw angle in radians or degrees
        """
        # Get the Z unit vector
        v1 = np.subtract(p2, p1)
        v2 = np.subtract(p3, p1)
        Z = np.cross(v1, v2)
        Z /= np.linalg.norm(Z)

        # Get the X unit vector
        X = np.subtract(np.add(p1, p2) / 2, p3)
        X /= np.linalg.norm(X)

        # Get the Y unit vector
        Y = np.cross(Z, X)
        Y /= np.linalg.norm(Y)

        # Build the rotation matrix
        rotation = np.array([X, Y, Z]).transpose()
        roll = math.atan2(-rotation[2][1], rotation[2][2])
        pitch = math.asin(rotation[2][0])
        yaw = math.atan2(-rotation[1][0], rotation[0][0])

        if unit == 'degrees':
            roll *= 180 / np.pi
            pitch *= 180 / np.pi
            yaw = yaw*180 / np.pi - 90 # subtracting 90 so that the yaw is 0 when the hand is pointing up

        if self.verbose:
            print("roll: {}, pitch: {}, yaw: {}".format(roll, pitch, yaw))

        return roll, pitch, yaw

    def get_diff(self, thumb_tip, index_tip):
        """ Function that returns the difference between the thumb and index finger tips

        Args:
            thumb_tip (list): List containing the x, y, and z coordinates of the thumb tip
            index_tip (list): List containing the x, y, and z coordinates of the index tip

        Returns:
            list: List containing the x, y, and z differences between the two tips
        """
        return np.sqrt((thumb_tip[0]-index_tip[0])**2 + (thumb_tip[1]-index_tip[1])**2 + (thumb_tip[2]-index_tip[2])**2)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(verbose=False)
    sys.exit(app.exec_())
