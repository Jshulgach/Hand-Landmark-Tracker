import sys
import cv2
import socket
import mediapipe as mp
from PyQt5 import QtCore, QtGui, QtWidgets
from utils import FrontUI

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

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
    def __init__(self, cap=None, cam_height=480, cam_width=640):
        super().__init__()
        self.setup_ui(self)
        self.control_bt.clicked.connect(self.start_webcam)
        self.image_label.setScaledContents(True)
        self.enable_stream_bt.stateChanged.connect(self.enable_stream_cb)
        self._enable_stream = self.enable_stream_bt.isChecked()

        # Set up socket client attributes (IP address comes from GUI input)
        #self.HOST = self.ip_edit.text()
        self.PORT = 5000
        self._server_connected = False
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Initialize the camera attribute, either inheriting or creating a new instance
        self.cap = cap
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

                self.display_image(image, True)  # Update the GUI with the webcam image

                if self._enable_stream:
                    if results.multi_hand_landmarks:
                        print("Streaming data: \n")
                        for i in results.multi_hand_landmarks:
                            print(str(i))
                            if self._server_connected:
                                self.s.send(str(i).encode('utf-8'))

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


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
