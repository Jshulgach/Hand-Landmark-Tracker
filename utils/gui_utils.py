#from PyQt5 import QtWidgets
import cv2
import mediapipe as mp
import socket
import time
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

Landmarks = {
    #Dictionary for the mapping of the Mediapipe hand tracking landmarks to binary numbers
    "WRIST": 0,
    "THUMB_CMC": 1,
    "THUMB_MCP": 2,
    "THUMB_DIP": 3,
    "THUMB_TIP": 4,
    "INDEX_FINGER_MCP": 5,
    "INDEX_FINGER_PIP": 6,
    "INDEX_FINGER_DIP": 7,
    "INDEX_FINGER_TIP": 8,
    "MIDDLE_FINGER_MCP": 9,
    "MIDDLE_FINGER_PIP": 10,
    "MIDDLE_FINGER_DIP": 11,
    "MIDDLE_FINGER_TIP": 12,
    "RING_FINGER_MCP": 13,
    "RING_FINGER_PIP": 14,
    "RING_FINGER_DIP": 15,
    "RING_FINGER_TIP": 16,
    "PINKY_MCP": 17,
    "PINKY_PIP": 18,
    "PINKY_DIP": 19,
    "PINKY_TIP": 20,
}

class FrontUI(object):
    """

    """

    def setup_ui(self, Form):
        Form.setObjectName("Form")
        Form.setWindowTitle("Hand Tracker")
        Form.setStyleSheet("background : lightgrey;")
        Form.resize(525, 386)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")

        self.image_label = QtWidgets.QLabel(Form)
        self.image_label.setText("Webcam")
        self.image_label.setObjectName("image_label")
        self.verticalLayout.addWidget(self.image_label)

        self.control_bt = QtWidgets.QPushButton(Form)
        self.control_bt.setObjectName("control_bt")
        self.verticalLayout.addWidget(self.control_bt)

        bottom_layout = QtWidgets.QHBoxLayout()
        bottom_layout.addWidget(QtWidgets.QLabel("Streaming address"))

        self.ip_edit = QtWidgets.QLineEdit(Form)
        #self.ip_edit.setText("128.237.82.10")
        #self.ip_edit.setText("127.0.0.1")
        self.ip_edit.setText("shulgach-robots.duckdns.org")
        bottom_layout.addWidget(self.ip_edit)

        bottom_layout.addWidget(QtWidgets.QLabel("Enable Streaming", alignment=QtCore.Qt.AlignRight))

        self.enable_stream_bt = QtWidgets.QCheckBox()
        bottom_layout.addWidget(self.enable_stream_bt)

        self.verticalLayout.addLayout(bottom_layout)
        self.horizontalLayout.addLayout(self.verticalLayout)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Hand Landmark Tracker"))
        self.control_bt.setText(_translate("Form", "Start"))
        icon = QIcon("assets/video-camera.png") # <a href="https://www.flaticon.com/free-icons/video-camera" title="video camera icons">Video camera icons created by Freepik - Flaticon</a>
        self.control_bt.setIcon(icon)


class MainWindow(QtWidgets.QDialog, FrontUI):
    def __init__(self, name='Hand Tracker', rate=10, cap=None, cam_height=480, cam_width=640, verbose=False):
        super().__init__()
        self.name = name
        self.verbose = verbose
        self.setup_ui(self)
        self.control_bt.clicked.connect(self.start_webcam)
        self.image_label.setScaledContents(True)
        self.enable_stream_bt.stateChanged.connect(self.enable_stream_cb)
        self._enable_stream = self.enable_stream_bt.isChecked()

        self.rate = rate
        self.toc = 0.0
        self.PORT = 5000
        self._server_connected = False
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.cap = cap or cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)

        self.timer = QtCore.QTimer(self, interval=5)
        self.timer.timeout.connect(self.update_frame)
        self.hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.show()

    def start_webcam(self):
        self.timer.start()

    def update_frame(self):
        if self.cap.isOpened():
            success, image = self.cap.read()
            if success:
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = self.hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                self.display_image(image, True)

    def display_image(self, img, window=True):
        qformat = QtGui.QImage.Format_RGB888 if img.shape[2] == 3 else QtGui.QImage.Format_RGBA8888
        outImage = QtGui.QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat).rgbSwapped()
        if window:
            self.image_label.setPixmap(QtGui.QPixmap.fromImage(outImage))
