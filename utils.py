from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QIcon

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
        self.ip_edit.setText("128.237.82.10")
        #self.ip_edit.setText("127.0.0.1")
        #self.ip_edit.setText("shulgach-robots.duckdns.org")
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
