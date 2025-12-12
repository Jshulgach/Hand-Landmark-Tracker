import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout,
    QWidget, QPushButton, QFileDialog, QSlider, QSpinBox
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

from handtrack.tracker import CircularMarkerTracker


class MarkerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Passive Marker Tracking GUI")
        self.setGeometry(100, 100, 1000, 700)

        self.tracker = CircularMarkerTracker()
        self.capture = None
        self.image_path = None

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)

        # Controls
        self.load_btn = QPushButton("Load Image/Video")
        self.param2_slider = QSlider(Qt.Horizontal)
        self.param2_slider.setRange(10, 100)
        self.param2_slider.setValue(40)
        self.param2_slider.valueChanged.connect(self.update_tracker_param)

        self.min_dist_spin = QSpinBox()
        self.min_dist_spin.setRange(5, 100)
        self.min_dist_spin.setValue(30)
        self.min_dist_spin.valueChanged.connect(self.update_tracker_param)

        # Layout
        controls = QHBoxLayout()
        controls.addWidget(self.load_btn)
        controls.addWidget(QLabel("Sensitivity"))
        controls.addWidget(self.param2_slider)
        controls.addWidget(QLabel("Min Dist"))
        controls.addWidget(self.min_dist_spin)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(controls)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Connect button
        self.load_btn.clicked.connect(self.load_media)

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def update_tracker_param(self):
        self.tracker.param2 = self.param2_slider.value()
        self.tracker.min_dist = self.min_dist_spin.value()

    def load_media(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image or Video")
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            self.image_path = file_path
            self.timer.stop()
            frame = cv2.imread(self.image_path)
            self.display_frame(frame)
        else:
            self.capture = cv2.VideoCapture(file_path)
            self.timer.start(30)

    def update_frame(self):
        if self.capture:
            ret, frame = self.capture.read()
            if not ret:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return
            self.display_frame(frame)

    def display_frame(self, frame):
        frame = cv2.resize(frame, (960, 540))
        processed, _ = self.tracker.detect(frame)
        rgb_image = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        if self.capture:
            self.capture.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = MarkerGUI()
    gui.show()
    sys.exit(app.exec_())
