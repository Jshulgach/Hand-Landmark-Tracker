"""
Face Tracker with Control Panel

An interactive face tracking application with a PyQt5 control panel for:
- Toggling Kalman filter smoothing
- Setting save directory for landmark data (CSV/NPZ)
- Switching between webcam and video file input
"""

import argparse
import sys
import os
import csv
import time
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QCheckBox, QLineEdit, QPushButton, QFileDialog, QGroupBox,
    QRadioButton, QButtonGroup, QSpinBox, QSlider, QFrame
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

# Add parent directory to path if handtrack is not installed
src_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
sys.path.insert(0, os.path.abspath(src_path))

try:
    from handtrack.processing import Kalman3D
except ImportError:
    # Fallback: inline Kalman3D if handtrack not available
    class Kalman3D:
        def __init__(self, dt=1/30, process_noise=1e-3, measurement_noise=1e-2):
            self.x = np.zeros((6, 1))
            self.F = np.eye(6)
            for i in range(3): self.F[i, i+3] = dt
            self.H = np.hstack((np.eye(3), np.zeros((3, 3))))
            self.P = np.eye(6)
            self.Q = np.eye(6) * process_noise
            self.R = np.eye(3) * measurement_noise

        def update(self, z):
            z = np.reshape(z, (3, 1))
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q
            y = z - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x += K @ y
            self.P = (np.eye(6) - K @ self.H) @ self.P
            return self.x[:3].flatten()


class FaceTrackerGUI(QMainWindow):
    """Main window with video display and control panel."""

    # Number of face mesh landmarks (468 base + 10 iris with refinement)
    NUM_LANDMARKS = 478

    def __init__(self, camera_id=0, img_size=(1080, 720)):
        super().__init__()
        self.camera_id = camera_id
        self.img_size = img_size
        self.video_path = None
        self.use_webcam = True

        # Tracking state
        self.is_running = False
        self.apply_kalman = False
        self.save_dir = ""
        self.is_recording = False
        self.recorded_landmarks = []
        self.frame_count = 0

        # Initialize Kalman filters for all landmarks
        self.kalman_filters = [Kalman3D(process_noise=1e-3, measurement_noise=1e-4) 
                               for _ in range(self.NUM_LANDMARKS)]

        # Video capture
        self.cap = None

        # MediaPipe Face Mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.init_ui()
        self.setup_timer()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Face Tracker with Controls")
        self.setMinimumSize(1400, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Video display (left side)
        self.video_label = QLabel()
        self.video_label.setMinimumSize(960, 720)
        self.video_label.setStyleSheet("background-color: #1a1a1a; border: 2px solid #333;")
        self.video_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.video_label, stretch=3)

        # Control panel (right side)
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, stretch=1)

    def create_control_panel(self):
        """Create the right-side control panel."""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame { background-color: #2d2d2d; border-radius: 8px; }
            QLabel { color: #ffffff; font-size: 12px; }
            QGroupBox { 
                color: #ffffff; 
                font-weight: bold; 
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #1084d8; }
            QPushButton:pressed { background-color: #006abc; }
            QPushButton:disabled { background-color: #555; color: #888; }
            QCheckBox { color: #ffffff; }
            QRadioButton { color: #ffffff; }
            QLineEdit {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555;
                padding: 5px;
                border-radius: 3px;
            }
            QSpinBox {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555;
                padding: 5px;
            }
        """)
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # Title
        title = QLabel("Control Panel")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #0078d4;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # --- Source Selection ---
        source_group = QGroupBox("Video Source")
        source_layout = QVBoxLayout(source_group)

        self.webcam_radio = QRadioButton("Webcam")
        self.webcam_radio.setChecked(True)
        self.webcam_radio.toggled.connect(self.on_source_changed)
        source_layout.addWidget(self.webcam_radio)

        # Camera ID selector
        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("Camera ID:"))
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 10)
        self.camera_spin.setValue(self.camera_id if isinstance(self.camera_id, int) else 0)
        camera_layout.addWidget(self.camera_spin)
        source_layout.addLayout(camera_layout)

        self.video_radio = QRadioButton("Video File")
        self.video_radio.toggled.connect(self.on_source_changed)
        source_layout.addWidget(self.video_radio)

        # Video file path
        video_path_layout = QHBoxLayout()
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setPlaceholderText("Select video file...")
        self.video_path_edit.setEnabled(False)
        video_path_layout.addWidget(self.video_path_edit)

        self.browse_video_btn = QPushButton("Browse")
        self.browse_video_btn.setEnabled(False)
        self.browse_video_btn.clicked.connect(self.browse_video_file)
        video_path_layout.addWidget(self.browse_video_btn)
        source_layout.addLayout(video_path_layout)

        layout.addWidget(source_group)

        # --- Processing Options ---
        processing_group = QGroupBox("Processing")
        processing_layout = QVBoxLayout(processing_group)

        self.kalman_checkbox = QCheckBox("Enable Kalman Filter Smoothing")
        self.kalman_checkbox.toggled.connect(self.on_kalman_toggled)
        processing_layout.addWidget(self.kalman_checkbox)

        layout.addWidget(processing_group)

        # --- Data Recording ---
        recording_group = QGroupBox("Data Recording")
        recording_layout = QVBoxLayout(recording_group)

        # Save directory
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Save Directory:"))
        recording_layout.addLayout(dir_layout)

        dir_path_layout = QHBoxLayout()
        self.save_dir_edit = QLineEdit()
        self.save_dir_edit.setPlaceholderText("Select save directory...")
        self.save_dir_edit.textChanged.connect(self.on_save_dir_changed)
        dir_path_layout.addWidget(self.save_dir_edit)

        self.browse_dir_btn = QPushButton("Browse")
        self.browse_dir_btn.clicked.connect(self.browse_save_directory)
        dir_path_layout.addWidget(self.browse_dir_btn)
        recording_layout.addLayout(dir_path_layout)

        # Record button
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.setEnabled(False)
        self.record_btn.clicked.connect(self.toggle_recording)
        recording_layout.addWidget(self.record_btn)

        # Recording status
        self.record_status = QLabel("Not recording")
        self.record_status.setStyleSheet("color: #888;")
        recording_layout.addWidget(self.record_status)

        layout.addWidget(recording_group)

        # --- Start/Stop Controls ---
        control_group = QGroupBox("Tracking Control")
        control_layout = QVBoxLayout(control_group)

        self.start_btn = QPushButton("Start Tracking")
        self.start_btn.clicked.connect(self.toggle_tracking)
        control_layout.addWidget(self.start_btn)

        # Status label
        self.status_label = QLabel("Status: Stopped")
        self.status_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.status_label)

        # Frame counter
        self.frame_label = QLabel("Frames: 0")
        self.frame_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.frame_label)

        layout.addWidget(control_group)

        # --- Instructions ---
        instructions = QLabel("Press ESC or close window to quit")
        instructions.setStyleSheet("color: #666; font-size: 10px;")
        instructions.setAlignment(Qt.AlignCenter)
        layout.addWidget(instructions)

        # Spacer
        layout.addStretch()

        return panel

    def setup_timer(self):
        """Setup the frame update timer."""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def on_source_changed(self):
        """Handle source radio button changes."""
        self.use_webcam = self.webcam_radio.isChecked()
        self.camera_spin.setEnabled(self.use_webcam)
        self.video_path_edit.setEnabled(not self.use_webcam)
        self.browse_video_btn.setEnabled(not self.use_webcam)

        # Stop tracking if source changes
        if self.is_running:
            self.stop_tracking()

    def on_kalman_toggled(self, checked):
        """Handle Kalman filter toggle."""
        self.apply_kalman = checked
        if checked:
            # Reset filters when enabling
            self.reset_kalman_filters()

    def on_save_dir_changed(self, text):
        """Handle save directory text change."""
        self.save_dir = text
        self.record_btn.setEnabled(bool(text) and os.path.isdir(text))

    def browse_video_file(self):
        """Open file dialog to select video file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if file_path:
            self.video_path = file_path
            self.video_path_edit.setText(file_path)

    def browse_save_directory(self):
        """Open directory dialog to select save directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if dir_path:
            self.save_dir_edit.setText(dir_path)

    def toggle_recording(self):
        """Toggle landmark recording on/off."""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        """Start recording landmarks."""
        self.is_recording = True
        self.recorded_landmarks = []
        self.record_btn.setText("Stop Recording")
        self.record_btn.setStyleSheet("background-color: #d43333;")
        self.record_status.setText("Recording...")
        self.record_status.setStyleSheet("color: #ff4444;")

    def stop_recording(self):
        """Stop recording and save landmarks."""
        self.is_recording = False
        self.record_btn.setText("Start Recording")
        self.record_btn.setStyleSheet("")

        if self.recorded_landmarks and self.save_dir:
            self.save_landmarks()

        self.record_status.setText("Not recording")
        self.record_status.setStyleSheet("color: #888;")

    def save_landmarks(self):
        """Save recorded landmarks to CSV and NPZ files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"face_landmarks_{timestamp}"

        # Convert to numpy array
        landmarks_array = np.array(self.recorded_landmarks)

        # Save as NPZ
        npz_path = os.path.join(self.save_dir, f"{base_name}.npz")
        np.savez(npz_path, landmarks=landmarks_array)

        # Save as CSV
        csv_path = os.path.join(self.save_dir, f"{base_name}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            header = ['frame']
            for i in range(self.NUM_LANDMARKS):
                header.extend([f'lm{i}_x', f'lm{i}_y', f'lm{i}_z'])
            writer.writerow(header)
            # Data
            for frame_idx, frame_landmarks in enumerate(self.recorded_landmarks):
                row = [frame_idx]
                for lm in frame_landmarks:
                    row.extend([lm[0], lm[1], lm[2]])
                writer.writerow(row)

        self.record_status.setText(f"Saved: {base_name}")
        self.record_status.setStyleSheet("color: #44ff44;")
        print(f"Saved landmarks to:\n  {npz_path}\n  {csv_path}")

    def toggle_tracking(self):
        """Toggle tracking on/off."""
        if self.is_running:
            self.stop_tracking()
        else:
            self.start_tracking()

    def start_tracking(self):
        """Start video capture and tracking."""
        # Determine source
        if self.use_webcam:
            source = self.camera_spin.value()
        else:
            source = self.video_path
            if not source or not os.path.exists(source):
                self.status_label.setText("Status: Invalid video path")
                return

        # Open video capture
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.status_label.setText("Status: Cannot open source")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size[1])

        # Reset state
        self.frame_count = 0
        self.reset_kalman_filters()

        # Update UI
        self.is_running = True
        self.start_btn.setText("Stop Tracking")
        self.start_btn.setStyleSheet("background-color: #d43333;")
        self.status_label.setText("Status: Running")
        self.status_label.setStyleSheet("color: #44ff44;")

        # Start timer (30 FPS)
        self.timer.start(33)

    def stop_tracking(self):
        """Stop video capture and tracking."""
        self.timer.stop()

        if self.cap:
            self.cap.release()
            self.cap = None

        # Stop recording if active
        if self.is_recording:
            self.stop_recording()

        # Update UI
        self.is_running = False
        self.start_btn.setText("Start Tracking")
        self.start_btn.setStyleSheet("")
        self.status_label.setText("Status: Stopped")
        self.status_label.setStyleSheet("color: #ffffff;")

    def reset_kalman_filters(self):
        """Reset all Kalman filters."""
        self.kalman_filters = [Kalman3D(process_noise=1e-3, measurement_noise=1e-4) 
                               for _ in range(self.NUM_LANDMARKS)]

    def update_frame(self):
        """Process and display the next frame."""
        if not self.cap or not self.cap.isOpened():
            self.stop_tracking()
            return

        ret, frame = self.cap.read()
        if not ret:
            # End of video file
            if not self.use_webcam:
                self.stop_tracking()
                self.status_label.setText("Status: Video ended")
            return

        # Flip for webcam (mirror effect)
        if self.use_webcam:
            frame = cv2.flip(frame, 1)

        # Process frame with MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        landmarks_array = None

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract landmarks as numpy array
                raw_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])

                # Apply Kalman filtering if enabled
                if self.apply_kalman:
                    filtered_landmarks = np.array([
                        self.kalman_filters[i].update(raw_landmarks[i]) 
                        for i in range(len(raw_landmarks))
                    ])
                    landmarks_array = filtered_landmarks
                else:
                    landmarks_array = raw_landmarks

                # Draw mesh
                mp.solutions.drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
                )
                mp.solutions.drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
                )
                mp.solutions.drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style()
                )

                # Draw filtered landmarks if Kalman is enabled (as white dots)
                if self.apply_kalman and landmarks_array is not None:
                    h, w = frame.shape[:2]
                    for x, y, _ in landmarks_array[::10]:  # Draw every 10th for performance
                        cv2.circle(frame, (int(x * w), int(y * h)), 2, (255, 255, 255), -1)

        # Record landmarks if recording
        if self.is_recording and landmarks_array is not None:
            self.recorded_landmarks.append(landmarks_array.copy())

        # Draw overlay info
        self.frame_count += 1
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if self.apply_kalman:
            cv2.putText(frame, "Kalman: ON", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if self.is_recording:
            cv2.putText(frame, f"REC [{len(self.recorded_landmarks)}]", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        num_faces = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0
        cv2.putText(frame, f"Faces: {num_faces}", (frame.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Convert to Qt format and display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

        # Update frame counter
        self.frame_label.setText(f"Frames: {self.frame_count}")

    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key_Escape:
            self.close()

    def closeEvent(self, event):
        """Handle window close."""
        self.stop_tracking()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="Face Tracker with Control Panel")
    parser.add_argument('--camera_id', type=int, default=0, help='Default camera ID')
    parser.add_argument('--img_size', type=int, nargs=2, default=(1080, 720), help='Image size (width height)')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Apply dark theme
    palette = app.palette()
    from PyQt5.QtGui import QPalette, QColor
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(45, 45, 45))
    palette.setColor(QPalette.AlternateBase, QColor(30, 30, 30))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(0, 120, 212))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)

    window = FaceTrackerGUI(
        camera_id=args.camera_id,
        img_size=tuple(args.img_size)
    )
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
