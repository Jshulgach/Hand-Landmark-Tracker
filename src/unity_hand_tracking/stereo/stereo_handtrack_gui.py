"""
Stereo Hand Tracker with Control Panel and Joint Angle Broadcasting

Multi-camera hand tracking application with:
- Multiple camera feeds displayed simultaneously
- 3D triangulation from calibrated cameras
- Kalman filter smoothing on 3D landmarks
- Broadcasting landmarks on port 5005
- Broadcasting joint angles on port 5010
"""

import argparse
import sys
import os
import time
import socket
import json
import numpy as np
import cv2
import mediapipe as mp
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QCheckBox, QPushButton, QGroupBox, QSpinBox, QFrame, QScrollArea, QGridLayout
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor

# Import configuration and multi-camera tracker
from config import (
    CAMERA_IDS, NUM_CAMERAS, NUM_LANDMARKS, ANGLE_NAMES,
    KALMAN_3D_PROCESS_NOISE, KALMAN_3D_MEASUREMENT_NOISE,
    KALMAN_1D_PROCESS_NOISE, KALMAN_1D_MEASUREMENT_NOISE,
    UDP_IP, UDP_PORT_LANDMARKS, UDP_PORT_ANGLES, MAX_HANDS
)
from multi_camera_tracker import MultiCameraTracker


class Kalman3D:
    """3D Kalman filter for landmark smoothing."""
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


class Kalman1D:
    """1D Kalman filter for angle smoothing."""
    def __init__(self, dt=1/30, process_noise=0.1, measurement_noise=1.0):
        self.x = np.zeros((2, 1))
        self.F = np.array([[1, dt], [0, 1]])
        self.H = np.array([[1, 0]])
        self.P = np.eye(2)
        self.Q = np.eye(2) * process_noise
        self.R = np.array([[measurement_noise]])

    def update(self, z):
        z = np.array([[z]])
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x[0, 0]


def angle_between(v1, v2):
    """Return angle in degrees between vectors v1 and v2."""
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm < 1e-8 or v2_norm < 1e-8:
        return 0.0
    
    v1 = v1 / v1_norm
    v2 = v2 / v2_norm
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.degrees(np.arccos(dot))
    return angle


def finger_bend_angles(landmarks):
    """
    Calculate finger bend angles from 3D landmarks.
    landmarks: (21, 3) numpy array
    returns: dict of 14 bending angles (degrees)
    """
    def joint_angle(a, b, c):
        v1 = a - b
        v2 = c - b
        ang = angle_between(v1, v2)
        bend = 180.0 - ang
        return np.clip(bend, 0.0, 180.0)

    angles = {}
    fingers = {
        "index":  [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring":   [13, 14, 15, 16],
        "pinky":  [17, 18, 19, 20],
    }

    wrist = landmarks[0]

    for name, (mcp, pip, dip, tip) in fingers.items():
        angles[f"{name}_mcp"] = joint_angle(wrist, landmarks[mcp], landmarks[pip])
        angles[f"{name}_pip"] = joint_angle(landmarks[mcp], landmarks[pip], landmarks[dip])
        angles[f"{name}_dip"] = joint_angle(landmarks[pip], landmarks[dip], landmarks[tip])

    angles["thumb_cmc_mcp"] = joint_angle(landmarks[0], landmarks[2], landmarks[3])
    angles["thumb_ip"] = joint_angle(landmarks[2], landmarks[3], landmarks[4])

    return angles


class StereoHandTrackerGUI(QMainWindow):
    """Main window with multi-camera display and control panel."""

    def __init__(self):
        super().__init__()
        
        # Tracking state
        self.is_running = False
        self.apply_kalman = False
        self.frame_count = 0
        self.max_hands = MAX_HANDS

        # Multi-camera tracker
        self.tracker = None

        # UDP Broadcasting state
        self.udp_enabled = False
        self.udp_ip = UDP_IP
        self.udp_port_landmarks = UDP_PORT_LANDMARKS
        self.udp_port_angles = UDP_PORT_ANGLES
        self.udp_socket = None

        # Recording state
        self.is_recording = False

        # Initialize Kalman filters for landmarks (per hand)
        self.kalman_filters = [
            [Kalman3D(process_noise=KALMAN_3D_PROCESS_NOISE, 
                     measurement_noise=KALMAN_3D_MEASUREMENT_NOISE) 
             for _ in range(NUM_LANDMARKS)] 
            for _ in range(self.max_hands)
        ]

        # Initialize Kalman filters for angles (per hand)
        self.angle_kalman_filters = [
            {name: Kalman1D(process_noise=KALMAN_1D_PROCESS_NOISE, 
                           measurement_noise=KALMAN_1D_MEASUREMENT_NOISE) 
             for name in ANGLE_NAMES} 
            for _ in range(self.max_hands)
        ]

        self.init_ui()
        self.setup_timer()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle(f"Stereo Hand Tracker - {NUM_CAMERAS} Cameras")
        self.setMinimumSize(1600, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Video displays (left side) - grid layout for multiple cameras
        video_widget = QWidget()
        video_layout = QGridLayout(video_widget)
        video_layout.setSpacing(5)
        
        self.video_labels = []
        cols = 2 if NUM_CAMERAS <= 4 else 3
        rows = (NUM_CAMERAS + cols - 1) // cols
        
        for idx in range(NUM_CAMERAS):
            label = QLabel()
            label.setMinimumSize(640, 480)
            label.setStyleSheet("background-color: #1a1a1a; border: 2px solid #333;")
            label.setAlignment(Qt.AlignCenter)
            self.video_labels.append(label)
            
            row = idx // cols
            col = idx % cols
            video_layout.addWidget(label, row, col)
        
        main_layout.addWidget(video_widget, stretch=3)

        # Control panel (right side)
        control_panel = self.create_control_panel()
        scroll_area = QScrollArea()
        scroll_area.setWidget(control_panel)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background-color: #2d2d2d; }")
        main_layout.addWidget(scroll_area, stretch=1)

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

        # Camera info
        camera_group = QGroupBox("Camera Setup")
        camera_layout = QVBoxLayout(camera_group)
        camera_info = QLabel(f"Cameras: {NUM_CAMERAS}\nIDs: {CAMERA_IDS}")
        camera_info.setStyleSheet("color: #aaa;")
        camera_layout.addWidget(camera_info)
        layout.addWidget(camera_group)

        # Processing Options
        processing_group = QGroupBox("Processing")
        processing_layout = QVBoxLayout(processing_group)

        self.kalman_checkbox = QCheckBox("Enable Kalman Filter Smoothing")
        self.kalman_checkbox.toggled.connect(self.on_kalman_toggled)
        processing_layout.addWidget(self.kalman_checkbox)

        layout.addWidget(processing_group)

        # UDP Broadcasting
        udp_group = QGroupBox("UDP Broadcasting")
        udp_layout = QVBoxLayout(udp_group)

        self.udp_checkbox = QCheckBox("Enable UDP Broadcasting")
        self.udp_checkbox.toggled.connect(self.on_udp_toggled)
        udp_layout.addWidget(self.udp_checkbox)

        ports_info = QLabel(f"Landmarks: {UDP_PORT_LANDMARKS}\nAngles: {UDP_PORT_ANGLES}")
        ports_info.setStyleSheet("color: #aaa; font-size: 10px;")
        udp_layout.addWidget(ports_info)

        self.udp_status = QLabel("UDP: Disabled")
        self.udp_status.setStyleSheet("color: #888;")
        udp_layout.addWidget(self.udp_status)

        layout.addWidget(udp_group)

        # Data Recording Control
        recording_group = QGroupBox("Data Recording")
        recording_layout = QVBoxLayout(recording_group)

        self.record_checkbox = QCheckBox("Enable Recording")
        self.record_checkbox.setEnabled(False)  # Disabled until tracking starts
        self.record_checkbox.toggled.connect(self.on_recording_toggled)
        recording_layout.addWidget(self.record_checkbox)

        self.record_status = QLabel("Recording: Disabled")
        self.record_status.setStyleSheet("color: #888;")
        recording_layout.addWidget(self.record_status)

        layout.addWidget(recording_group)

        # Tracking Control
        control_group = QGroupBox("Tracking Control")
        control_layout = QVBoxLayout(control_group)

        self.start_btn = QPushButton("Start Tracking")
        self.start_btn.clicked.connect(self.toggle_tracking)
        control_layout.addWidget(self.start_btn)

        self.status_label = QLabel("Status: Stopped")
        self.status_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.status_label)

        self.frame_label = QLabel("Frames: 0")
        self.frame_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.frame_label)

        self.hands_label = QLabel("Hands: 0")
        self.hands_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.hands_label)

        layout.addWidget(control_group)

        # Instructions
        instructions = QLabel("Press ESC to quit")
        instructions.setStyleSheet("color: #666; font-size: 10px;")
        instructions.setAlignment(Qt.AlignCenter)
        layout.addWidget(instructions)

        layout.addStretch()
        return panel

    def setup_timer(self):
        """Setup the frame update timer."""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def on_kalman_toggled(self, checked):
        """Handle Kalman filter toggle."""
        self.apply_kalman = checked
        if checked:
            self.reset_kalman_filters()

    def on_udp_toggled(self, checked):
        """Handle UDP broadcasting toggle."""
        self.udp_enabled = checked
        if checked:
            self.start_udp_broadcast()
        else:
            self.stop_udp_broadcast()

    def on_recording_toggled(self, checked):
        """Handle recording toggle."""
        self.is_recording = checked
        if checked:
            self.record_status.setText("Recording: Active")
            self.record_status.setStyleSheet("color: #ff4444;")
        else:
            self.record_status.setText("Recording: Disabled")
            self.record_status.setStyleSheet("color: #888;")

    def reset_kalman_filters(self):
        """Reset all Kalman filters."""
        self.kalman_filters = [
            [Kalman3D(process_noise=KALMAN_3D_PROCESS_NOISE, 
                     measurement_noise=KALMAN_3D_MEASUREMENT_NOISE) 
             for _ in range(NUM_LANDMARKS)] 
            for _ in range(self.max_hands)
        ]
        self.angle_kalman_filters = [
            {name: Kalman1D(process_noise=KALMAN_1D_PROCESS_NOISE, 
                           measurement_noise=KALMAN_1D_MEASUREMENT_NOISE) 
             for name in ANGLE_NAMES} 
            for _ in range(self.max_hands)
        ]

    def start_udp_broadcast(self):
        """Initialize UDP socket for broadcasting."""
        try:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self.udp_status.setText("UDP: Active")
            self.udp_status.setStyleSheet("color: #44ff44;")
            print(f"UDP broadcast initialized: {self.udp_ip}:{UDP_PORT_LANDMARKS} (landmarks) and :{UDP_PORT_ANGLES} (angles)")
        except Exception as e:
            print(f"Error initializing UDP socket: {e}")
            self.udp_enabled = False
            self.udp_checkbox.setChecked(False)
            self.udp_status.setText(f"UDP: Error")
            self.udp_status.setStyleSheet("color: #ff4444;")

    def stop_udp_broadcast(self):
        """Close UDP socket."""
        if self.udp_socket:
            try:
                self.udp_socket.close()
            except:
                pass
            self.udp_socket = None
        self.udp_status.setText("UDP: Disabled")
        self.udp_status.setStyleSheet("color: #888;")

    def broadcast_landmarks(self, frame_landmarks, num_hands):
        """Broadcast hand landmarks via UDP."""
        if not self.udp_enabled or not self.udp_socket or not frame_landmarks:
            return

        try:
            data = {
                "frame": self.frame_count,
                "num_hands": num_hands,
                "timestamp": time.time(),
                "hands": []
            }

            for hand_idx, landmarks in enumerate(frame_landmarks):
                hand_data = {
                    "hand_index": hand_idx,
                    "landmarks": landmarks.tolist()
                }
                data["hands"].append(hand_data)

            message = json.dumps(data).encode('utf-8')
            self.udp_socket.sendto(message, (self.udp_ip, self.udp_port_landmarks))
        except Exception as e:
            print(f"Error broadcasting landmarks: {e}")

    def broadcast_joint_angles(self, hand_angles):
        """Broadcast joint angles via UDP."""
        if not self.udp_enabled or not self.udp_socket:
            return

        payload = {
            "frame": self.frame_count,
            "timestamp": time.time(),
            "recording" : self.is_recording,
            "hands": hand_angles
        }

        try:
            msg = json.dumps(payload).encode("utf-8")
            self.udp_socket.sendto(msg, (self.udp_ip, self.udp_port_angles))
        except Exception as e:
            print(f"Joint angle UDP error: {e}")

    def toggle_tracking(self):
        """Toggle tracking on/off."""
        if self.is_running:
            self.stop_tracking()
        else:
            self.start_tracking()

    def start_tracking(self):
        """Start multi-camera tracking."""
        try:
            # Initialize tracker
            self.tracker = MultiCameraTracker()
            
            if not self.tracker.initialize_cameras():
                self.status_label.setText("Status: Camera init failed")
                return

            # Reset state
            self.frame_count = 0
            self.reset_kalman_filters()

            # Update UI
            self.is_running = True
            self.start_btn.setText("Stop Tracking")
            self.start_btn.setStyleSheet("background-color: #d43333;")
            self.status_label.setText("Status: Running")
            self.status_label.setStyleSheet("color: #44ff44;")

            # Enable recording checkbox when tracking starts
            self.record_checkbox.setEnabled(True)

            # Start timer (30 FPS)
            self.timer.start(33)
            
        except Exception as e:
            self.status_label.setText(f"Status: Error - {str(e)}")
            print(f"Error starting tracker: {e}")

    def stop_tracking(self):
        """Stop tracking."""
        self.timer.stop()

        if self.tracker:
            self.tracker.cleanup()
            self.tracker = None

        # Update UI
        self.is_running = False
        self.start_btn.setText("Start Tracking")
        self.start_btn.setStyleSheet("")
        self.status_label.setText("Status: Stopped")
        self.status_label.setStyleSheet("color: #ffffff;")

        # Disable recording and checkbox when tracking stops
        self.is_recording = False
        self.record_checkbox.setChecked(False)
        self.record_checkbox.setEnabled(False)
        self.record_status.setText("Recording: Disabled")
        self.record_status.setStyleSheet("color: #888;")

    def update_frame(self):
        """Process and display frames from all cameras."""
        if not self.tracker:
            self.stop_tracking()
            return

        try:
            # Get frames and triangulated hands
            frames, triangulated_hands, all_results = self.tracker.process_frame()
            
            num_hands = len(triangulated_hands)
            
            # Process triangulated landmarks
            frame_landmarks = []
            hand_angle_packets = []
            
            for hand_idx, landmarks_3d in enumerate(triangulated_hands):
                # Apply Kalman filtering if enabled
                if self.apply_kalman and hand_idx < len(self.kalman_filters):
                    filtered_landmarks = np.array([
                        self.kalman_filters[hand_idx][i].update(landmarks_3d[i]) 
                        for i in range(len(landmarks_3d))
                    ])
                    landmarks_array = filtered_landmarks
                else:
                    landmarks_array = landmarks_3d
                
                frame_landmarks.append(landmarks_array)
                
                # Calculate joint angles
                joint_angles = finger_bend_angles(landmarks_array)
                
                # Apply Kalman filtering to angles if enabled
                if self.apply_kalman and hand_idx < len(self.angle_kalman_filters):
                    filtered_angles = {}
                    for angle_name, angle_value in joint_angles.items():
                        filtered_angles[angle_name] = self.angle_kalman_filters[hand_idx][angle_name].update(angle_value)
                    joint_angles = filtered_angles
                
                hand_angle_packets.append({
                    "hand_index": hand_idx,
                    "angles": joint_angles
                })
                
                # Debug output: print index finger angles for first hand (same line update)
                if hand_idx == 0:
                    print(f"\rFrame {self.frame_count} [3D Triangulated] - "
                          f"Index MCP: {joint_angles['index_mcp']:.2f}°, "
                          f"Index PIP: {joint_angles['index_pip']:.2f}°, "
                          f"Index DIP: {joint_angles['index_dip']:.2f}°", end='', flush=True)
            
            # Broadcast data
            if self.udp_enabled:
                if frame_landmarks:
                    self.broadcast_landmarks(frame_landmarks, num_hands)
                if hand_angle_packets:
                    self.broadcast_joint_angles(hand_angle_packets)
            
            # Display frames
            for idx, (frame, results) in enumerate(zip(frames, all_results)):
                if frame is None:
                    continue
                
                # Draw landmarks on each camera view
                if results and results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp.solutions.hands.HAND_CONNECTIONS,
                            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                            mp.solutions.drawing_styles.get_default_hand_connections_style()
                        )
                
                # Add camera label
                cv2.putText(frame, f"Camera {CAMERA_IDS[idx]}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Convert to Qt format and display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                    self.video_labels[idx].size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.video_labels[idx].setPixmap(scaled_pixmap)
            
            self.frame_count += 1
            self.frame_label.setText(f"Frames: {self.frame_count}")
            self.hands_label.setText(f"Hands: {num_hands}")
            
        except Exception as e:
            print(f"Error in update_frame: {e}")
            self.stop_tracking()

    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key_Escape:
            self.close()

    def closeEvent(self, event):
        """Handle window close."""
        self.stop_tracking()
        self.stop_udp_broadcast()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="Stereo Hand Tracker")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Apply dark theme
    palette = app.palette()
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

    window = StereoHandTrackerGUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()