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
    QLabel, QCheckBox, QPushButton, QGroupBox, QSpinBox, QFrame, QScrollArea, QGridLayout, QComboBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor

# Optional: use LSL local_clock for tighter time sync
try:
    from pylsl import local_clock
    HAS_LSL_CLOCK = True
except Exception:
    local_clock = None
    HAS_LSL_CLOCK = False
# Import configuration and multi-camera tracker
from config import (
    CAMERA_IDS, NUM_CAMERAS, NUM_LANDMARKS, ANGLE_NAMES,
    KALMAN_3D_PROCESS_NOISE, KALMAN_3D_MEASUREMENT_NOISE,
    KALMAN_1D_PROCESS_NOISE, KALMAN_1D_MEASUREMENT_NOISE,
    UDP_IP, UDP_PORT_LANDMARKS, UDP_PORT_ANGLES, MAX_HANDS
)

# Use new package modules
from handtrack.tracker.stereo import MultiCameraTracker
from handtrack.io.broadcast import UDPBroadcaster, LSLBroadcaster


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
        self.show_raw_video = True
        self.show_3d_view = False
        self.frame_count = 0
        self.max_hands = MAX_HANDS
        self.show_raw_filtered_overlay = False

        # Multi-camera tracker
        self.tracker = None

        # UDP Broadcasting state
        self.udp_enabled = False
        self.udp_ip = UDP_IP
        self.udp_ip = UDP_IP
        self.udp_port_landmarks = UDP_PORT_LANDMARKS
        self.udp_port_angles = UDP_PORT_ANGLES
        self.broadcaster = None
        self.lsl_broadcaster = None
        self._last_angle_log = 0.0
        self.use_lsl_clock = True
        self.hand_preference = "Any"
        self.remap_selected_hand = True

        # Kalman parameters (tunable)
        self.kalman_3d_process_noise = KALMAN_3D_PROCESS_NOISE
        self.kalman_3d_measurement_noise = KALMAN_3D_MEASUREMENT_NOISE
        self.kalman_1d_process_noise = KALMAN_1D_PROCESS_NOISE
        self.kalman_1d_measurement_noise = KALMAN_1D_MEASUREMENT_NOISE

        # Initialize Kalman filters for landmarks (per hand)
        self.kalman_filters = [
            [Kalman3D(process_noise=self.kalman_3d_process_noise, 
                     measurement_noise=self.kalman_3d_measurement_noise) 
             for _ in range(NUM_LANDMARKS)] 
            for _ in range(self.max_hands)
        ]

        # Initialize Kalman filters for angles (per hand)
        self.angle_kalman_filters = [
            {name: Kalman1D(process_noise=self.kalman_1d_process_noise, 
                           measurement_noise=self.kalman_1d_measurement_noise) 
             for name in ANGLE_NAMES} 
            for _ in range(self.max_hands)
        ]

        # Jitter tracking
        self.prev_raw_landmarks = None
        self.prev_filtered_landmarks = None
        self.raw_jitter_window = []
        self.filtered_jitter_window = []
        self.jitter_window_size = 30

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
        self.video_labels = []
        cols = 1  # Force single column for vertical stacking
        rows = NUM_CAMERAS
        
        for idx in range(NUM_CAMERAS):
            label = QLabel()
            label.setMinimumSize(480, 360) # Slightly smaller min size to fit vertically
            label.setStyleSheet("background-color: #1a1a1a; border: 2px solid #333;")
            label.setAlignment(Qt.AlignCenter)
            self.video_labels.append(label)
            
            video_layout.addWidget(label, idx, 0)
        
        main_layout.addWidget(video_widget, stretch=2)
        
        # 3D Visualization Widget (hidden by default)
        # 3D Visualization Widget (hidden by default)
        self.vis_3d_label = QLabel()
        self.vis_3d_label.setMinimumSize(600, 600)
        self.vis_3d_label.setStyleSheet("background-color: #000; border: 2px solid #555;")
        self.vis_3d_label.setAlignment(Qt.AlignCenter)
        self.vis_3d_label.hide()
        main_layout.addWidget(self.vis_3d_label, stretch=4)

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
        self.kalman_checkbox.setChecked(True)
        self.kalman_checkbox.toggled.connect(self.on_kalman_toggled)
        processing_layout.addWidget(self.kalman_checkbox)

        self.raw_video_checkbox = QCheckBox("Show Raw Video")
        self.raw_video_checkbox.setChecked(True)
        self.raw_video_checkbox.toggled.connect(self.on_raw_video_toggled)
        processing_layout.addWidget(self.raw_video_checkbox)
        
        self.show_3d_checkbox = QCheckBox("Show 3D View")
        self.show_3d_checkbox.setChecked(False)
        self.show_3d_checkbox.toggled.connect(self.on_show_3d_toggled)
        processing_layout.addWidget(self.show_3d_checkbox)

        layout.addWidget(processing_group)

        # Smoothing / Jitter
        smooth_group = QGroupBox("Smoothing")
        smooth_layout = QVBoxLayout(smooth_group)

        row_3d_p = QHBoxLayout()
        row_3d_p.addWidget(QLabel("3D Process"))
        self.k3d_p = QDoubleSpinBox()
        self.k3d_p.setDecimals(6)
        self.k3d_p.setRange(1e-6, 1.0)
        self.k3d_p.setSingleStep(1e-4)
        self.k3d_p.setValue(self.kalman_3d_process_noise)
        self.k3d_p.valueChanged.connect(self.on_kalman_params_changed)
        row_3d_p.addWidget(self.k3d_p)
        smooth_layout.addLayout(row_3d_p)

        row_3d_m = QHBoxLayout()
        row_3d_m.addWidget(QLabel("3D Meas"))
        self.k3d_m = QDoubleSpinBox()
        self.k3d_m.setDecimals(6)
        self.k3d_m.setRange(1e-6, 1.0)
        self.k3d_m.setSingleStep(1e-4)
        self.k3d_m.setValue(self.kalman_3d_measurement_noise)
        self.k3d_m.valueChanged.connect(self.on_kalman_params_changed)
        row_3d_m.addWidget(self.k3d_m)
        smooth_layout.addLayout(row_3d_m)

        row_1d_p = QHBoxLayout()
        row_1d_p.addWidget(QLabel("Angle Proc"))
        self.k1d_p = QDoubleSpinBox()
        self.k1d_p.setDecimals(6)
        self.k1d_p.setRange(1e-6, 10.0)
        self.k1d_p.setSingleStep(1e-3)
        self.k1d_p.setValue(self.kalman_1d_process_noise)
        self.k1d_p.valueChanged.connect(self.on_kalman_params_changed)
        row_1d_p.addWidget(self.k1d_p)
        smooth_layout.addLayout(row_1d_p)

        row_1d_m = QHBoxLayout()
        row_1d_m.addWidget(QLabel("Angle Meas"))
        self.k1d_m = QDoubleSpinBox()
        self.k1d_m.setDecimals(6)
        self.k1d_m.setRange(1e-6, 100.0)
        self.k1d_m.setSingleStep(1e-2)
        self.k1d_m.setValue(self.kalman_1d_measurement_noise)
        self.k1d_m.valueChanged.connect(self.on_kalman_params_changed)
        row_1d_m.addWidget(self.k1d_m)
        smooth_layout.addLayout(row_1d_m)

        self.jitter_label = QLabel("Jitter (raw/filtered): N/A")
        smooth_layout.addWidget(self.jitter_label)

        self.overlay_checkbox = QCheckBox("Overlay raw vs filtered (3D)")
        self.overlay_checkbox.toggled.connect(self.on_overlay_toggled)
        smooth_layout.addWidget(self.overlay_checkbox)

        btn_row = QHBoxLayout()
        self.kalman_reset_btn = QPushButton("Reset Defaults")
        self.kalman_reset_btn.clicked.connect(self.on_kalman_reset)
        self.kalman_save_btn = QPushButton("Save to config.py")
        self.kalman_save_btn.clicked.connect(self.on_kalman_save)
        btn_row.addWidget(self.kalman_reset_btn)
        btn_row.addWidget(self.kalman_save_btn)
        smooth_layout.addLayout(btn_row)

        layout.addWidget(smooth_group)

        # UDP Broadcasting
        udp_group = QGroupBox("UDP Broadcasting")
        udp_layout = QVBoxLayout(udp_group)

        self.udp_checkbox = QCheckBox("Enable UDP/LSL Broadcasting")
        self.udp_checkbox.toggled.connect(self.on_udp_toggled)
        udp_layout.addWidget(self.udp_checkbox)

        self.lsl_clock_checkbox = QCheckBox("Use LSL clock (tighter sync)")
        self.lsl_clock_checkbox.setChecked(True)
        self.lsl_clock_checkbox.toggled.connect(self.on_lsl_clock_toggled)
        self.lsl_clock_checkbox.setEnabled(HAS_LSL_CLOCK)
        udp_layout.addWidget(self.lsl_clock_checkbox)

        hand_pref_row = QHBoxLayout()
        hand_pref_label = QLabel("Broadcast Hand")
        self.hand_pref_combo = QComboBox()
        self.hand_pref_combo.addItems(["Any", "Left", "Right"])
        self.hand_pref_combo.currentTextChanged.connect(self.on_hand_pref_changed)
        hand_pref_row.addWidget(hand_pref_label)
        hand_pref_row.addWidget(self.hand_pref_combo)
        udp_layout.addLayout(hand_pref_row)

        self.remap_checkbox = QCheckBox("Remap selected hand to slot 0")
        self.remap_checkbox.setChecked(True)
        self.remap_checkbox.toggled.connect(self.on_remap_toggled)
        udp_layout.addWidget(self.remap_checkbox)

        ports_info = QLabel(f"Landmarks: {UDP_PORT_LANDMARKS}\nAngles: {UDP_PORT_ANGLES}")
        ports_info.setStyleSheet("color: #aaa; font-size: 10px;")
        udp_layout.addWidget(ports_info)

        self.lsl_info = QLabel("LSL: StereoHandTracker_Landmarks / _Angles")
        self.lsl_info.setStyleSheet("color: #aaa; font-size: 10px;")
        udp_layout.addWidget(self.lsl_info)

        self.udp_status = QLabel("Broadcasting: Disabled")
        self.udp_status.setStyleSheet("color: #888;")
        udp_layout.addWidget(self.udp_status)

        layout.addWidget(udp_group)

        # FPS Monitoring
        fps_group = QGroupBox("Performance Monitor")
        fps_layout = QVBoxLayout(fps_group)

        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet("color: #00ff00; font-weight: bold;")
        self.fps_label.setAlignment(Qt.AlignCenter)
        fps_layout.addWidget(self.fps_label)

        self.timing_label = QLabel("Capture: 0ms | Detect: 0ms | Tri: 0ms")
        self.timing_label.setStyleSheet("color: #aaa; font-size: 10px;")
        self.timing_label.setAlignment(Qt.AlignCenter)
        fps_layout.addWidget(self.timing_label)

        layout.addWidget(fps_group)

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

        self.tracked_hand_label = QLabel("Tracked Hand: N/A")
        self.tracked_hand_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.tracked_hand_label)

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

    def on_lsl_clock_toggled(self, checked):
        self.use_lsl_clock = checked and HAS_LSL_CLOCK

    def on_hand_pref_changed(self, value):
        self.hand_preference = value

    def on_remap_toggled(self, checked):
        self.remap_selected_hand = checked

    def on_raw_video_toggled(self, checked):
        """Handle raw video toggle."""
        self.show_raw_video = checked

    def on_show_3d_toggled(self, checked):
        """Handle 3D view toggle."""
        self.show_3d_view = checked
        if checked:
            self.vis_3d_label.show()
        else:
            self.vis_3d_label.hide()

    def on_overlay_toggled(self, checked):
        self.show_raw_filtered_overlay = checked

    def on_kalman_params_changed(self):
        self.kalman_3d_process_noise = float(self.k3d_p.value())
        self.kalman_3d_measurement_noise = float(self.k3d_m.value())
        self.kalman_1d_process_noise = float(self.k1d_p.value())
        self.kalman_1d_measurement_noise = float(self.k1d_m.value())
        if self.apply_kalman:
            self.reset_kalman_filters()

    def on_kalman_reset(self):
        self.k3d_p.setValue(KALMAN_3D_PROCESS_NOISE)
        self.k3d_m.setValue(KALMAN_3D_MEASUREMENT_NOISE)
        self.k1d_p.setValue(KALMAN_1D_PROCESS_NOISE)
        self.k1d_m.setValue(KALMAN_1D_MEASUREMENT_NOISE)
        self.on_kalman_params_changed()

    def on_kalman_save(self):
        try:
            config_path = os.path.join(os.path.dirname(__file__), "config.py")
            with open(config_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            def replace_value(key, value):
                for i, line in enumerate(lines):
                    if line.strip().startswith(f"{key} ="):
                        lines[i] = f"{key} = {value}\n"
            replace_value("KALMAN_3D_PROCESS_NOISE", self.kalman_3d_process_noise)
            replace_value("KALMAN_3D_MEASUREMENT_NOISE", self.kalman_3d_measurement_noise)
            replace_value("KALMAN_1D_PROCESS_NOISE", self.kalman_1d_process_noise)
            replace_value("KALMAN_1D_MEASUREMENT_NOISE", self.kalman_1d_measurement_noise)
            with open(config_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            self.status_label.setText("Status: Saved Kalman params")
        except Exception as exc:
            self.status_label.setText(f"Status: Save failed - {exc}")

    def reset_kalman_filters(self):
        """Reset all Kalman filters."""
        self.kalman_filters = [
            [Kalman3D(process_noise=self.kalman_3d_process_noise, 
                     measurement_noise=self.kalman_3d_measurement_noise) 
             for _ in range(NUM_LANDMARKS)] 
            for _ in range(self.max_hands)
        ]
        self.angle_kalman_filters = [
            {name: Kalman1D(process_noise=self.kalman_1d_process_noise, 
                           measurement_noise=self.kalman_1d_measurement_noise) 
             for name in ANGLE_NAMES} 
            for _ in range(self.max_hands)
        ]

    def start_udp_broadcast(self):
        """Initialize broadcaster (UDP + LSL)."""
        try:
            # We can use both or select one. For this demo, let's use the modular class
            # encapsulated in a way that handles both if needed, but here we just use UDP
            # and potentially LSL if available. 
            
            # Since the user requested LSL support, let's init both if possible or wrap them.
            # But wait, our `UDPBroadcaster` only does UDP. `LSLBroadcaster` only does LSL.
            # Let's use a composite approach or just init UDP for now to match legacy behavior
            # AND LSL if we want.
            
            self.broadcaster = UDPBroadcaster(self.udp_ip, self.udp_port_landmarks, self.udp_port_angles)
            
            # Optionally add LSL
            self.lsl_broadcaster = LSLBroadcaster(stream_name="StereoHandTracker")
            
            self.udp_status.setText("Broadcasting: Active (UDP+LSL)")
            self.udp_status.setStyleSheet("color: #44ff44;")
            
        except Exception as e:
            print(f"Error initializing broadcasters: {e}")
            self.udp_enabled = False
            self.udp_checkbox.setChecked(False)
            self.udp_status.setText(f"Error: {str(e)}")
            self.udp_status.setStyleSheet("color: #ff4444;")

    def stop_udp_broadcast(self):
        """Close broadcasters."""
        if self.broadcaster:
            self.broadcaster.close()
            self.broadcaster = None
        
        self.lsl_broadcaster = None # LSL outlets don't strictly need close(), garbage collection handles it
            
        self.udp_status.setText("Broadcasting: Disabled")
        self.udp_status.setStyleSheet("color: #888;")

    def broadcast_landmarks(self, frame_landmarks, num_hands):
        """Broadcast hand landmarks."""
        if not frame_landmarks:
            return
        if not (self.broadcaster or self.lsl_broadcaster):
            return

        # Prepare data format for broadcasters
        hands_data = []
        for hand_idx, landmarks in enumerate(frame_landmarks):
            hands_data.append({
                "hand_index": hand_idx,
                "landmarks": landmarks.tolist()
            })
            
        if self.broadcaster:
            self.broadcaster.send_landmarks(self.frame_count, self._broadcast_time(), hands_data)
        
        if self.lsl_broadcaster:
            self.lsl_broadcaster.send_landmarks(self.frame_count, self._broadcast_time(), hands_data)

    def broadcast_joint_angles(self, hand_angle_packets):
        """Broadcast joint angles."""
        if not hand_angle_packets:
            return
        if not (self.broadcaster or self.lsl_broadcaster):
            return

        # Prepare data format (already in packet list but needs harmonizing)
        # hand_angle_packets is list of dicts: {"hand_index": i, "angles": {...}}
        # UDPBroadcaster expects exactly this list of dicts.
        
        if self.broadcaster:
            self.broadcaster.send_angles(self.frame_count, self._broadcast_time(), hand_angle_packets)
            
        if self.lsl_broadcaster:
            self.lsl_broadcaster.send_angles(self.frame_count, self._broadcast_time(), hand_angle_packets)

    def _broadcast_time(self):
        if self.use_lsl_clock and HAS_LSL_CLOCK and local_clock:
            return local_clock()
        return time.time()

    def _handedness_labels(self, all_results):
        """
        Best-effort handedness labels from camera 0 MediaPipe results.
        Returns list aligned with camera0 hand order: ["Left"/"Right"...]
        """
        if not all_results or all_results[0] is None:
            return []
        result0 = all_results[0]
        if not result0.multi_handedness:
            return []
        labels = []
        for h in result0.multi_handedness:
            try:
                label = h.classification[0].label
                if label.lower() == "left":
                    label = "Right"
                elif label.lower() == "right":
                    label = "Left"
                labels.append(label)
            except Exception:
                labels.append("Unknown")
        return labels

    def _hand_matches_preference(self, label):
        if self.hand_preference == "Any":
            return True
        return label.lower() == self.hand_preference.lower()

    def toggle_tracking(self):
        """Toggle tracking on/off."""
        if self.is_running:
            self.stop_tracking()
        else:
            self.start_tracking()

    def start_tracking(self):
        """Start multi-camera tracking."""
        try:
            # Initialize tracker using package class
            # We pass in config values from our local config.py
            from config import (
                CALIBRATION_FILE, CAMERA_WIDTH, CAMERA_HEIGHT, 
                MIN_CAMERAS_FOR_TRIANGULATION, HAND_MATCH_THRESHOLD,
                ENABLE_PARALLEL_PROCESSING, NUM_WORKER_THREADS
            )
            
            self.tracker = MultiCameraTracker(
                camera_ids=CAMERA_IDS,
                calibration_file=CALIBRATION_FILE,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                num_landmarks=NUM_LANDMARKS,
                max_hands=MAX_HANDS,
                min_cameras=MIN_CAMERAS_FOR_TRIANGULATION,
                match_threshold=HAND_MATCH_THRESHOLD,
                enable_parallel=ENABLE_PARALLEL_PROCESSING,
                num_workers=NUM_WORKER_THREADS
            )
            
            if not self.tracker.initialize_cameras():
                self.status_label.setText("Status: Camera init failed")
                return

            # Ensure LSL outlet exists even if UDP is disabled
            if self.lsl_broadcaster is None:
                try:
                    self.lsl_broadcaster = LSLBroadcaster(stream_name="StereoHandTracker")
                except Exception as e:
                    print(f"Error initializing LSL broadcaster: {e}")

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

    def update_frame(self):
        """Process and display frames from all cameras."""
        if not self.tracker:
            self.stop_tracking()
            return

        try:
            # Get frames and triangulated hands
            frames, triangulated_hands, all_results = self.tracker.process_frame()
            
            num_hands = len(triangulated_hands)
            handedness_labels = self._handedness_labels(all_results)
            
            # Process triangulated landmarks
            frame_landmarks = []
            hand_angle_packets = []
            selected_idx = 0
            raw_hands = []
            filtered_hands = []
            
            tracked_label = "None"
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
                raw_hands.append(landmarks_3d)
                filtered_hands.append(landmarks_array)
                
                hand_label = handedness_labels[hand_idx] if hand_idx < len(handedness_labels) else "Unknown"
                if not self._hand_matches_preference(hand_label):
                    continue

                frame_landmarks.append(landmarks_array)
                if tracked_label == "None":
                    tracked_label = hand_label
                
                # Calculate joint angles
                joint_angles = finger_bend_angles(landmarks_array)
                
                # Apply Kalman filtering to angles if enabled
                if self.apply_kalman and hand_idx < len(self.angle_kalman_filters):
                    filtered_angles = {}
                    for angle_name, angle_value in joint_angles.items():
                        filtered_angles[angle_name] = self.angle_kalman_filters[hand_idx][angle_name].update(angle_value)
                    joint_angles = filtered_angles

                # Sanitize NaNs and log periodically
                nan_count = 0
                for k, v in list(joint_angles.items()):
                    if not np.isfinite(v):
                        nan_count += 1
                        joint_angles[k] = 0.0
                now = time.time()
                if hand_idx == 0 and now - self._last_angle_log > 2.0:
                    self._last_angle_log = now
                    print(
                        f"[Angles] nan={nan_count} sample={{'index_mcp': {joint_angles['index_mcp']:.2f}, "
                        f"'index_pip': {joint_angles['index_pip']:.2f}, 'index_dip': {joint_angles['index_dip']:.2f}}}"
                    )
                
                out_index = 0 if self.remap_selected_hand else selected_idx
                hand_angle_packets.append({
                    "hand_index": out_index,
                    "angles": joint_angles
                })
                selected_idx += 1
                
                # Debug output: print index finger angles for first hand (same line update)
                if hand_idx == 0:
                    print(f"\rFrame {self.frame_count} [3D Triangulated] - "
                          f"Index MCP: {joint_angles['index_mcp']:.2f}°, "
                          f"Index PIP: {joint_angles['index_pip']:.2f}°, "
                          f"Index DIP: {joint_angles['index_dip']:.2f}°", end='', flush=True)
            
            # Broadcast data
            if self.udp_enabled or self.lsl_broadcaster:
                if frame_landmarks:
                    self.broadcast_landmarks(frame_landmarks, num_hands)
                if hand_angle_packets:
                    self.broadcast_joint_angles(hand_angle_packets)
            
            # Display frames
            for idx, (frame, results) in enumerate(zip(frames, all_results)):
                if frame is None:
                    continue
                
                display_frame = frame.copy()
                
                if not self.show_raw_video:
                    # Create black background
                    display_frame[:] = 0
                
                # Draw landmarks on each camera view
                if results and results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(
                            display_frame,
                            hand_landmarks,
                            mp.solutions.hands.HAND_CONNECTIONS,
                            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                            mp.solutions.drawing_styles.get_default_hand_connections_style()
                        )
                
                # Add camera label
                cv2.putText(display_frame, f"Camera {CAMERA_IDS[idx]}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Convert to Qt format and display
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                    self.video_labels[idx].size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.video_labels[idx].setPixmap(scaled_pixmap)
            
            # Update jitter meter (hand 0)
            if raw_hands:
                raw0 = raw_hands[0]
                filt0 = filtered_hands[0]
                if self.prev_raw_landmarks is not None:
                    raw_speed = np.linalg.norm(raw0 - self.prev_raw_landmarks, axis=1).mean()
                    self.raw_jitter_window.append(raw_speed)
                    if len(self.raw_jitter_window) > self.jitter_window_size:
                        self.raw_jitter_window = self.raw_jitter_window[-self.jitter_window_size:]
                if self.prev_filtered_landmarks is not None:
                    filt_speed = np.linalg.norm(filt0 - self.prev_filtered_landmarks, axis=1).mean()
                    self.filtered_jitter_window.append(filt_speed)
                    if len(self.filtered_jitter_window) > self.jitter_window_size:
                        self.filtered_jitter_window = self.filtered_jitter_window[-self.jitter_window_size:]
                self.prev_raw_landmarks = raw0
                self.prev_filtered_landmarks = filt0

                if self.raw_jitter_window and self.filtered_jitter_window:
                    raw_std = np.std(self.raw_jitter_window)
                    filt_std = np.std(self.filtered_jitter_window)
                    self.jitter_label.setText(f"Jitter (raw/filtered): {raw_std:.4f} / {filt_std:.4f}")

            # Update 3D visualization if enabled
            if self.show_3d_view and triangulated_hands:
                self.update_3d_view(raw_hands, filtered_hands)
            
            self.frame_count += 1
            self.frame_label.setText(f"Frames: {self.frame_count}")
            self.hands_label.setText(f"Hands: {num_hands}")
            self.tracked_hand_label.setText(f"Tracked Hand: {tracked_label}")
            
            # Update FPS display
            if self.tracker:
                fps_stats = self.tracker.get_fps_stats()
                fps = fps_stats.get("fps", 0)
                capture_ms = fps_stats.get("capture_ms", 0)
                detection_ms = fps_stats.get("detection_ms", 0)
                tri_ms = fps_stats.get("triangulation_ms", 0)
                
                # Color code FPS: green if >25, yellow if 15-25, red if <15
                if fps > 25:
                    fps_color = "#00ff00"
                elif fps > 15:
                    fps_color = "#ffff00"
                else:
                    fps_color = "#ff0000"
                
                self.fps_label.setText(f"FPS: {fps:.1f}")
                self.fps_label.setStyleSheet(f"color: {fps_color}; font-weight: bold;")
                self.timing_label.setText(
                    f"Capture: {capture_ms:.1f}ms | Detect: {detection_ms:.1f}ms | Tri: {tri_ms:.1f}ms"
                )
            
        except Exception as e:
            print(f"Error in update_frame: {e}")
            self.stop_tracking()

    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key_Escape:
            self.close()

    def update_3d_view(self, raw_hands, filtered_hands):
        """Render simple 3D visualization using OpenCV drawing on a 2D plane (Top, Front, Side views)."""
        # Canvas size
        w, h = 600, 600
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Colors for different fingers
        colors = [
            (0, 0, 255),    # Thumb - Red
            (0, 255, 255),  # Index - Yellow
            (0, 255, 0),    # Middle - Green
            (255, 255, 0),  # Ring - Cyan
            (255, 0, 255)   # Pinky - Magenta
        ]
        
        # Define viewports
        # Top Left: Top View (X-Z plane)
        # Bottom Left: Front View (X-Y plane)
        # Bottom Right: Side View (Z-Y plane)
        
        half_w, half_h = w // 2, h // 2
        
        # Draw dividers
        cv2.line(canvas, (half_w, 0), (half_w, h), (50, 50, 50), 1)
        cv2.line(canvas, (0, half_h), (w, half_h), (50, 50, 50), 1)
        
        # Labels
        cv2.putText(canvas, "Top View (XZ)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(canvas, "Front View (XY)", (10, half_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(canvas, "Side View (ZY)", (half_w + 10, half_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Scaling factor for visualization (points are in mm, need to fit canvas)
        scale = 1.5  # pixels per mm
        center_offset = 150 # mm offset to center in viewport
        
        for idx, hand_landmarks in enumerate(filtered_hands):
            if hand_landmarks is None or len(hand_landmarks) == 0:
                continue
                
            # Connections for hand skeleton
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),       # Index
                (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                (0, 13), (13, 14), (14, 15), (15, 16),# Ring
                (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
            ]
            
            # Helper to draw skeletal lines
            def draw_skeleton(view_type, offset_x, offset_y, points, color):
                for start_idx, end_idx in connections:
                    p1 = points[start_idx]
                    p2 = points[end_idx]
                    
                    if np.all(p1 == 0) or np.all(p2 == 0): continue
                    
                    if view_type == 'top': # X, Z
                         u1, v1 = int(p1[0] * scale) + offset_x + 150, int(p1[2] * scale) + offset_y + 150
                         u2, v2 = int(p2[0] * scale) + offset_x + 150, int(p2[2] * scale) + offset_y + 150
                    elif view_type == 'front': # X, -Y (Y is down in CV, but up in 3D usually? MP Y is down. Let's assume standard)
                         u1, v1 = int(p1[0] * scale) + offset_x + 150, int(p1[1] * scale) + offset_y + 50
                         u2, v2 = int(p2[0] * scale) + offset_x + 150, int(p2[1] * scale) + offset_y + 50
                    elif view_type == 'side': # Z, Y
                         u1, v1 = int(p1[2] * scale) + offset_x + 150, int(p1[1] * scale) + offset_y + 50
                         u2, v2 = int(p2[2] * scale) + offset_x + 150, int(p2[1] * scale) + offset_y + 50
                    
                    cv2.line(canvas, (u1, v1), (u2, v2), color, 1)
                    cv2.circle(canvas, (u1, v1), 2, color, -1)

            # Draw views
            draw_skeleton('top', 0, 0, hand_landmarks, (0, 255, 0))
            draw_skeleton('front', 0, half_h, hand_landmarks, (0, 255, 0))
            draw_skeleton('side', half_w, half_h, hand_landmarks, (0, 255, 0))

            if self.show_raw_filtered_overlay and idx < len(raw_hands):
                raw_points = raw_hands[idx]
                draw_skeleton('top', 0, 0, raw_points, (120, 120, 120))
                draw_skeleton('front', 0, half_h, raw_points, (120, 120, 120))
                draw_skeleton('side', half_w, half_h, raw_points, (120, 120, 120))

        # Display on Qt Label
        rgb_frame = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.vis_3d_label.setPixmap(QPixmap.fromImage(qt_image))

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
