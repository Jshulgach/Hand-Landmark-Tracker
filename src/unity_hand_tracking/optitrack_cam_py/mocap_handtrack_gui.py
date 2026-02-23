"""
Stereo Hand Tracker with Control Panel and Joint Angle Broadcasting
Multi-camera hand tracking application with:
- OptiTrack cameras via CameraManager (multi_mjpeg.py)
- Calibration loaded from our .npz calibration files
- Threaded MediaPipe hand detection per camera
- 3D triangulation from calibrated cameras
- Kalman filter smoothing on 3D landmarks
- Broadcasting landmarks on port 5005
- Broadcasting joint angles on port 5010
"""

import argparse
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QImage, QPalette, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# Optional: use LSL local_clock for tighter time sync
try:
    from pylsl import local_clock

    HAS_LSL_CLOCK = True
except Exception:
    local_clock = None
    HAS_LSL_CLOCK = False

# Add parent directory to path for handtrack imports
sys.path.insert(0, "../../src/")

# Import configuration
from config import (
    ANGLE_NAMES,
    KALMAN_1D_MEASUREMENT_NOISE,
    KALMAN_1D_PROCESS_NOISE,
    KALMAN_3D_MEASUREMENT_NOISE,
    KALMAN_3D_PROCESS_NOISE,
    MAX_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    NUM_LANDMARKS,
    TRIANGULATION_METHOD,
    UDP_IP,
    UDP_PORT_ANGLES,
    UDP_PORT_LANDMARKS,
)

# Local tracker (uses CameraManager + threaded MediaPipe)
from mocap_tracker import MultiCameraTracker

# Broadcast helpers
try:
    from broadcast import LSLBroadcaster, UDPBroadcaster
except ImportError:
    print("Error: Could not import from broadcast module")
    print("Make sure you're running from the correct directory")
    sys.exit(1)


# =============================================================================
# Kalman Filters
# =============================================================================
class Kalman3D:
    """3D Kalman filter for landmark smoothing."""

    def __init__(self, dt=1 / 30, process_noise=1e-3, measurement_noise=1e-2):
        self.x = np.zeros((6, 1))
        self.F = np.eye(6)
        for i in range(3):
            self.F[i, i + 3] = dt
        self.H = np.hstack((np.eye(3), np.zeros((3, 3))))
        self.P = np.eye(6)
        self.Q = np.eye(6) * process_noise
        self.R = np.eye(3) * measurement_noise

    def update(self, z):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        if z is not None and not (isinstance(z, np.ndarray) and np.all(z == 0)):
            z = np.reshape(z, (3, 1))
            y = z - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x += K @ y
            self.P = (np.eye(6) - K @ self.H) @ self.P

        return self.x[:3].flatten()


class Kalman1D:
    """1D Kalman filter for angle smoothing."""

    def __init__(self, dt=1 / 30, process_noise=0.1, measurement_noise=1.0):
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


# =============================================================================
# Joint-angle helpers
# =============================================================================
def angle_between(v1, v2):
    """Return angle in degrees between vectors v1 and v2."""
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm < 1e-8 or v2_norm < 1e-8:
        return 0.0
    v1 = v1 / v1_norm
    v2 = v2 / v2_norm
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def finger_bend_angles(landmarks):
    """
    Calculate finger bend angles from 3D landmarks.
    landmarks: (21, 3) numpy array
    returns: dict of bending angles (degrees)
    """

    def joint_angle(a, b, c):
        v1 = a - b
        v2 = c - b
        ang = angle_between(v1, v2)
        bend = 180.0 - ang
        return np.clip(bend, 0.0, 180.0)

    angles = {}
    fingers = {
        "index": [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring": [13, 14, 15, 16],
        "pinky": [17, 18, 19, 20],
    }
    wrist = landmarks[0]
    for name, (mcp, pip, dip, tip) in fingers.items():
        angles[f"{name}_mcp"] = joint_angle(wrist, landmarks[mcp], landmarks[pip])
        angles[f"{name}_pip"] = joint_angle(
            landmarks[mcp], landmarks[pip], landmarks[dip]
        )
        angles[f"{name}_dip"] = joint_angle(
            landmarks[pip], landmarks[dip], landmarks[tip]
        )
    # Thumb landmarks: 0(wrist) -> 1(CMC) -> 2(MCP) -> 3(IP) -> 4(tip)
    # Two distal bend angles matching Unity's Proximal and Distal joints
    # Use key name that matches broadcast.py: thumb_cmc_mcp
    angles["thumb_cmc_mcp"] = joint_angle(landmarks[1], landmarks[2], landmarks[3])
    angles["thumb_ip"] = joint_angle(landmarks[2], landmarks[3], landmarks[4])
    return angles


def finger_splay_angles(landmarks):
    """
    Calculate finger splay angles from 3D landmarks.
    Simple approach: signed angle between wrist->TIP and wrist->middle_TIP reference.
    landmarks: (21, 3) numpy array
    returns: dict of splay angles (degrees) for each finger's MCP
    """
    wrist = landmarks[0]

    # Reference: wrist -> middle TIP
    ref = landmarks[12] - wrist
    ref_norm = np.linalg.norm(ref)
    if ref_norm < 1e-8:
        return {}
    ref = ref / ref_norm

    # Rough palm normal for sign (cross of wrist->index_tip and wrist->pinky_tip)
    v1 = landmarks[8] - wrist
    v2 = landmarks[20] - wrist
    palm_normal = np.cross(v1, v2)
    norm = np.linalg.norm(palm_normal)
    if norm < 1e-8:
        return {}
    palm_normal = palm_normal / norm

    tips = {
        "index": landmarks[8],
        "middle": landmarks[12],
        "ring": landmarks[16],
        "pinky": landmarks[20],
    }

    # Unity rest offsets (what Unity expects at neutral)
    rest_offsets = {
        "index": 0.0,
        "middle": 0.0,
        "ring": 0.0,
        "pinky": -15.383,
    }

    angles = {}
    for name, tip in tips.items():
        finger_vec = tip - wrist
        finger_norm = np.linalg.norm(finger_vec)
        if finger_norm < 1e-8:
            angles[f"{name}_splay"] = rest_offsets[name]
            continue
        finger_vec = finger_vec / finger_norm

        # Unsigned angle
        dot = np.clip(np.dot(finger_vec, ref), -1.0, 1.0)
        angle = np.degrees(np.arccos(dot))

        # Sign: clockwise = negative, counterclockwise = positive
        cross = np.cross(ref, finger_vec)
        if np.dot(cross, palm_normal) < 0:
            angle = -angle

        angles[f"{name}_splay"] = angle + rest_offsets[name]

    return angles


# =============================================================================
# GUI
# =============================================================================
class StereoHandTrackerGUI(QMainWindow):
    """Main window with multi-camera display and control panel."""

    def __init__(self):
        super().__init__()
        # Tracking state
        self.is_running = False
        self.apply_kalman = True
        self.show_raw_video = True
        self.show_3d_view = False
        self.frame_count = 0
        self.max_hands = MAX_HANDS
        self.show_raw_filtered_overlay = False
        # Multi-camera tracker
        self.tracker = None
        self.num_cameras = 0
        # Per-camera exposure (default 10 each, max 6 cameras)
        self.default_exposure = 10
        self.exposure_spinboxes: list[QSpinBox] = []
        # Per-camera triangulation enable (all on by default)
        self.cam_enabled_checkboxes: list[QCheckBox] = []
        self.enabled_cameras: set[int] = set()  # Will be populated dynamically
        # UDP Broadcasting state
        self.udp_enabled = True  # Auto-enable broadcasting on startup
        self.udp_ip = UDP_IP
        self.udp_port_landmarks = UDP_PORT_LANDMARKS
        self.udp_port_angles = UDP_PORT_ANGLES
        self.broadcaster = None
        self.lsl_broadcaster = None
        self._last_angle_log = 0.0
        self.use_lsl_clock = True
        self.hand_preference = "Any"
        self.remap_selected_hand = True
        # Debug flag for LSL stream info
        self._debug_stream_logged = False
        # Kalman parameters (tunable)
        self.kalman_3d_process_noise = KALMAN_3D_PROCESS_NOISE
        self.kalman_3d_measurement_noise = KALMAN_3D_MEASUREMENT_NOISE
        self.kalman_1d_process_noise = KALMAN_1D_PROCESS_NOISE
        self.kalman_1d_measurement_noise = KALMAN_1D_MEASUREMENT_NOISE
        # Initialize Kalman filters for landmarks (per hand)
        self.kalman_filters = [
            [
                Kalman3D(
                    process_noise=self.kalman_3d_process_noise,
                    measurement_noise=self.kalman_3d_measurement_noise,
                )
                for _ in range(NUM_LANDMARKS)
            ]
            for _ in range(self.max_hands)
        ]
        # Initialize Kalman filters for angles (per hand)
        self.angle_kalman_filters = [
            {
                name: Kalman1D(
                    process_noise=self.kalman_1d_process_noise,
                    measurement_noise=self.kalman_1d_measurement_noise,
                )
                for name in ANGLE_NAMES
            }
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
        # Auto-start broadcasting after UI is initialized
        self.start_udp_broadcast()

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Stereo Hand Tracker")
        self.setMinimumSize(1600, 900)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        # Video displays (left side) - dynamic grid
        video_widget = QWidget()
        self.video_layout = QGridLayout(video_widget)
        self.video_layout.setSpacing(5)
        self.video_labels = []
        # We will populate video_labels dynamically when cameras are initialized
        main_layout.addWidget(video_widget, stretch=2)
        # 3D Visualization Widget (hidden by default)
        self.vis_3d_label = QLabel()
        self.vis_3d_label.setMinimumSize(600, 600)
        self.vis_3d_label.setStyleSheet(
            "background-color: #000; border: 2px solid #555;"
        )
        self.vis_3d_label.setAlignment(Qt.AlignCenter)
        self.vis_3d_label.hide()
        main_layout.addWidget(self.vis_3d_label, stretch=4)
        # Control panel (right side)
        control_panel = self.create_control_panel()
        scroll_area = QScrollArea()
        scroll_area.setWidget(control_panel)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(
            "QScrollArea { border: none; background-color: #2d2d2d; }"
        )
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
            QSpinBox, QDoubleSpinBox {
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
        # ---- Camera Setup group ----
        camera_group = QGroupBox("Camera Setup")
        camera_layout = QVBoxLayout(camera_group)
        self.camera_info_label = QLabel("Cameras: \u2014")
        self.camera_info_label.setStyleSheet("color: #aaa;")
        camera_layout.addWidget(self.camera_info_label)
        # Per-camera exposure controls
        exp_title = QLabel("Per-Camera Exposure & Triangulation")
        exp_title.setStyleSheet("color: #ccc; font-weight: bold; margin-top: 6px;")
        camera_layout.addWidget(exp_title)
        self.exposure_spinboxes = []
        self.cam_enabled_checkboxes = []
        self.exposure_grid = QGridLayout()
        self.exposure_grid.setSpacing(4)
        # Header row
        hdr_use = QLabel("Tri")
        hdr_use.setStyleSheet("color: #888; font-size: 10px;")
        hdr_use.setAlignment(Qt.AlignCenter)
        self.exposure_grid.addWidget(hdr_use, 0, 0)
        hdr_cam = QLabel("Camera")
        hdr_cam.setStyleSheet("color: #888; font-size: 10px;")
        self.exposure_grid.addWidget(hdr_cam, 0, 1)
        hdr_exp = QLabel("Exposure")
        hdr_exp.setStyleSheet("color: #888; font-size: 10px;")
        self.exposure_grid.addWidget(hdr_exp, 0, 2)
        # We will populate the grid dynamically when cameras are initialized
        camera_layout.addLayout(self.exposure_grid)
        # "Set all" convenience row
        apply_all_row = QHBoxLayout()
        apply_all_label = QLabel("Set all:")
        apply_all_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self.exposure_all_spin = QSpinBox()
        self.exposure_all_spin.setRange(1, 480)
        self.exposure_all_spin.setValue(self.default_exposure)
        self.exposure_all_spin.setSuffix(" ms")
        apply_all_btn = QPushButton("Apply")
        apply_all_btn.setStyleSheet(
            "padding: 4px 10px; font-size: 11px; background-color: #555;"
        )
        apply_all_btn.clicked.connect(self._on_apply_all_exposure)
        apply_all_row.addWidget(apply_all_label)
        apply_all_row.addWidget(self.exposure_all_spin)
        apply_all_row.addWidget(apply_all_btn)
        camera_layout.addLayout(apply_all_row)
        layout.addWidget(camera_group)
        # ---- Processing Options ----
        processing_group = QGroupBox("Processing")
        processing_layout = QVBoxLayout(processing_group)

        # MediaPipe Confidences
        mp_layout = QGridLayout()
        mp_layout.addWidget(QLabel("Min Detection Conf:"), 0, 0)
        self.min_det_spin = QDoubleSpinBox()
        self.min_det_spin.setRange(0.1, 1.0)
        self.min_det_spin.setSingleStep(0.1)
        self.min_det_spin.setValue(MIN_DETECTION_CONFIDENCE)
        self.min_det_spin.valueChanged.connect(self.on_mp_params_changed)
        mp_layout.addWidget(self.min_det_spin, 0, 1)

        mp_layout.addWidget(QLabel("Min Tracking Conf:"), 1, 0)
        self.min_track_spin = QDoubleSpinBox()
        self.min_track_spin.setRange(0.1, 1.0)
        self.min_track_spin.setSingleStep(0.1)
        self.min_track_spin.setValue(MIN_TRACKING_CONFIDENCE)
        self.min_track_spin.valueChanged.connect(self.on_mp_params_changed)
        mp_layout.addWidget(self.min_track_spin, 1, 1)

        mp_layout.addWidget(QLabel("Occlusion Threshold:"), 2, 0)
        self.occlusion_spin = QDoubleSpinBox()
        self.occlusion_spin.setRange(5.0, 200.0)
        self.occlusion_spin.setSingleStep(5.0)
        # We'll set the value later when tracker is initialized, or use config default
        from config import MAX_REPROJECTION_ERROR
        self.occlusion_spin.setValue(MAX_REPROJECTION_ERROR)
        self.occlusion_spin.valueChanged.connect(self.on_occlusion_changed)
        mp_layout.addWidget(self.occlusion_spin, 2, 1)

        processing_layout.addLayout(mp_layout)

        self.kalman_checkbox = QCheckBox("Enable Kalman Filter Smoothing")
        self.kalman_checkbox.toggled.connect(self.on_kalman_toggled)
        self.kalman_checkbox.setChecked(True)
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
        # ---- Smoothing / Jitter ----
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
        # ---- UDP Broadcasting ----
        udp_group = QGroupBox("UDP Broadcasting")
        udp_layout = QVBoxLayout(udp_group)
        self.udp_checkbox = QCheckBox("Enable UDP/LSL Broadcasting")
        self.udp_checkbox.setChecked(True)  # Auto-enable on startup
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
        ports_info = QLabel(
            f"Landmarks: {UDP_PORT_LANDMARKS}\nAngles: {UDP_PORT_ANGLES}"
        )
        ports_info.setStyleSheet("color: #aaa; font-size: 10px;")
        udp_layout.addWidget(ports_info)
        self.lsl_info = QLabel("LSL: StereoHandTracker_Landmarks / _Angles")
        self.lsl_info.setStyleSheet("color: #aaa; font-size: 10px;")
        udp_layout.addWidget(self.lsl_info)
        self.udp_status = QLabel("Broadcasting: Disabled")
        self.udp_status.setStyleSheet("color: #888;")
        udp_layout.addWidget(self.udp_status)
        layout.addWidget(udp_group)
        # ---- FPS Monitoring ----
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
        # ---- Tracking Control ----
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

    # ------------------------------------------------------------------ #
    # Exposure callbacks
    # ------------------------------------------------------------------ #
    def _on_exposure_changed(self, cam_index, value):
        """Called when an individual camera exposure spinbox changes."""
        if self.tracker and cam_index < self.num_cameras:
            self.tracker.set_exposure(cam_index, value)
            print(f"[GUI] Camera {cam_index} exposure -> {value}")

    def _on_apply_all_exposure(self):
        """Set all camera exposure spinboxes to the 'Set all' value."""
        val = self.exposure_all_spin.value()
        for spin in self.exposure_spinboxes:
            spin.setValue(val)  # triggers _on_exposure_changed per camera

    def _on_cam_enabled_toggled(self, cam_index, checked):
        """Called when a per-camera triangulation checkbox changes."""
        if checked:
            self.enabled_cameras.add(cam_index)
        else:
            self.enabled_cameras.discard(cam_index)
        # Push to tracker
        if self.tracker:
            self.tracker.set_enabled_cameras(set(self.enabled_cameras))
        active = sorted(self.enabled_cameras)
        print(f"[GUI] Triangulation cameras: {active}")

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    def on_kalman_toggled(self, checked):
        self.apply_kalman = checked
        if checked:
            self.reset_kalman_filters()

    def on_udp_toggled(self, checked):
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
        self.show_raw_video = checked

    def on_show_3d_toggled(self, checked):
        self.show_3d_view = checked
        if checked:
            self.vis_3d_label.show()
        else:
            self.vis_3d_label.hide()

    def on_overlay_toggled(self, checked):
        self.show_raw_filtered_overlay = checked

    def on_mp_params_changed(self):
        if self.tracker:
            self.tracker.update_mp_params(
                self.min_det_spin.value(), self.min_track_spin.value()
            )

    def on_occlusion_changed(self):
        if self.tracker:
            self.tracker.max_reprojection_error = self.occlusion_spin.value()

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
            replace_value(
                "KALMAN_3D_MEASUREMENT_NOISE", self.kalman_3d_measurement_noise
            )
            replace_value("KALMAN_1D_PROCESS_NOISE", self.kalman_1d_process_noise)
            replace_value(
                "KALMAN_1D_MEASUREMENT_NOISE", self.kalman_1d_measurement_noise
            )
            with open(config_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            self.status_label.setText("Status: Saved Kalman params")
        except Exception as exc:
            self.status_label.setText(f"Status: Save failed - {exc}")

    def reset_kalman_filters(self):
        """Reset all Kalman filters."""
        self.kalman_filters = [
            [
                Kalman3D(
                    process_noise=self.kalman_3d_process_noise,
                    measurement_noise=self.kalman_3d_measurement_noise,
                )
                for _ in range(NUM_LANDMARKS)
            ]
            for _ in range(self.max_hands)
        ]
        self.angle_kalman_filters = [
            {
                name: Kalman1D(
                    process_noise=self.kalman_1d_process_noise,
                    measurement_noise=self.kalman_1d_measurement_noise,
                )
                for name in ANGLE_NAMES
            }
            for _ in range(self.max_hands)
        ]

    # ------------------------------------------------------------------ #
    # Broadcasting
    # ------------------------------------------------------------------ #
    def start_udp_broadcast(self):
        """Initialize broadcaster (UDP + LSL)."""
        try:
            self.broadcaster = UDPBroadcaster(
                self.udp_ip, self.udp_port_landmarks, self.udp_port_angles
            )
            self.lsl_broadcaster = LSLBroadcaster(stream_name="StereoHandTracker")
            self.udp_status.setText("Broadcasting: Active (UDP+LSL)")
            self.udp_status.setStyleSheet("color: #44ff44;")
            # Print LSL stream info
            print("\n" + "=" * 70)
            print("LSL STREAM INFORMATION")
            print("=" * 70)
            print("Landmarks Stream: StereoHandTracker_Landmarks")
            print("  - Type: MOCAP")
            print("  - Channels: 126 (2 hands × 21 landmarks × 3 coords)")
            print("  - Format: float32")
            print("  - Sample Rate: Irregular (0 Hz)")
            print("\nAngles Stream: StereoHandTracker_Angles")
            print("  - Type: MOCAP")
            print("  - Channels: 28 (2 hands × 14 angles)")
            print("  - Format: float32")
            print("  - Sample Rate: Irregular (0 Hz)")
            print(f"  - Angle Names: {ANGLE_NAMES}")
            print("=" * 70 + "\n")
            self._debug_stream_logged = False
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
        self.lsl_broadcaster = None
        self.udp_status.setText("Broadcasting: Disabled")
        self.udp_status.setStyleSheet("color: #888;")

    def broadcast_landmarks(self, frame_landmarks, num_hands):
        if not frame_landmarks:
            return
        if not (self.broadcaster or self.lsl_broadcaster):
            return
        hands_data = []
        for hand_idx, landmarks in enumerate(frame_landmarks):
            hands_data.append({"hand_index": hand_idx, "landmarks": landmarks.tolist()})

        # Debug: Print example packet once
        if not self._debug_stream_logged and hands_data:
            print("\n" + "=" * 70)
            print("EXAMPLE LANDMARK PACKET")
            print("=" * 70)
            print(f"Frame: {self.frame_count}")
            print(f"Timestamp: {self._broadcast_time():.6f}")
            print(f"Number of hands: {len(hands_data)}")
            for hand in hands_data:
                print(f"\n  Hand {hand['hand_index']}:")
                landmarks = hand["landmarks"]
                print(f"    Total landmarks: {len(landmarks)}")
                print("    First 3 landmarks (wrist, thumb_cmc, thumb_mcp):")
                for i in range(min(3, len(landmarks))):
                    print(
                        f"      L{i}: [{landmarks[i][0]:.4f}, {landmarks[i][1]:.4f}, {landmarks[i][2]:.4f}]"
                    )
            print("=" * 70 + "\n")

        if self.broadcaster:
            self.broadcaster.send_landmarks(
                self.frame_count, self._broadcast_time(), hands_data
            )
        if self.lsl_broadcaster:
            self.lsl_broadcaster.send_landmarks(
                self.frame_count, self._broadcast_time(), hands_data
            )

    def broadcast_joint_angles(self, hand_angle_packets):
        if not hand_angle_packets:
            return
        if not (self.broadcaster or self.lsl_broadcaster):
            return

        # Debug: Print example packet once
        if not self._debug_stream_logged and hand_angle_packets:
            print("\n" + "=" * 70)
            print("EXAMPLE ANGLE PACKET")
            print("=" * 70)
            print(f"Frame: {self.frame_count}")
            print(f"Timestamp: {self._broadcast_time():.6f}")
            print(f"Number of hands: {len(hand_angle_packets)}")
            for hand in hand_angle_packets:
                print(f"\n  Hand {hand['hand_index']} Angles (degrees):")
                angles = hand["angles"]
                print(f"    Total angles: {len(angles)}")
                # Print in organized groups
                print("    Thumb:")
                for key in ["thumb_cmc_mcp", "thumb_ip"]:
                    if key in angles:
                        print(f"      {key}: {angles[key]:.2f}°")
                print("    Index:")
                for key in ["index_mcp", "index_pip", "index_dip"]:
                    if key in angles:
                        print(f"      {key}: {angles[key]:.2f}°")
                print("    Middle:")
                for key in ["middle_mcp", "middle_pip", "middle_dip"]:
                    if key in angles:
                        print(f"      {key}: {angles[key]:.2f}°")
                print("    Ring:")
                for key in ["ring_mcp", "ring_pip", "ring_dip"]:
                    if key in angles:
                        print(f"      {key}: {angles[key]:.2f}°")
                print("    Pinky:")
                for key in ["pinky_mcp", "pinky_pip", "pinky_dip"]:
                    if key in angles:
                        print(f"      {key}: {angles[key]:.2f}°")
            print("=" * 70 + "\n")
            self._debug_stream_logged = True

        if self.broadcaster:
            self.broadcaster.send_angles(
                self.frame_count, self._broadcast_time(), hand_angle_packets
            )
        if self.lsl_broadcaster:
            self.lsl_broadcaster.send_angles(
                self.frame_count, self._broadcast_time(), hand_angle_packets
            )

    def _broadcast_time(self):
        if self.use_lsl_clock and HAS_LSL_CLOCK and local_clock:
            return local_clock()
        return time.time()

    # ------------------------------------------------------------------ #
    # Handedness helpers
    # ------------------------------------------------------------------ #
    def _handedness_labels(self, all_results):
        """Best-effort handedness labels from the first camera with results."""
        if not all_results:
            return []
        # Find first camera that has results (not necessarily camera 0)
        ref_result = None
        for result in all_results:
            if (
                result is not None
                and hasattr(result, "multi_handedness")
                and result.multi_handedness
            ):
                ref_result = result
                break
        if ref_result is None:
            return []
        labels = []
        for h in ref_result.multi_handedness:
            try:
                label = h.classification[0].label
                # MediaPipe mirrors, so flip
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

    # ------------------------------------------------------------------ #
    # Tracking start / stop
    # ------------------------------------------------------------------ #
    def toggle_tracking(self):
        if self.is_running:
            self.stop_tracking()
        else:
            self.start_tracking()

    def start_tracking(self):
        """Start multi-camera tracking."""
        try:
            self.tracker = MultiCameraTracker()
            self.tracker.max_reprojection_error = self.occlusion_spin.value()
            if not self.tracker.initialize_cameras(
                min_det_conf=self.min_det_spin.value(),
                min_track_conf=self.min_track_spin.value(),
            ):
                self.status_label.setText("Status: Camera init failed")
                return
            self.num_cameras = self.tracker.num_cameras
            self.camera_info_label.setText(
                f"Cameras: {self.num_cameras}\nIDs: {list(range(self.num_cameras))}"
            )

            # Dynamically populate video labels
            for i in reversed(range(self.video_layout.count())):
                self.video_layout.itemAt(i).widget().setParent(None)
            self.video_labels = []

            cols = 2 if self.num_cameras <= 4 else 3
            for idx in range(self.num_cameras):
                label = QLabel()
                label.setMinimumSize(320, 240)
                label.setStyleSheet(
                    "background-color: #1a1a1a; border: 2px solid #333;"
                )
                label.setAlignment(Qt.AlignCenter)
                self.video_labels.append(label)
                self.video_layout.addWidget(label, idx // cols, idx % cols)

            # Dynamically populate exposure grid
            for i in reversed(range(self.exposure_grid.count())):
                widget = self.exposure_grid.itemAt(i).widget()
                if widget and widget not in [
                    self.exposure_grid.itemAt(0).widget(),
                    self.exposure_grid.itemAt(1).widget(),
                    self.exposure_grid.itemAt(2).widget(),
                ]:
                    widget.setParent(None)

            self.exposure_spinboxes = []
            self.cam_enabled_checkboxes = []
            self.enabled_cameras = set(range(self.num_cameras))

            for idx in range(self.num_cameras):
                cb = QCheckBox()
                cb.setChecked(True)
                cb.setToolTip(f"Include camera {idx} in triangulation")
                cb.toggled.connect(
                    lambda checked, i=idx: self._on_cam_enabled_toggled(i, checked)
                )
                self.cam_enabled_checkboxes.append(cb)
                lbl = QLabel(f"Cam {idx}")
                lbl.setStyleSheet("color: #aaa; font-size: 11px;")
                spin = QSpinBox()
                spin.setRange(1, 480)
                spin.setValue(self.default_exposure)
                spin.setSuffix(" ms")
                spin.setToolTip(
                    f"Exposure for camera {idx}.\n"
                    "Lower = faster/darker, Higher = slower/brighter."
                )
                spin.valueChanged.connect(
                    lambda val, i=idx: self._on_exposure_changed(i, val)
                )
                self.exposure_spinboxes.append(spin)
                row = idx + 1  # row 0 is header
                self.exposure_grid.addWidget(cb, row, 0, Qt.AlignCenter)
                self.exposure_grid.addWidget(lbl, row, 1)
                self.exposure_grid.addWidget(spin, row, 2)

            # Apply current exposure spinbox values to each camera
            for idx in range(min(self.num_cameras, len(self.exposure_spinboxes))):
                self.tracker.set_exposure(idx, self.exposure_spinboxes[idx].value())
            # Apply current triangulation camera selection
            self.tracker.set_enabled_cameras(set(self.enabled_cameras))
            # Ensure LSL outlet exists even if UDP toggle is off
            if self.lsl_broadcaster is None:
                try:
                    self.lsl_broadcaster = LSLBroadcaster(
                        stream_name="StereoHandTracker"
                    )
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
            # Start timer — as fast as possible (1 ms)
            self.timer.start(1)
        except Exception as e:
            self.status_label.setText(f"Status: Error - {str(e)}")
            print(f"Error starting tracker: {e}")

    def stop_tracking(self):
        """Stop tracking."""
        self.timer.stop()
        if self.tracker:
            self.tracker.cleanup()
            self.tracker = None
        self.is_running = False
        self.start_btn.setText("Start Tracking")
        self.start_btn.setStyleSheet("")
        self.status_label.setText("Status: Stopped")
        self.status_label.setStyleSheet("color: #ffffff;")

    # ------------------------------------------------------------------ #
    # Main frame loop
    # ------------------------------------------------------------------ #
    def update_frame(self):
        """Process and display frames from all cameras."""
        if not self.tracker:
            self.stop_tracking()
            return
        try:
            frames, triangulated_hands_data, all_results, valid_2d_landmarks = (
                self.tracker.process_frame()
            )
            num_hands = len(triangulated_hands_data)
            handedness_labels = self._handedness_labels(all_results)
            frame_landmarks = []
            hand_angle_packets = []
            selected_idx = 0
            raw_hands = []
            filtered_hands = []
            tracked_label = "None"

            all_best_cams = set()

            for hand_idx, (landmarks_3d, best_cams, valid_lms) in enumerate(
                triangulated_hands_data
            ):
                all_best_cams.update(best_cams)
                if self.apply_kalman and hand_idx < len(self.kalman_filters):
                    filtered_landmarks = np.array(
                        [
                            self.kalman_filters[hand_idx][i].update(landmarks_3d[i])
                            for i in range(len(landmarks_3d))
                        ]
                    )
                    landmarks_array = filtered_landmarks
                else:
                    landmarks_array = landmarks_3d
                raw_hands.append(landmarks_3d)
                filtered_hands.append(landmarks_array)
                hand_label = (
                    handedness_labels[hand_idx]
                    if hand_idx < len(handedness_labels)
                    else "Unknown"
                )
                if not self._hand_matches_preference(hand_label):
                    continue
                frame_landmarks.append(landmarks_array)
                if tracked_label == "None":
                    tracked_label = hand_label
                joint_angles = finger_bend_angles(landmarks_array)

                # # -- Trying to do splay angles --#
                # splay_angles = finger_splay_angles(landmarks_array)
                # joint_angles.update(splay_angles)

                # # -- Comment if unecessary or bad lol --#
                if self.apply_kalman and hand_idx < len(self.angle_kalman_filters):
                    filtered_angles = {}
                    for angle_name, angle_value in joint_angles.items():
                        filtered_angles[angle_name] = self.angle_kalman_filters[
                            hand_idx
                        ][angle_name].update(angle_value)
                    joint_angles = filtered_angles
                nan_count = 0
                for k, v in list(joint_angles.items()):
                    if not np.isfinite(v):
                        nan_count += 1
                        joint_angles[k] = 0.0
                now = time.time()
                if hand_idx == 0 and now - self._last_angle_log > 2.0:
                    self._last_angle_log = now
                    # print(
                    #     f"\n[Angles] nan={nan_count} "
                    #     f"index=[{joint_angles['index_mcp']:.1f}, {joint_angles['index_pip']:.1f}, {joint_angles['index_dip']:.1f}] "
                    #     f"thumb=[{joint_angles['thumb_cmc_mcp']:.1f}, {joint_angles['thumb_ip']:.1f}]"
                    # )
                out_index = 0 if self.remap_selected_hand else selected_idx
                hand_angle_packets.append(
                    {"hand_index": out_index, "angles": joint_angles}
                )
                selected_idx += 1
                if hand_idx == 0:
                    print(
                        f"\rFrame {self.frame_count} [3D Triangulated] - "
                        f"Index MCP: {joint_angles['index_mcp']:.2f}\u00b0, "
                        f"Index PIP: {joint_angles['index_pip']:.2f}\u00b0, "
                        f"Index DIP: {joint_angles['index_dip']:.2f}\u00b0 | ",
                        end="",
                        flush=True,
                    )
            # Broadcast
            if self.udp_enabled or self.lsl_broadcaster:
                if frame_landmarks:
                    self.broadcast_landmarks(frame_landmarks, num_hands)
                if hand_angle_packets:
                    self.broadcast_joint_angles(hand_angle_packets)
            # Display frames
            for idx, (frame, results) in enumerate(zip(frames, all_results)):
                if frame is None or idx >= len(self.video_labels):
                    continue
                display_frame = frame.copy()
                if not self.show_raw_video:
                    display_frame[:] = 0

                # Draw green border if this camera is one of the best cameras
                if idx in all_best_cams:
                    cv2.rectangle(
                        display_frame,
                        (0, 0),
                        (display_frame.shape[1] - 1, display_frame.shape[0] - 1),
                        (0, 255, 0),
                        10,
                    )

                # Draw per-finger coloured skeleton
                if results and results.multi_hand_landmarks:
                    for hand_idx_in_cam, hand_landmarks in enumerate(
                        results.multi_hand_landmarks
                    ):
                        mp.solutions.drawing_utils.draw_landmarks(
                            display_frame,
                            hand_landmarks,
                            mp.solutions.hands.HAND_CONNECTIONS,
                            mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4),
                            mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                        )
                cv2.putText(
                    display_frame,
                    f"Camera {idx}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(
                    rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888
                )
                scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                    self.video_labels[idx].size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                self.video_labels[idx].setPixmap(scaled_pixmap)
            # Jitter meter
            if raw_hands:
                raw0 = raw_hands[0]
                filt0 = filtered_hands[0]
                if self.prev_raw_landmarks is not None:
                    raw_speed = np.linalg.norm(
                        raw0 - self.prev_raw_landmarks, axis=1
                    ).mean()
                    self.raw_jitter_window.append(raw_speed)
                    if len(self.raw_jitter_window) > self.jitter_window_size:
                        self.raw_jitter_window = self.raw_jitter_window[
                            -self.jitter_window_size :
                        ]
                if self.prev_filtered_landmarks is not None:
                    filt_speed = np.linalg.norm(
                        filt0 - self.prev_filtered_landmarks, axis=1
                    ).mean()
                    self.filtered_jitter_window.append(filt_speed)
                    if len(self.filtered_jitter_window) > self.jitter_window_size:
                        self.filtered_jitter_window = self.filtered_jitter_window[
                            -self.jitter_window_size :
                        ]
                self.prev_raw_landmarks = raw0
                self.prev_filtered_landmarks = filt0
                if self.raw_jitter_window and self.filtered_jitter_window:
                    raw_std = np.std(self.raw_jitter_window)
                    filt_std = np.std(self.filtered_jitter_window)
                    self.jitter_label.setText(
                        f"Jitter (raw/filtered): {raw_std:.4f} / {filt_std:.4f}"
                    )
            if self.show_3d_view and triangulated_hands_data:
                self.update_3d_view(raw_hands, filtered_hands)
            self.frame_count += 1
            self.frame_label.setText(f"Frames: {self.frame_count}")
            self.hands_label.setText(f"Hands: {num_hands}")
            self.tracked_hand_label.setText(f"Tracked Hand: {tracked_label}")
            if self.tracker:
                fps_stats = self.tracker.get_fps_stats()
                fps = fps_stats.get("fps", 0)
                capture_ms = fps_stats.get("capture_ms", 0)
                detection_ms = fps_stats.get("detection_ms", 0)
                tri_ms = fps_stats.get("triangulation_ms", 0)
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
        if event.key() == Qt.Key_Escape:
            self.close()

    # ------------------------------------------------------------------ #
    # 3D Visualization
    # ------------------------------------------------------------------ #
    def update_3d_view(self, raw_hands, filtered_hands):
        """Render 3D visualization (Top, Front, Side views)."""
        w, h = 600, 600
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        half_w, half_h = w // 2, h // 2
        cv2.line(canvas, (half_w, 0), (half_w, h), (50, 50, 50), 1)
        cv2.line(canvas, (0, half_h), (w, half_h), (50, 50, 50), 1)
        cv2.putText(
            canvas,
            "Top View (XZ)",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        cv2.putText(
            canvas,
            "Front View (XY)",
            (10, half_h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        cv2.putText(
            canvas,
            "Side View (ZY)",
            (half_w + 10, half_h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        scale = 1.5
        for idx, hand_landmarks in enumerate(filtered_hands):
            if hand_landmarks is None or len(hand_landmarks) == 0:
                continue
            connections = [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (0, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (0, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (0, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (0, 17),
                (17, 18),
                (18, 19),
                (19, 20),
            ]

            def draw_skeleton(view_type, offset_x, offset_y, points, color):
                for start_idx, end_idx in connections:
                    p1 = points[start_idx]
                    p2 = points[end_idx]
                    if np.all(p1 == 0) or np.all(p2 == 0):
                        continue
                    if view_type == "top":
                        u1, v1 = (
                            int(p1[0] * scale) + offset_x + 150,
                            int(p1[2] * scale) + offset_y + 150,
                        )
                        u2, v2 = (
                            int(p2[0] * scale) + offset_x + 150,
                            int(p2[2] * scale) + offset_y + 150,
                        )
                    elif view_type == "front":
                        u1, v1 = (
                            int(p1[0] * scale) + offset_x + 150,
                            int(p1[1] * scale) + offset_y + 50,
                        )
                        u2, v2 = (
                            int(p2[0] * scale) + offset_x + 150,
                            int(p2[1] * scale) + offset_y + 50,
                        )
                    elif view_type == "side":
                        u1, v1 = (
                            int(p1[2] * scale) + offset_x + 150,
                            int(p1[1] * scale) + offset_y + 50,
                        )
                        u2, v2 = (
                            int(p2[2] * scale) + offset_x + 150,
                            int(p2[1] * scale) + offset_y + 50,
                        )
                    cv2.line(canvas, (u1, v1), (u2, v2), color, 1)
                    cv2.circle(canvas, (u1, v1), 2, color, -1)

            draw_skeleton("top", 0, 0, hand_landmarks, (0, 255, 0))
            draw_skeleton("front", 0, half_h, hand_landmarks, (0, 255, 0))
            draw_skeleton("side", half_w, half_h, hand_landmarks, (0, 255, 0))
            if self.show_raw_filtered_overlay and idx < len(raw_hands):
                raw_points = raw_hands[idx]
                draw_skeleton("top", 0, 0, raw_points, (120, 120, 120))
                draw_skeleton("front", 0, half_h, raw_points, (120, 120, 120))
                draw_skeleton("side", half_w, half_h, raw_points, (120, 120, 120))
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


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Stereo Hand Tracker")
    args = parser.parse_args()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
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
