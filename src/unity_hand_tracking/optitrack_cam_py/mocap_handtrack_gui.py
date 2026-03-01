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
import importlib
import os
import sys
import time

# Preload MediaPipe before importing cv2/PyQt native stacks.
# This minimizes Windows DLL initialization order conflicts.
try:
    importlib.import_module("mediapipe")
    _MP_PRELOAD_OK = True
    _MP_PRELOAD_ERROR = ""
except Exception as _mp_exc:
    _MP_PRELOAD_OK = False
    _MP_PRELOAD_ERROR = str(_mp_exc)

import cv2
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

if __package__:
    from ._processing_thread import ProcessingThread
    from .broadcast import LSLBroadcaster, UDPBroadcaster
    from .config import (
        ANGLE_NAMES,
        EMA_1D_ALPHA,
        EMA_3D_ALPHA,
        KALMAN_1D_MEASUREMENT_NOISE,
        KALMAN_1D_PROCESS_NOISE,
        KALMAN_3D_MEASUREMENT_NOISE,
        KALMAN_3D_PROCESS_NOISE,
        MAX_HANDS,
        MAX_REPROJECTION_ERROR,
        MIN_DETECTION_CONFIDENCE,
        MIN_TRACKING_CONFIDENCE,
        NUM_LANDMARKS,
        SMOOTHING_METHOD,
        UDP_IP,
        UDP_PORT_ANGLES,
        UDP_PORT_LANDMARKS,
    )
else:
    from _processing_thread import ProcessingThread
    from broadcast import LSLBroadcaster, UDPBroadcaster
    from config import (
        ANGLE_NAMES,
        EMA_1D_ALPHA,
        EMA_3D_ALPHA,
        KALMAN_1D_MEASUREMENT_NOISE,
        KALMAN_1D_PROCESS_NOISE,
        KALMAN_3D_MEASUREMENT_NOISE,
        KALMAN_3D_PROCESS_NOISE,
        MAX_HANDS,
        MAX_REPROJECTION_ERROR,
        MIN_DETECTION_CONFIDENCE,
        MIN_TRACKING_CONFIDENCE,
        NUM_LANDMARKS,
        SMOOTHING_METHOD,
        UDP_IP,
        UDP_PORT_ANGLES,
        UDP_PORT_LANDMARKS,
    )

try:
    from handtrack.processing import (
        build_smoother_factories,
        enforce_pip_constraints,
        finger_bend_angles,
    )
except ImportError:
    from pathlib import Path

    _src_root = Path(__file__).resolve().parents[2]
    if str(_src_root) not in sys.path:
        sys.path.insert(0, str(_src_root))
    from handtrack.processing import (
        build_smoother_factories,
        finger_bend_angles,
    )


def _get_multi_camera_tracker_class():
    if __package__:
        from .mocap_tracker import MultiCameraTracker
    else:
        from mocap_tracker import MultiCameraTracker
    return MultiCameraTracker


def _preload_mediapipe_runtime() -> tuple[bool, str]:
    """Eagerly load MediaPipe native bindings before other native stacks.

    This reduces Windows DLL initialization conflicts that can appear when
    `_framework_bindings` is imported later in the session.
    """
    if _MP_PRELOAD_OK:
        return True, ""
    try:
        importlib.import_module("mediapipe")
        return True, ""
    except Exception as exc:
        return False, str(exc)


# =============================================================================
# GUI
# =============================================================================
class StereoHandTrackerGUI(QMainWindow):
    """Main window with multi-camera display and control panel."""

    def __init__(self):
        super().__init__()
        self._mediapipe_ready = _MP_PRELOAD_OK
        self._mediapipe_error = _MP_PRELOAD_ERROR
        if not self._mediapipe_ready:
            # Retry once in case environment was adjusted between import and init.
            self._mediapipe_ready, self._mediapipe_error = _preload_mediapipe_runtime()
        # Tracking state
        self.is_running = False
        self.apply_kalman = True
        self.show_raw_video = True
        self.show_3d_view = False
        self.show_3d_top = False
        self.show_3d_front = False
        self.show_3d_side = False
        self.show_3d_iso = True
        self.frame_count = 0
        self.max_hands = MAX_HANDS
        self.show_raw_filtered_overlay = False
        # Multi-camera tracker
        self.tracker = None
        self._proc_thread: ProcessingThread | None = None
        self.num_cameras = 0
        self._display_fps = 0.0
        self._display_t_prev = time.perf_counter()
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
        self.smoothing_method = str(SMOOTHING_METHOD).strip().lower()
        if self.smoothing_method not in ("kalman", "ema"):
            self.smoothing_method = "kalman"
        self.ema_3d_alpha = float(EMA_3D_ALPHA)
        self.ema_1d_alpha = float(EMA_1D_ALPHA)
        self.kalman_filters = []
        self.angle_kalman_filters = []
        self.reset_kalman_filters()
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

        if not self._mediapipe_ready:
            print(f"[Startup] MediaPipe preload failed: {self._mediapipe_error}")

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
        self.vis_3d_label.setMinimumSize(350, 350)
        self.vis_3d_label.setStyleSheet(
            "background-color: #000; border: 1px solid #555;"
        )
        self.vis_3d_label.setAlignment(Qt.AlignCenter)
        self.vis_3d_label.hide()
        main_layout.addWidget(self.vis_3d_label, stretch=3)
        # Control panel (right side)
        control_panel = self.create_control_panel()
        scroll_area = QScrollArea()
        scroll_area.setWidget(control_panel)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(260)
        scroll_area.setStyleSheet(
            "QScrollArea { border: none; background-color: #2d2d2d; }"
        )
        main_layout.addWidget(scroll_area, stretch=2)

    def create_control_panel(self):
        """Create the right-side control panel."""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame { background-color: #2d2d2d; border-radius: 6px; }
            QLabel { color: #ffffff; font-size: 11px; }
            QGroupBox {
                color: #ffffff;
                font-weight: bold;
                font-size: 11px;
                border: 1px solid #444;
                border-radius: 4px;
                margin-top: 6px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 3px;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #1084d8; }
            QPushButton:pressed { background-color: #006abc; }
            QPushButton:disabled { background-color: #555; color: #888; }
            QCheckBox { color: #ffffff; font-size: 11px; }
            QSpinBox, QDoubleSpinBox {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555;
                padding: 2px;
                font-size: 11px;
            }
        """)
        layout = QVBoxLayout(panel)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)
        # ---- Tracking Control + FPS (top for quick access) ----
        control_group = QGroupBox("Tracking")
        control_layout = QVBoxLayout(control_group)
        control_layout.setSpacing(3)
        self.start_btn = QPushButton("Start Tracking")
        self.start_btn.clicked.connect(self.toggle_tracking)
        control_layout.addWidget(self.start_btn)
        self.status_label = QLabel("Status: Stopped")
        self.status_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.status_label)
        status_row = QHBoxLayout()
        status_row.setSpacing(4)
        self.frame_label = QLabel("Frames: 0")
        self.frame_label.setStyleSheet("font-size: 10px; color: #aaa;")
        self.hands_label = QLabel("Hands: 0")
        self.hands_label.setStyleSheet("font-size: 10px; color: #aaa;")
        self.tracked_hand_label = QLabel("Hand: N/A")
        self.tracked_hand_label.setStyleSheet("font-size: 10px; color: #aaa;")
        status_row.addWidget(self.frame_label)
        status_row.addWidget(self.hands_label)
        status_row.addWidget(self.tracked_hand_label)
        control_layout.addLayout(status_row)
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet(
            "color: #00ff00; font-weight: bold; font-size: 11px;"
        )
        self.fps_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.fps_label)
        self.timing_label = QLabel("Capture: 0ms | Detect: 0ms | Tri: 0ms")
        self.timing_label.setStyleSheet("color: #aaa; font-size: 9px;")
        self.timing_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.timing_label)
        layout.addWidget(control_group)
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
        self.occlusion_spin.setValue(MAX_REPROJECTION_ERROR)
        self.occlusion_spin.valueChanged.connect(self.on_occlusion_changed)
        mp_layout.addWidget(self.occlusion_spin, 2, 1)

        processing_layout.addLayout(mp_layout)

        self.kalman_checkbox = QCheckBox("Enable Smoothing")
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

        # ---- 3D View Toggles ----
        views_3d_group = QGroupBox("3D Views")
        views_3d_layout = QHBoxLayout(views_3d_group)
        views_3d_layout.setContentsMargins(4, 4, 4, 4)

        self.cb_3d_top = QCheckBox("Top")
        self.cb_3d_top.setChecked(self.show_3d_top)
        self.cb_3d_top.toggled.connect(lambda c: setattr(self, "show_3d_top", c))
        views_3d_layout.addWidget(self.cb_3d_top)

        self.cb_3d_front = QCheckBox("Front")
        self.cb_3d_front.setChecked(self.show_3d_front)
        self.cb_3d_front.toggled.connect(lambda c: setattr(self, "show_3d_front", c))
        views_3d_layout.addWidget(self.cb_3d_front)

        self.cb_3d_side = QCheckBox("Side")
        self.cb_3d_side.setChecked(self.show_3d_side)
        self.cb_3d_side.toggled.connect(lambda c: setattr(self, "show_3d_side", c))
        views_3d_layout.addWidget(self.cb_3d_side)

        self.cb_3d_iso = QCheckBox("Iso")
        self.cb_3d_iso.setChecked(self.show_3d_iso)
        self.cb_3d_iso.toggled.connect(lambda c: setattr(self, "show_3d_iso", c))
        views_3d_layout.addWidget(self.cb_3d_iso)

        processing_layout.addWidget(views_3d_group)
        layout.addWidget(processing_group)
        # ---- Smoothing / Jitter ----
        smooth_group = QGroupBox("Smoothing")
        smooth_layout = QVBoxLayout(smooth_group)
        smooth_layout.setSpacing(3)

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method"))
        self.smooth_method_combo = QComboBox()
        self.smooth_method_combo.addItem("Kalman (adaptive)", "kalman")
        self.smooth_method_combo.addItem("EMA (low-latency)", "ema")
        idx = self.smooth_method_combo.findData(self.smoothing_method)
        self.smooth_method_combo.setCurrentIndex(max(idx, 0))
        self.smooth_method_combo.currentIndexChanged.connect(
            self.on_smoothing_method_changed
        )
        method_row.addWidget(self.smooth_method_combo)
        smooth_layout.addLayout(method_row)

        kalman_grid = QGridLayout()
        kalman_grid.setSpacing(2)
        kalman_grid.addWidget(QLabel("3D Proc"), 0, 0)
        self.k3d_p = QDoubleSpinBox()
        self.k3d_p.setDecimals(6)
        self.k3d_p.setRange(1e-6, 1.0)
        self.k3d_p.setSingleStep(1e-4)
        self.k3d_p.setValue(self.kalman_3d_process_noise)
        self.k3d_p.valueChanged.connect(self.on_kalman_params_changed)
        kalman_grid.addWidget(self.k3d_p, 0, 1)
        kalman_grid.addWidget(QLabel("3D Meas"), 0, 2)
        self.k3d_m = QDoubleSpinBox()
        self.k3d_m.setDecimals(6)
        self.k3d_m.setRange(1e-6, 1.0)
        self.k3d_m.setSingleStep(1e-4)
        self.k3d_m.setValue(self.kalman_3d_measurement_noise)
        self.k3d_m.valueChanged.connect(self.on_kalman_params_changed)
        kalman_grid.addWidget(self.k3d_m, 0, 3)
        kalman_grid.addWidget(QLabel("Ang Proc"), 1, 0)
        self.k1d_p = QDoubleSpinBox()
        self.k1d_p.setDecimals(6)
        self.k1d_p.setRange(1e-6, 10.0)
        self.k1d_p.setSingleStep(1e-3)
        self.k1d_p.setValue(self.kalman_1d_process_noise)
        self.k1d_p.valueChanged.connect(self.on_kalman_params_changed)
        kalman_grid.addWidget(self.k1d_p, 1, 1)
        kalman_grid.addWidget(QLabel("Ang Meas"), 1, 2)
        self.k1d_m = QDoubleSpinBox()
        self.k1d_m.setDecimals(6)
        self.k1d_m.setRange(1e-6, 100.0)
        self.k1d_m.setSingleStep(1e-2)
        self.k1d_m.setValue(self.kalman_1d_measurement_noise)
        self.k1d_m.valueChanged.connect(self.on_kalman_params_changed)
        kalman_grid.addWidget(self.k1d_m, 1, 3)
        smooth_layout.addLayout(kalman_grid)

        ema_grid = QGridLayout()
        ema_grid.setSpacing(2)
        ema_grid.addWidget(QLabel("EMA 3D α"), 0, 0)
        self.ema3d_alpha_spin = QDoubleSpinBox()
        self.ema3d_alpha_spin.setDecimals(3)
        self.ema3d_alpha_spin.setRange(0.01, 1.0)
        self.ema3d_alpha_spin.setSingleStep(0.01)
        self.ema3d_alpha_spin.setValue(self.ema_3d_alpha)
        self.ema3d_alpha_spin.valueChanged.connect(self.on_ema_params_changed)
        ema_grid.addWidget(self.ema3d_alpha_spin, 0, 1)

        ema_grid.addWidget(QLabel("EMA Ang α"), 0, 2)
        self.ema1d_alpha_spin = QDoubleSpinBox()
        self.ema1d_alpha_spin.setDecimals(3)
        self.ema1d_alpha_spin.setRange(0.01, 1.0)
        self.ema1d_alpha_spin.setSingleStep(0.01)
        self.ema1d_alpha_spin.setValue(self.ema_1d_alpha)
        self.ema1d_alpha_spin.valueChanged.connect(self.on_ema_params_changed)
        ema_grid.addWidget(self.ema1d_alpha_spin, 0, 3)
        smooth_layout.addLayout(ema_grid)
        self.jitter_label = QLabel("Jitter: N/A")
        self.jitter_label.setStyleSheet("font-size: 10px; color: #aaa;")
        smooth_layout.addWidget(self.jitter_label)
        self.overlay_checkbox = QCheckBox("Overlay raw vs filtered")
        self.overlay_checkbox.toggled.connect(self.on_overlay_toggled)
        smooth_layout.addWidget(self.overlay_checkbox)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self.kalman_reset_btn = QPushButton("Reset")
        self.kalman_reset_btn.clicked.connect(self.on_kalman_reset)
        self.kalman_save_btn = QPushButton("Save")
        self.kalman_save_btn.clicked.connect(self.on_kalman_save)
        btn_row.addWidget(self.kalman_reset_btn)
        btn_row.addWidget(self.kalman_save_btn)
        smooth_layout.addLayout(btn_row)
        self._set_smoothing_ui_state()
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
    def _set_smoothing_ui_state(self):
        is_kalman = self.smoothing_method == "kalman"
        for widget in (self.k3d_p, self.k3d_m, self.k1d_p, self.k1d_m):
            widget.setEnabled(is_kalman)
        for widget in (self.ema3d_alpha_spin, self.ema1d_alpha_spin):
            widget.setEnabled(not is_kalman)

    def on_smoothing_method_changed(self, _index):
        self.smoothing_method = self.smooth_method_combo.currentData()
        if self.smoothing_method not in ("kalman", "ema"):
            self.smoothing_method = "kalman"
        self._set_smoothing_ui_state()
        if self.apply_kalman:
            self.reset_kalman_filters()

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
            # Pause the processing thread while we recreate MediaPipe detectors
            if self._proc_thread is not None:
                self._proc_thread.pause()
            self.tracker.update_mp_params(
                self.min_det_spin.value(), self.min_track_spin.value()
            )
            if self._proc_thread is not None:
                self._proc_thread.resume()

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

    def on_ema_params_changed(self):
        self.ema_3d_alpha = float(self.ema3d_alpha_spin.value())
        self.ema_1d_alpha = float(self.ema1d_alpha_spin.value())
        if self.apply_kalman and self.smoothing_method == "ema":
            self.reset_kalman_filters()

    def on_kalman_reset(self):
        self.k3d_p.setValue(KALMAN_3D_PROCESS_NOISE)
        self.k3d_m.setValue(KALMAN_3D_MEASUREMENT_NOISE)
        self.k1d_p.setValue(KALMAN_1D_PROCESS_NOISE)
        self.k1d_m.setValue(KALMAN_1D_MEASUREMENT_NOISE)
        self.ema3d_alpha_spin.setValue(EMA_3D_ALPHA)
        self.ema1d_alpha_spin.setValue(EMA_1D_ALPHA)
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
            replace_value("SMOOTHING_METHOD", f'"{self.smoothing_method}"')
            replace_value("EMA_3D_ALPHA", self.ema_3d_alpha)
            replace_value("EMA_1D_ALPHA", self.ema_1d_alpha)
            with open(config_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            self.status_label.setText("Status: Saved smoothing params")
        except Exception as exc:
            self.status_label.setText(f"Status: Save failed - {exc}")

    def reset_kalman_filters(self):
        """Reset smoothing filters for landmarks and angles."""
        landmark_factory, angle_factory = build_smoother_factories(
            method=self.smoothing_method,
            kalman_3d_process_noise=self.kalman_3d_process_noise,
            kalman_3d_measurement_noise=self.kalman_3d_measurement_noise,
            kalman_1d_process_noise=self.kalman_1d_process_noise,
            kalman_1d_measurement_noise=self.kalman_1d_measurement_noise,
            ema_3d_alpha=self.ema_3d_alpha,
            ema_1d_alpha=self.ema_1d_alpha,
        )

        self.kalman_filters = [
            [landmark_factory() for _ in range(NUM_LANDMARKS)]
            for _ in range(self.max_hands)
        ]
        self.angle_kalman_filters = [
            {name: angle_factory() for name in ANGLE_NAMES}
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
            if not self._mediapipe_ready:
                self._mediapipe_ready, self._mediapipe_error = (
                    _preload_mediapipe_runtime()
                )
                if not self._mediapipe_ready:
                    msg = (
                        "MediaPipe preload failed. "
                        "Try running from PowerShell/CMD (not MINGW/MSYS), "
                        "and ensure VC++ Redistributable 2015-2022 is installed."
                    )
                    self.status_label.setText(f"Status: Error - {msg}")
                    print(f"Error starting tracker: {msg}")
                    print(f"MediaPipe preload detail: {self._mediapipe_error}")
                    return

            MultiCameraTracker = _get_multi_camera_tracker_class()
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
            # Start background processing thread
            self._proc_thread = ProcessingThread(self.tracker)
            self._proc_thread.start()
            # Start GUI refresh timer (~60 Hz display rate)
            self.timer.start(16)
        except Exception as e:
            self.status_label.setText(f"Status: Error - {str(e)}")
            print(f"Error starting tracker: {e}")

    def stop_tracking(self):
        """Stop tracking."""
        self.timer.stop()
        # Stop the processing thread first
        if self._proc_thread is not None:
            self._proc_thread.stop()
            self._proc_thread = None
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
        """Display the latest available result from the processing thread."""
        if not self.tracker or self._proc_thread is None:
            self.stop_tracking()
            return
        # Check if processing thread died unexpectedly
        if not self._proc_thread.is_alive():
            print("[GUI] Processing thread died — stopping.")
            self.stop_tracking()
            return
        try:
            latest = self._proc_thread.get_latest()
            if latest is None:
                return  # No new frame yet — skip this timer tick
            frames, triangulated_hands_data, all_results, valid_2d_landmarks = latest
            num_hands = len(triangulated_hands_data)
            handedness_labels = self._handedness_labels(all_results)
            frame_landmarks = []
            hand_angle_packets = []
            selected_idx = 0
            raw_hands = []
            filtered_hands = []
            tracked_label = "None"

            all_best_cams = set()

            for hand_idx, (
                landmarks_3d,
                best_cams,
                valid_lms,
                reproj_errs,
                n_cams_per_lm,
            ) in enumerate(triangulated_hands_data):
                all_best_cams.update(best_cams)
                # landmarks_3d = enforce_pip_constraints(landmarks_3d)
                if self.apply_kalman and hand_idx < len(self.kalman_filters):
                    filtered_landmarks = np.array(
                        [
                            self.kalman_filters[hand_idx][i].update(
                                landmarks_3d[i],
                                reprojection_error=reproj_errs[i],
                                num_cameras=n_cams_per_lm[i],
                            )
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
                        f"Thumb CMC-MCP: {joint_angles['thumb_cmc_mcp']:.2f}\u00b0, "
                        f"Thumb IP: {joint_angles['thumb_ip']:.2f}\u00b0 | ",
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

                # If the camera is upside down, rotate the frame 180 degrees permanently
                # so it looks right-side up in the GUI.
                # if idx in UPSIDE_DOWN_CAMERAS:
                #     display_frame = cv2.rotate(display_frame, cv2.ROTATE_180)

                # Draw per-finger coloured skeleton on 2D camera view
                if results and results.multi_hand_landmarks:
                    # Finger colors (BGR) matching the 3D view
                    _FINGER_COLORS_2D = {
                        "thumb": (0, 200, 255),  # Orange
                        "index": (0, 255, 100),  # Green
                        "middle": (255, 200, 0),  # Cyan-blue
                        "ring": (255, 0, 150),  # Magenta
                        "pinky": (100, 100, 255),  # Red-ish
                        "palm": (180, 180, 180),  # Grey
                    }
                    _COLORED_CONNS_2D = [
                        (0, 1, "thumb"),
                        (1, 2, "thumb"),
                        (2, 3, "thumb"),
                        (3, 4, "thumb"),
                        (0, 5, "index"),
                        (5, 6, "index"),
                        (6, 7, "index"),
                        (7, 8, "index"),
                        (5, 9, "palm"),
                        (9, 13, "palm"),
                        (13, 17, "palm"),
                        (0, 17, "palm"),
                        (9, 10, "middle"),
                        (10, 11, "middle"),
                        (11, 12, "middle"),
                        (13, 14, "ring"),
                        (14, 15, "ring"),
                        (15, 16, "ring"),
                        (17, 18, "pinky"),
                        (18, 19, "pinky"),
                        (19, 20, "pinky"),
                    ]
                    for hand_landmarks in results.multi_hand_landmarks:
                        h_img, w_img = display_frame.shape[:2]
                        lm = hand_landmarks.landmark
                        pts = [
                            (int(lm_point.x * w_img), int(lm_point.y * h_img))
                            for lm_point in lm
                        ]
                        for si, ei, finger in _COLORED_CONNS_2D:
                            col = _FINGER_COLORS_2D[finger]
                            cv2.line(display_frame, pts[si], pts[ei], col, 2)
                        for finger, indices in [
                            ("thumb", [1, 2, 3, 4]),
                            ("index", [5, 6, 7, 8]),
                            ("middle", [9, 10, 11, 12]),
                            ("ring", [13, 14, 15, 16]),
                            ("pinky", [17, 18, 19, 20]),
                            ("palm", [0]),
                        ]:
                            col = _FINGER_COLORS_2D[finger]
                            for i in indices:
                                cv2.circle(display_frame, pts[i], 3, col, -1)
                                cv2.circle(display_frame, pts[i], 4, (255, 255, 255), 1)
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
            # --- FPS metrics (processing vs display) ---
            now_d = time.perf_counter()
            dt_d = now_d - self._display_t_prev
            self._display_fps = 1.0 / dt_d if dt_d > 0 else 0.0
            self._display_t_prev = now_d

            if self.tracker:
                fps_stats = self.tracker.get_fps_stats()
                proc_fps = (
                    self._proc_thread.get_processing_fps() if self._proc_thread else 0.0
                )
                capture_ms = fps_stats.get("capture_ms", 0)
                detection_ms = fps_stats.get("detection_ms", 0)
                tri_ms = fps_stats.get("triangulation_ms", 0)
                if proc_fps > 25:
                    fps_color = "#00ff00"
                elif proc_fps > 15:
                    fps_color = "#ffff00"
                else:
                    fps_color = "#ff0000"
                self.fps_label.setText(
                    f"Proc: {proc_fps:.1f} | Draw: {self._display_fps:.1f} FPS"
                )
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
        """Render 3D visualization (Top, Front, Side, Isometric views)."""
        w, h = 600, 600
        canvas = np.full((h, w, 3), 40, dtype=np.uint8)

        # Draw grid lines
        grid_spacing = 50
        for i in range(0, w, grid_spacing):
            cv2.line(canvas, (i, 0), (i, h), (60, 60, 60), 1)
        for i in range(0, h, grid_spacing):
            cv2.line(canvas, (0, i), (w, i), (60, 60, 60), 1)

        half_w, half_h = w // 2, h // 2
        qw, qh = half_w, half_h  # quadrant size

        # Draw quadrant dividers
        cv2.line(canvas, (half_w, 0), (half_w, h), (100, 100, 100), 2)
        cv2.line(canvas, (0, half_h), (w, half_h), (100, 100, 100), 2)

        # Count how many views are enabled
        active_views = []
        if self.show_3d_top:
            active_views.append("top")
        if self.show_3d_front:
            active_views.append("front")
        if self.show_3d_side:
            active_views.append("side")
        if self.show_3d_iso:
            active_views.append("iso")

        if not active_views:
            # Nothing enabled — show placeholder
            cv2.putText(
                canvas,
                "Enable a 3D view",
                (w // 2 - 100, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (120, 120, 120),
                1,
            )
            rgb_frame = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            h_c, w_c, ch = rgb_frame.shape
            qt_image = QImage(rgb_frame.data, w_c, h_c, ch * w_c, QImage.Format_RGB888)
            self.vis_3d_label.setPixmap(QPixmap.fromImage(qt_image))
            return

        # Compute layout: 1 view = full canvas, 2-4 views = 2x2 grid
        n_views = len(active_views)
        if n_views == 1:
            # Single view uses the full canvas
            view_rects = {active_views[0]: (0, 0, w, h)}
        else:
            # 2x2 grid
            slots = [(0, 0), (half_w, 0), (0, half_h), (half_w, half_h)]
            view_rects = {}
            for i, vname in enumerate(active_views):
                ox, oy = slots[i]
                view_rects[vname] = (ox, oy, qw, qh)
            # Draw dividers only if multi-view
            if n_views > 1:
                if n_views > 2:
                    cv2.line(canvas, (0, half_h), (w, half_h), (100, 100, 100), 2)
                cv2.line(canvas, (half_w, 0), (half_w, h), (100, 100, 100), 2)

        view_labels = {
            "top": "Top (XZ)",
            "front": "Front (XY)",
            "side": "Side (ZY)",
            "iso": "Iso",
        }
        for vname, (ox, oy, vw, vh) in view_rects.items():
            cv2.putText(
                canvas,
                view_labels[vname],
                (ox + 10, oy + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

        # Isometric projection angles (30° standard isometric)
        iso_angle = np.radians(30)
        cos_a, sin_a = np.cos(iso_angle), np.sin(iso_angle)

        for idx, hand_landmarks in enumerate(filtered_hands):
            if hand_landmarks is None or len(hand_landmarks) == 0:
                continue

            # Calculate centroid of valid points
            valid_points = [p for p in hand_landmarks if np.linalg.norm(p) > 1e-3]
            if not valid_points:
                continue
            centroid = np.mean(valid_points, axis=0)

            # Auto-scale: find max extent from centroid and fit into quadrant
            max_extent = 0.0
            for p in valid_points:
                ext = np.max(np.abs(p - centroid))
                if ext > max_extent:
                    max_extent = ext
            if max_extent < 1.0:
                max_extent = 1.0
            # Fit the hand into the view rect (zoom)
            # Use the actual view rect width for scaling (full canvas if single view)
            ref_size = list(view_rects.values())[0][2]
            scale = (ref_size * 1.755) / max_extent

            # Per-finger colored connections: (start, end, color_BGR)
            THUMB_COLOR = (0, 200, 255)  # Orange
            INDEX_COLOR = (0, 255, 100)  # Green
            MIDDLE_COLOR = (255, 200, 0)  # Cyan-blue
            RING_COLOR = (255, 0, 150)  # Magenta
            PINKY_COLOR = (100, 100, 255)  # Red-ish
            PALM_COLOR = (180, 180, 180)  # Grey

            colored_connections = [
                # Thumb
                (0, 1, THUMB_COLOR),
                (1, 2, THUMB_COLOR),
                (2, 3, THUMB_COLOR),
                (3, 4, THUMB_COLOR),
                # Index
                (0, 5, INDEX_COLOR),
                (5, 6, INDEX_COLOR),
                (6, 7, INDEX_COLOR),
                (7, 8, INDEX_COLOR),
                # Palm base
                (5, 9, PALM_COLOR),
                (9, 13, PALM_COLOR),
                (13, 17, PALM_COLOR),
                (0, 17, PALM_COLOR),
                # Middle
                (9, 10, MIDDLE_COLOR),
                (10, 11, MIDDLE_COLOR),
                (11, 12, MIDDLE_COLOR),
                # Ring
                (13, 14, RING_COLOR),
                (14, 15, RING_COLOR),
                (15, 16, RING_COLOR),
                # Pinky
                (17, 18, PINKY_COLOR),
                (18, 19, PINKY_COLOR),
                (19, 20, PINKY_COLOR),
            ]

            def project_point(p, view_type, cx, cy):
                """Project a centered 3D point to 2D pixel coords within a view rect."""
                x, y, z = -p[0], -p[1], p[2]  # Mirror X & Y so user sees their own hand
                if view_type == "top":
                    u = int(x * scale) + cx
                    v = int(z * scale) + cy
                elif view_type == "front":
                    u = int(x * scale) + cx
                    v = int(-y * scale) + cy
                elif view_type == "side":
                    u = int(z * scale) + cx
                    v = int(-y * scale) + cy
                elif view_type == "iso":
                    u = int((x * cos_a - z * cos_a) * scale) + cx
                    v = int((-y + x * sin_a + z * sin_a) * scale * 0.8) + cy
                return u, v

            def draw_skeleton(
                view_type, rect_ox, rect_oy, rect_w, rect_h, points, override_color=None
            ):
                cx = rect_ox + rect_w // 2
                cy = rect_oy + rect_h // 2
                for start_idx, end_idx, seg_color in colored_connections:
                    p1 = points[start_idx]
                    p2 = points[end_idx]
                    if np.linalg.norm(p1) < 1e-3 or np.linalg.norm(p2) < 1e-3:
                        continue
                    c_p1 = p1 - centroid
                    c_p2 = p2 - centroid
                    u1, v1 = project_point(c_p1, view_type, cx, cy)
                    u2, v2 = project_point(c_p2, view_type, cx, cy)
                    col = override_color if override_color else seg_color
                    cv2.line(canvas, (u1, v1), (u2, v2), col, 1)
                    cv2.circle(canvas, (u1, v1), 2, col, -1)

            for vname, (ox, oy, vw, vh) in view_rects.items():
                draw_skeleton(vname, ox, oy, vw, vh, hand_landmarks)

            if self.show_raw_filtered_overlay and idx < len(raw_hands):
                raw_points = raw_hands[idx]
                for vname, (ox, oy, vw, vh) in view_rects.items():
                    draw_skeleton(
                        vname,
                        ox,
                        oy,
                        vw,
                        vh,
                        raw_points,
                        override_color=(120, 120, 120),
                    )

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
    # Lower process priority so we don't starve other applications
    try:
        import psutil

        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        print("[Priority] Set to BELOW_NORMAL")
    except Exception:
        try:
            import ctypes

            BELOW_NORMAL = 0x00004000
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL
            )
            print("[Priority] Set to BELOW_NORMAL (ctypes)")
        except Exception:
            pass  # Non-critical — just means we run at normal priority

    parser = argparse.ArgumentParser(description="Stereo Hand Tracker")
    parser.parse_args()
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
