from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

try:
    from handtrack.processing import Kalman3D, compute_3point_finger_angles
except ModuleNotFoundError:
    src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
    sys.path.insert(0, os.path.abspath(src_path))
    from handtrack.processing import Kalman3D, compute_3point_finger_angles


JOINT_ORDER = ("wrist", "thumb", "index", "middle", "ring", "pinky")
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGER_DISPLAY_INDICES = {
    "thumb": (2, 4),
    "index": (5, 8),
    "middle": (9, 12),
    "ring": (13, 16),
    "pinky": (17, 20),
}
DEFAULT_FRAME_SIZE = (1280, 720)


STYLESHEET = """
QMainWindow {
    background-color: #05070A;
}

QWidget#Root {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0 #040506,
        stop: 0.55 #0A0D12,
        stop: 1 #131820
    );
    color: #F3F6FA;
}

QFrame#GlassCard {
    background-color: rgba(17, 22, 29, 238);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 28px;
}

QFrame#PreviewCard {
    background-color: rgba(10, 13, 18, 244);
    border: 1px solid rgba(126, 211, 255, 0.2);
    border-radius: 30px;
}

QFrame#CompactBar {
    background-color: rgba(17, 22, 29, 244);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 24px;
}

QFrame#StatusChip {
    background-color: rgba(15, 19, 26, 236);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
}

QFrame#StatusChip[connected="true"] {
    background-color: rgba(18, 41, 31, 236);
    border: 1px solid rgba(74, 222, 128, 0.62);
}

QLabel#SectionTitle {
    color: #FFFFFF;
    font-size: 22px;
    font-weight: 600;
}

QLabel#HeroTitle {
    color: #FFFFFF;
    font-size: 32px;
    font-weight: 600;
}

QLabel#Muted {
    color: #8B95A5;
    font-size: 12px;
}

QLabel#TinyMuted {
    color: #6D7786;
    font-size: 11px;
}

QLabel#ChipCaption {
    color: #6D7786;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.08em;
}

QLabel#ChipValue {
    color: #F7FAFD;
    font-size: 14px;
    font-weight: 600;
}

QLabel#ChipValue[connected="true"] {
    color: #D7FBE5;
}

QLabel#PanelTitle {
    color: #FFFFFF;
    font-size: 16px;
    font-weight: 600;
}

QLabel#VideoLabel {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0 #0A0E13,
        stop: 1 #141C25
    );
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 22px;
    color: #8B95A5;
}

QPushButton {
    border-radius: 16px;
    padding: 11px 14px;
    background-color: #1A232D;
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: #F5F7FA;
}

QPushButton:hover {
    background-color: #202A36;
}

QPushButton:pressed {
    background-color: #18202A;
}

QPushButton#PrimaryButton {
    background-color: #E7ECF2;
    color: #0B1017;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

QPushButton#PrimaryButton:hover {
    background-color: #FFFFFF;
}

QLineEdit,
QComboBox,
QSpinBox,
QDoubleSpinBox,
QPlainTextEdit {
    background-color: rgba(13, 17, 24, 236);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 15px;
    padding: 10px 12px;
    color: #F8FAFC;
    selection-background-color: #6FD3FF;
}

QLineEdit:focus,
QComboBox:focus,
QSpinBox:focus,
QDoubleSpinBox:focus,
QPlainTextEdit:focus {
    border: 1px solid rgba(126, 211, 255, 0.68);
}

QComboBox::drop-down {
    border: none;
}

QScrollArea {
    border: none;
    background: transparent;
}

QAbstractScrollArea {
    background: transparent;
}

QWidget#ControlPanel,
QWidget#ScrollViewport {
    background: transparent;
}

QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 2px;
}

QScrollBar::handle:vertical {
    background: rgba(97, 109, 124, 0.62);
    border-radius: 5px;
}

QSlider::groove:horizontal {
    border: none;
    height: 8px;
    border-radius: 4px;
    background-color: rgba(26, 35, 45, 0.95);
}

QSlider::sub-page:horizontal {
    border-radius: 4px;
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 0,
        stop: 0 #82D8FF,
        stop: 1 #4BA8FF
    );
}

QSlider::handle:horizontal {
    width: 18px;
    margin: -6px 0;
    border-radius: 9px;
    background-color: #FFFFFF;
    border: 1px solid rgba(126, 211, 255, 0.42);
}
"""


@dataclass(frozen=True, slots=True)
class FingerCalibration:
    open_angle: float
    closed_angle: float
    open_command: int
    closed_command: int


@dataclass(frozen=True, slots=True)
class BridgeSettings:
    source: int | str = 0
    camera_enabled: bool = True
    udp_enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 9105
    stream_name: str = "finger-angles"
    hand_label: str = "Right"
    max_hands: int = 1
    confidence: float = 0.75
    fps: int = 30
    send_rate: float = 20.0
    wrist_angle: int = 90
    command_smoothing: float = 0.35
    open_command: int = 0
    closed_command: int = 140
    thumb_open_angle: float = 150.0
    thumb_closed_angle: float = 70.0
    finger_open_angle: float = 170.0
    finger_closed_angle: float = 70.0
    use_kalman: bool = True


def apply_theme(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    app.setFont(QtGui.QFont("Segoe UI", 10))
    app.setStyleSheet(STYLESHEET)


def parse_source(value: str | int) -> int | str:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return 0
    return int(text) if text.isdigit() else text


def map_angle_to_command(angle: float, calibration: FingerCalibration) -> int:
    source_span = calibration.closed_angle - calibration.open_angle
    if abs(source_span) < 1e-6:
        return int(round((calibration.open_command + calibration.closed_command) * 0.5))
    alpha = (float(angle) - calibration.open_angle) / source_span
    alpha = float(np.clip(alpha, 0.0, 1.0))
    command = calibration.open_command + alpha * (calibration.closed_command - calibration.open_command)
    return int(round(float(np.clip(command, 0.0, 180.0))))


def command_to_closure(command: int, calibration: FingerCalibration) -> float:
    command_span = calibration.closed_command - calibration.open_command
    if abs(command_span) < 1e-6:
        return 0.0
    return float(np.clip((command - calibration.open_command) / command_span, 0.0, 1.0))


def build_calibration(settings: BridgeSettings) -> dict[str, FingerCalibration]:
    return {
        "thumb": FingerCalibration(
            open_angle=settings.thumb_open_angle,
            closed_angle=settings.thumb_closed_angle,
            open_command=settings.open_command,
            closed_command=settings.closed_command,
        ),
        "index": FingerCalibration(
            open_angle=settings.finger_open_angle,
            closed_angle=settings.finger_closed_angle,
            open_command=settings.open_command,
            closed_command=settings.closed_command,
        ),
        "middle": FingerCalibration(
            open_angle=settings.finger_open_angle,
            closed_angle=settings.finger_closed_angle,
            open_command=settings.open_command,
            closed_command=settings.closed_command,
        ),
        "ring": FingerCalibration(
            open_angle=settings.finger_open_angle,
            closed_angle=settings.finger_closed_angle,
            open_command=settings.open_command,
            closed_command=settings.closed_command,
        ),
        "pinky": FingerCalibration(
            open_angle=settings.finger_open_angle,
            closed_angle=settings.finger_closed_angle,
            open_command=settings.open_command,
            closed_command=settings.closed_command,
        ),
    }


def build_filter_banks(max_hands: int, fps: float) -> list[list[Kalman3D]]:
    dt = 1.0 / max(float(fps), 1.0)
    return [[Kalman3D(dt=dt, process_noise=1e-3, measurement_noise=1e-4) for _ in range(21)] for _ in range(max_hands)]


def select_hand_index(results: Any, preferred_label: str) -> tuple[int | None, str | None]:
    if not results or not results.multi_hand_landmarks:
        return None, None
    handedness = results.multi_handedness or []
    preferred = preferred_label.lower()
    for index, _ in enumerate(results.multi_hand_landmarks):
        label = handedness[index].classification[0].label if index < len(handedness) else "Unknown"
        if preferred == "any" or label.lower() == preferred:
            return index, label
    return None, None


def smooth_joint_targets(current: dict[str, int], previous: dict[str, int] | None, alpha: float) -> dict[str, int]:
    if previous is None or alpha <= 0.0:
        return dict(current)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return {
        joint_name: int(round(previous.get(joint_name, current[joint_name]) * (1.0 - alpha) + current[joint_name] * alpha))
        for joint_name in current
    }


def build_udp_payload(
    stream_name: str,
    joints: dict[str, int],
    raw_angles: dict[str, float],
    calibration: dict[str, FingerCalibration],
) -> dict[str, Any]:
    inputs: dict[str, float] = {}
    for finger_name in FINGER_NAMES:
        inputs[f"{finger_name}_angle"] = round(float(raw_angles[finger_name]), 3)
        inputs[f"{finger_name}_closure"] = round(command_to_closure(joints[finger_name], calibration[finger_name]), 3)
    return {
        "stream": stream_name,
        "joint_states": joints,
        "inputs": inputs,
        "timestamp": time.time(),
    }


def build_message_preview(
    settings: BridgeSettings,
    joints: dict[str, int] | None,
    raw_angles: dict[str, float] | None,
) -> str:
    calibration = build_calibration(settings)
    preview_joints = dict(joints or {"wrist": settings.wrist_angle, "thumb": settings.open_command, "index": settings.open_command, "middle": settings.open_command, "ring": settings.open_command, "pinky": settings.open_command})
    preview_angles = dict(raw_angles or {finger_name: 0.0 for finger_name in FINGER_NAMES})
    preview_inputs: dict[str, float] = {}
    for finger_name in FINGER_NAMES:
        preview_inputs[f"{finger_name}_angle"] = round(float(preview_angles[finger_name]), 3)
        preview_inputs[f"{finger_name}_closure"] = round(command_to_closure(preview_joints[finger_name], calibration[finger_name]), 3)

    payload = {
        "stream": settings.stream_name,
        "joint_states": preview_joints,
        "inputs": preview_inputs,
        "timestamp": round(time.time(), 3),
    }
    return json.dumps(payload, indent=2)


def draw_overlay(
    frame: np.ndarray,
    hand_label: str | None,
    joints: dict[str, int] | None,
    raw_angles: dict[str, float] | None,
    settings: BridgeSettings,
    camera_status: str,
    udp_status: str,
) -> None:
    lines = [
        f"Camera: {camera_status}",
        f"UDP: {udp_status}  |  Target: {settings.host}:{settings.port}",
        f"Hand: {hand_label or 'none'}  |  Stream: {settings.stream_name}",
    ]
    if joints:
        lines.append("Commands: " + "  ".join(f"{joint_name[:2]}={joints[joint_name]:3d}" for joint_name in JOINT_ORDER))
    if raw_angles:
        lines.append("Angles:   " + "  ".join(f"{finger_name[:2]}={raw_angles[finger_name]:5.1f}" for finger_name in FINGER_NAMES))

    y = 32
    for line in lines:
        cv2.putText(frame, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (244, 248, 252), 2, cv2.LINE_AA)
        y += 30


def draw_finger_angle_overlay(
    frame: np.ndarray,
    filtered_landmarks: np.ndarray,
    joints: dict[str, int],
    raw_angles: dict[str, float],
) -> None:
    height, width = frame.shape[:2]
    for finger_name, (mcp_idx, tip_idx) in FINGER_DISPLAY_INDICES.items():
        mcp = filtered_landmarks[mcp_idx]
        tip = filtered_landmarks[tip_idx]

        mcp_px = (int(mcp[0] * width), int(mcp[1] * height))
        tip_px = (int(tip[0] * width), int(tip[1] * height))
        label_anchor = (
            int(mcp_px[0] * 0.35 + tip_px[0] * 0.65),
            int(mcp_px[1] * 0.35 + tip_px[1] * 0.65),
        )

        command_text = f"{finger_name[:2].upper()} {joints[finger_name]:3d}deg"
        raw_text = f"{raw_angles[finger_name]:.0f}"
        text_size, _ = cv2.getTextSize(command_text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        box_w = text_size[0] + 12
        box_h = 20
        box_x = int(np.clip(label_anchor[0] - box_w // 2, 4, max(4, width - box_w - 4)))
        box_y = int(np.clip(label_anchor[1] - 24, 4, max(4, height - box_h - 4)))

        cv2.line(frame, mcp_px, tip_px, (111, 211, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, mcp_px, 4, (255, 225, 132), -1)
        cv2.circle(frame, tip_px, 4, (255, 225, 132), -1)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (10, 15, 22), -1)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (111, 211, 255), 1)
        cv2.putText(frame, command_text, (box_x + 6, box_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (244, 248, 252), 1, cv2.LINE_AA)
        cv2.putText(frame, raw_text, (tip_px[0] + 6, tip_px[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


def make_placeholder_frame(title: str, subtitle: str, detail: str) -> np.ndarray:
    width, height = DEFAULT_FRAME_SIZE
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    gradient = np.linspace(16, 36, width, dtype=np.uint8)
    frame[:, :, 0] = gradient
    frame[:, :, 1] = gradient // 2
    frame[:, :, 2] = gradient // 4
    cv2.putText(frame, title, (54, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (247, 250, 253), 3, cv2.LINE_AA)
    cv2.putText(frame, subtitle, (56, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (139, 149, 165), 2, cv2.LINE_AA)
    cv2.putText(frame, detail, (56, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (109, 119, 134), 2, cv2.LINE_AA)
    return frame


def frame_to_qimage(frame: np.ndarray) -> QtGui.QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    height, width, channels = rgb.shape
    bytes_per_line = channels * width
    return QtGui.QImage(rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).copy()


class ToggleSwitch(QtWidgets.QAbstractButton):
    def __init__(self, on_text: str = "ON", off_text: str = "OFF", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._on_text = on_text
        self._off_text = off_text
        self.setCheckable(True)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setFixedSize(62, 32)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(62, 32)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        rect = QtCore.QRectF(self.rect()).adjusted(2, 2, -2, -2)
        radius = rect.height() / 2
        checked = self.isChecked()
        track_color = QtGui.QColor("#7ED3FF") if checked else QtGui.QColor("#2A3440")
        knob_color = QtGui.QColor("#F8FAFC")
        text_color = QtGui.QColor("#071018") if checked else QtGui.QColor("#AEB8C6")

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(track_color)
        painter.drawRoundedRect(rect, radius, radius)

        knob_diameter = rect.height() - 6
        knob_x = rect.right() - knob_diameter - 3 if checked else rect.left() + 3
        knob_rect = QtCore.QRectF(knob_x, rect.top() + 3, knob_diameter, knob_diameter)
        painter.setBrush(knob_color)
        painter.drawEllipse(knob_rect)

        painter.setPen(text_color)
        font = QtGui.QFont("Segoe UI", 8)
        font.setBold(True)
        painter.setFont(font)
        text_rect = rect.adjusted(10, 0, -10, 0)
        painter.drawText(text_rect, QtCore.Qt.AlignCenter, self._on_text if checked else self._off_text)


class VideoPreview(QtWidgets.QLabel):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("VideoLabel")
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(760, 520)
        self._pixmap: QtGui.QPixmap | None = None
        self.setText("Starting camera preview...")

    def set_frame(self, image: QtGui.QImage) -> None:
        self._pixmap = QtGui.QPixmap.fromImage(image)
        self._render_pixmap()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._render_pixmap()

    def _render_pixmap(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.setPixmap(scaled)


class FingerAngleBridgeWorker(QtCore.QObject):
    frame_ready = QtCore.pyqtSignal(QtGui.QImage)
    snapshot_ready = QtCore.pyqtSignal(object)
    log_message = QtCore.pyqtSignal(str)

    def __init__(self, settings: BridgeSettings) -> None:
        super().__init__()
        self._settings = settings
        self._settings_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._capture: cv2.VideoCapture | None = None
        self._hands: Any | None = None
        self._filters: list[list[Kalman3D]] = []
        self._calibration = build_calibration(settings)
        self._socket: socket.socket | None = None
        self._socket_broadcast = False
        self._capture_signature: tuple[Any, ...] | None = None

        self._last_joints: dict[str, int] | None = None
        self._last_send_at = 0.0
        self._last_frame_at = 0.0
        self._display_fps = 0.0
        self._packets_sent = 0
        self._last_sent_at = 0.0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="finger-angle-bridge", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._release_resources()

    def update_settings(self, settings: BridgeSettings) -> None:
        with self._settings_lock:
            self._settings = settings
            self._calibration = build_calibration(settings)

    def _current_settings(self) -> BridgeSettings:
        with self._settings_lock:
            return self._settings

    def _capture_key(self, settings: BridgeSettings) -> tuple[Any, ...]:
        return (
            settings.source,
            settings.max_hands,
            round(settings.confidence, 3),
            settings.fps,
            settings.use_kalman,
        )

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            settings = self._current_settings()
            if not settings.camera_enabled:
                self._release_capture()
                frame = make_placeholder_frame(
                    "Camera Disabled",
                    "Enable Live Camera on the right to resume tracking.",
                    f"UDP target remains {settings.host}:{settings.port} for when you resume.",
                )
                self._emit_frame(frame)
                self._emit_snapshot(settings, camera_live=False, detected_label=None, joints=self._last_joints, raw_angles=None)
                self._stop_event.wait(0.12)
                continue

            if not self._ensure_capture(settings):
                frame = make_placeholder_frame(
                    "Camera Unavailable",
                    "The selected source could not be opened.",
                    "Check the camera index or path, then apply the camera settings again.",
                )
                self._emit_frame(frame)
                self._emit_snapshot(settings, camera_live=False, detected_label=None, joints=self._last_joints, raw_angles=None)
                self._stop_event.wait(0.35)
                continue

            success, frame = self._capture.read() if self._capture is not None else (False, None)
            if not success or frame is None:
                self.log_message.emit(f"Source {settings.source!r} stopped delivering frames; retrying.")
                self._release_capture()
                self._stop_event.wait(0.2)
                continue

            frame = cv2.flip(frame, 1)
            now = time.time()
            if self._last_frame_at > 0.0:
                instantaneous = 1.0 / max(now - self._last_frame_at, 1e-6)
                self._display_fps = instantaneous if self._display_fps <= 0.0 else self._display_fps * 0.82 + instantaneous * 0.18
            self._last_frame_at = now

            joints: dict[str, int] | None = None
            raw_angles: dict[str, float] | None = None
            detected_label: str | None = None

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._hands.process(rgb) if self._hands is not None else None
            hand_index, detected_label = select_hand_index(results, settings.hand_label)

            if results and results.multi_hand_landmarks and hand_index is not None:
                hand_landmarks = results.multi_hand_landmarks[hand_index]
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style(),
                )

                landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
                if settings.use_kalman:
                    filter_bank = self._filters[min(hand_index, len(self._filters) - 1)]
                    filtered = np.array([filter_bank[i].update(landmark_array[i]) for i in range(21)], dtype=np.float32)
                    for x, y, _ in filtered:
                        cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 3, (255, 255, 255), -1)
                else:
                    filtered = landmark_array

                raw_angles = compute_3point_finger_angles(filtered)
                unsmoothed = {
                    "wrist": int(np.clip(settings.wrist_angle, 0, 180)),
                    "thumb": map_angle_to_command(raw_angles["thumb"], self._calibration["thumb"]),
                    "index": map_angle_to_command(raw_angles["index"], self._calibration["index"]),
                    "middle": map_angle_to_command(raw_angles["middle"], self._calibration["middle"]),
                    "ring": map_angle_to_command(raw_angles["ring"], self._calibration["ring"]),
                    "pinky": map_angle_to_command(raw_angles["pinky"], self._calibration["pinky"]),
                }
                joints = smooth_joint_targets(unsmoothed, self._last_joints, settings.command_smoothing)
                draw_finger_angle_overlay(frame, filtered, joints, raw_angles)

                if settings.udp_enabled and self._should_send_packet(settings, now):
                    try:
                        self._ensure_socket(settings)
                        payload = build_udp_payload(settings.stream_name, joints, raw_angles, self._calibration)
                        if self._socket is not None:
                            self._socket.sendto(json.dumps(payload).encode("utf-8"), (settings.host, settings.port))
                            self._packets_sent += 1
                            self._last_sent_at = now
                            self._last_send_at = now
                    except OSError as exc:
                        self.log_message.emit(f"UDP send failed: {exc}")

                self._last_joints = dict(joints)

            camera_status = "LIVE" if settings.camera_enabled else "OFF"
            udp_status = "SENDING" if settings.udp_enabled else "PAUSED"
            draw_overlay(frame, detected_label, joints or self._last_joints, raw_angles, settings, camera_status, udp_status)
            self._emit_frame(frame)
            self._emit_snapshot(settings, camera_live=True, detected_label=detected_label, joints=joints or self._last_joints, raw_angles=raw_angles)

            wait_time = 1.0 / max(float(settings.fps), 1.0)
            self._stop_event.wait(min(wait_time, 0.05))

    def _ensure_capture(self, settings: BridgeSettings) -> bool:
        desired_key = self._capture_key(settings)
        if self._capture is not None and self._hands is not None and desired_key == self._capture_signature:
            return True

        self._release_capture()
        capture = cv2.VideoCapture(settings.source)
        if not capture.isOpened():
            capture.release()
            return False

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_FRAME_SIZE[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_FRAME_SIZE[1])
        capture.set(cv2.CAP_PROP_FPS, settings.fps)

        hands = mp.solutions.hands.Hands(
            max_num_hands=max(1, settings.max_hands),
            min_detection_confidence=settings.confidence,
            min_tracking_confidence=0.7,
        )

        self._capture = capture
        self._hands = hands
        self._filters = build_filter_banks(max(1, settings.max_hands), settings.fps)
        self._capture_signature = desired_key
        self._last_joints = None
        self.log_message.emit(f"Camera source {settings.source!r} ready.")
        return True

    def _ensure_socket(self, settings: BridgeSettings) -> None:
        wants_broadcast = settings.host.endswith(".255") or settings.host == "255.255.255.255"
        if self._socket is not None and wants_broadcast == self._socket_broadcast:
            return
        self._release_socket()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if wants_broadcast:
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._socket_broadcast = wants_broadcast

    def _should_send_packet(self, settings: BridgeSettings, now: float) -> bool:
        interval = 1.0 / max(float(settings.send_rate), 1.0)
        return (now - self._last_send_at) >= interval

    def _emit_frame(self, frame: np.ndarray) -> None:
        self.frame_ready.emit(frame_to_qimage(frame))

    def _emit_snapshot(
        self,
        settings: BridgeSettings,
        *,
        camera_live: bool,
        detected_label: str | None,
        joints: dict[str, int] | None,
        raw_angles: dict[str, float] | None,
    ) -> None:
        snapshot = {
            "camera_enabled": settings.camera_enabled,
            "camera_live": camera_live,
            "udp_enabled": settings.udp_enabled,
            "stream_name": settings.stream_name,
            "target": f"{settings.host}:{settings.port}",
            "detected_hand": detected_label or "None",
            "fps": self._display_fps,
            "packets_sent": self._packets_sent,
            "last_send_ms": (time.time() - self._last_sent_at) * 1000.0 if self._last_sent_at else None,
            "joints": joints or {},
            "raw_angles": raw_angles or {},
        }
        self.snapshot_ready.emit(snapshot)

    def _release_capture(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if self._hands is not None:
            self._hands.close()
            self._hands = None
        self._filters = []
        self._capture_signature = None

    def _release_socket(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._socket_broadcast = False

    def _release_resources(self) -> None:
        self._release_capture()
        self._release_socket()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, initial_settings: BridgeSettings) -> None:
        super().__init__()
        self._settings = initial_settings
        self._snapshot: dict[str, Any] = {}
        self.setWindowTitle("Finger Angle Stream Console")
        self.resize(1420, 900)
        self.setMinimumSize(1220, 760)

        root = QtWidgets.QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)

        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(14)

        outer.addWidget(self._build_header())

        body = QtWidgets.QGridLayout()
        body.setHorizontalSpacing(14)
        body.setVerticalSpacing(14)
        outer.addLayout(body, stretch=1)

        body.addWidget(self._build_preview_card(), 0, 0)
        body.addWidget(self._build_control_panel(), 0, 1)
        body.setColumnStretch(0, 7)
        body.setColumnStretch(1, 4)

        self.worker = FingerAngleBridgeWorker(initial_settings)
        self.worker.frame_ready.connect(self.preview.set_frame)
        self.worker.snapshot_ready.connect(self._refresh_snapshot)
        self.worker.log_message.connect(self._append_log)
        self.worker.start()
        self._push_settings("Applied startup defaults.")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.worker.stop()
        super().closeEvent(event)

    def _build_header(self) -> QtWidgets.QWidget:
        card = QtWidgets.QFrame()
        card.setObjectName("CompactBar")
        layout = QtWidgets.QHBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)

        left = QtWidgets.QVBoxLayout()
        left.setSpacing(2)

        title = QtWidgets.QLabel("Finger Angle Stream")
        title.setObjectName("HeroTitle")
        subtitle = QtWidgets.QLabel("Camera preview on the left, live routing controls on the right")
        subtitle.setObjectName("Muted")
        left.addWidget(title)
        left.addWidget(subtitle)
        layout.addLayout(left, stretch=3)
        layout.addStretch(1)

        chips = QtWidgets.QHBoxLayout()
        chips.setSpacing(10)
        chips.addWidget(self._status_chip("CAMERA", "Starting", "camera_chip"))
        chips.addWidget(self._status_chip("UDP", "Waiting", "udp_chip"))
        chips.addWidget(self._status_chip("HAND", "None", "hand_chip"))
        chips.addWidget(self._status_chip("FPS", "0.0", "fps_chip"))
        layout.addLayout(chips)
        return card

    def _status_chip(self, caption: str, value: str, attr_name: str) -> QtWidgets.QFrame:
        chip = QtWidgets.QFrame()
        chip.setObjectName("StatusChip")
        layout = QtWidgets.QVBoxLayout(chip)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(2)

        caption_label = QtWidgets.QLabel(caption)
        caption_label.setObjectName("ChipCaption")
        value_label = QtWidgets.QLabel(value)
        value_label.setObjectName("ChipValue")

        layout.addWidget(caption_label)
        layout.addWidget(value_label)
        setattr(self, attr_name, value_label)
        setattr(self, f"{attr_name}_frame", chip)
        return chip

    def _build_preview_card(self) -> QtWidgets.QWidget:
        card = QtWidgets.QFrame()
        card.setObjectName("PreviewCard")
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        heading = QtWidgets.QHBoxLayout()
        heading.setSpacing(12)
        title = QtWidgets.QLabel("Camera View")
        title.setObjectName("SectionTitle")
        subtitle = QtWidgets.QLabel("Live landmarks and sent command overlays")
        subtitle.setObjectName("Muted")
        heading_text = QtWidgets.QVBoxLayout()
        heading_text.setSpacing(2)
        heading_text.addWidget(title)
        heading_text.addWidget(subtitle)
        heading.addLayout(heading_text)
        heading.addStretch(1)
        layout.addLayout(heading)

        self.preview = VideoPreview()
        layout.addWidget(self.preview, stretch=1)

        footer = QtWidgets.QHBoxLayout()
        footer.setSpacing(12)
        self.preview_status_label = QtWidgets.QLabel("Waiting for camera...")
        self.preview_status_label.setObjectName("TinyMuted")
        self.command_summary_label = QtWidgets.QLabel("No command frames sent yet.")
        self.command_summary_label.setObjectName("TinyMuted")
        footer.addWidget(self.preview_status_label, stretch=1)
        footer.addWidget(self.command_summary_label, stretch=1)
        layout.addLayout(footer)
        return card

    def _build_control_panel(self) -> QtWidgets.QWidget:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.viewport().setObjectName("ScrollViewport")

        panel = QtWidgets.QWidget()
        panel.setObjectName("ControlPanel")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        layout.addWidget(self._build_camera_card())
        layout.addWidget(self._build_udp_card())
        layout.addWidget(self._build_mapping_card())
        layout.addWidget(self._build_status_card())
        layout.addWidget(self._build_message_card())
        layout.addStretch(1)

        scroll.setWidget(panel)
        return scroll

    def _build_camera_card(self) -> QtWidgets.QWidget:
        card = self._card()
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        layout.addWidget(self._panel_title("Camera"))

        toggle_row = QtWidgets.QHBoxLayout()
        toggle_row.setSpacing(10)
        camera_label = QtWidgets.QLabel("Live Camera")
        camera_label.setObjectName("Muted")
        self.camera_switch = ToggleSwitch("LIVE", "OFF")
        self.camera_switch.setChecked(self._settings.camera_enabled)
        self.camera_switch.toggled.connect(lambda _: self._push_settings("Camera state updated."))
        toggle_row.addWidget(camera_label)
        toggle_row.addStretch(1)
        toggle_row.addWidget(self.camera_switch)
        layout.addLayout(toggle_row)

        self.source_input = QtWidgets.QLineEdit(str(self._settings.source))
        self.source_input.setPlaceholderText("0 for default webcam, or a video file path")
        source_row = QtWidgets.QHBoxLayout()
        source_row.setSpacing(8)
        source_row.addWidget(self.source_input, stretch=1)
        browse_button = QtWidgets.QPushButton("Browse")
        browse_button.clicked.connect(self._browse_source)
        source_row.addWidget(browse_button)
        layout.addWidget(self._field_label("Camera index or source path"))
        layout.addLayout(source_row)

        self.hand_label_combo = QtWidgets.QComboBox()
        self.hand_label_combo.addItems(["Any", "Left", "Right"])
        self.hand_label_combo.setCurrentText(self._settings.hand_label)
        layout.addWidget(self._field_label("Preferred hand"))
        layout.addWidget(self.hand_label_combo)

        two_col = QtWidgets.QHBoxLayout()
        two_col.setSpacing(10)
        self.max_hands_input = QtWidgets.QSpinBox()
        self.max_hands_input.setRange(1, 2)
        self.max_hands_input.setValue(self._settings.max_hands)
        self.confidence_input = QtWidgets.QDoubleSpinBox()
        self.confidence_input.setRange(0.1, 0.99)
        self.confidence_input.setDecimals(2)
        self.confidence_input.setSingleStep(0.05)
        self.confidence_input.setValue(self._settings.confidence)
        two_col.addLayout(self._field_stack("Max hands", self.max_hands_input))
        two_col.addLayout(self._field_stack("Detection confidence", self.confidence_input))
        layout.addLayout(two_col)

        helper_row = QtWidgets.QHBoxLayout()
        helper_row.setSpacing(10)
        kalman_label = QtWidgets.QLabel("Kalman smoothing")
        kalman_label.setObjectName("Muted")
        self.kalman_switch = ToggleSwitch("ON", "OFF")
        self.kalman_switch.setChecked(self._settings.use_kalman)
        self.kalman_switch.toggled.connect(lambda _: self._push_settings("Camera pipeline updated."))
        helper_row.addWidget(kalman_label)
        helper_row.addStretch(1)
        helper_row.addWidget(self.kalman_switch)
        layout.addLayout(helper_row)

        apply_button = QtWidgets.QPushButton("Apply Camera")
        apply_button.setObjectName("PrimaryButton")
        apply_button.clicked.connect(lambda: self._push_settings("Camera settings applied."))
        layout.addWidget(apply_button)

        helper = QtWidgets.QLabel("Default mode keeps the camera live and streams overlays directly inside this window.")
        helper.setObjectName("TinyMuted")
        helper.setWordWrap(True)
        layout.addWidget(helper)
        return card

    def _build_udp_card(self) -> QtWidgets.QWidget:
        card = self._card()
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        layout.addWidget(self._panel_title("UDP"))

        toggle_row = QtWidgets.QHBoxLayout()
        toggle_row.setSpacing(10)
        udp_label = QtWidgets.QLabel("Send Commands")
        udp_label.setObjectName("Muted")
        self.udp_switch = ToggleSwitch("SEND", "PAUSE")
        self.udp_switch.setChecked(self._settings.udp_enabled)
        self.udp_switch.toggled.connect(lambda _: self._push_settings("UDP state updated."))
        toggle_row.addWidget(udp_label)
        toggle_row.addStretch(1)
        toggle_row.addWidget(self.udp_switch)
        layout.addLayout(toggle_row)

        self.host_input = QtWidgets.QLineEdit(self._settings.host)
        self.port_input = QtWidgets.QSpinBox()
        self.port_input.setRange(1024, 65535)
        self.port_input.setValue(self._settings.port)
        self.stream_name_input = QtWidgets.QLineEdit(self._settings.stream_name)
        self.send_rate_input = QtWidgets.QDoubleSpinBox()
        self.send_rate_input.setRange(1.0, 240.0)
        self.send_rate_input.setDecimals(1)
        self.send_rate_input.setValue(self._settings.send_rate)
        self.send_rate_input.setSuffix(" Hz")

        layout.addWidget(self._field_label("Host"))
        layout.addWidget(self.host_input)
        layout.addWidget(self._field_label("Port"))
        layout.addWidget(self.port_input)
        layout.addWidget(self._field_label("Stream name"))
        layout.addWidget(self.stream_name_input)
        layout.addWidget(self._field_label("Send rate"))
        layout.addWidget(self.send_rate_input)

        apply_button = QtWidgets.QPushButton("Apply UDP")
        apply_button.setObjectName("PrimaryButton")
        apply_button.clicked.connect(lambda: self._push_settings("UDP settings applied."))
        layout.addWidget(apply_button)
        return card

    def _build_mapping_card(self) -> QtWidgets.QWidget:
        card = self._card()
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        layout.addWidget(self._panel_title("Mapping"))

        self.wrist_angle_input = QtWidgets.QSpinBox()
        self.wrist_angle_input.setRange(0, 180)
        self.wrist_angle_input.setValue(self._settings.wrist_angle)

        self.command_smoothing_input = QtWidgets.QDoubleSpinBox()
        self.command_smoothing_input.setRange(0.0, 1.0)
        self.command_smoothing_input.setDecimals(2)
        self.command_smoothing_input.setSingleStep(0.05)
        self.command_smoothing_input.setValue(self._settings.command_smoothing)

        self.open_command_input = QtWidgets.QSpinBox()
        self.open_command_input.setRange(0, 180)
        self.open_command_input.setValue(self._settings.open_command)
        self.closed_command_input = QtWidgets.QSpinBox()
        self.closed_command_input.setRange(0, 180)
        self.closed_command_input.setValue(self._settings.closed_command)

        self.thumb_open_angle_input = QtWidgets.QDoubleSpinBox()
        self.thumb_open_angle_input.setRange(0.0, 180.0)
        self.thumb_open_angle_input.setDecimals(1)
        self.thumb_open_angle_input.setValue(self._settings.thumb_open_angle)
        self.thumb_closed_angle_input = QtWidgets.QDoubleSpinBox()
        self.thumb_closed_angle_input.setRange(0.0, 180.0)
        self.thumb_closed_angle_input.setDecimals(1)
        self.thumb_closed_angle_input.setValue(self._settings.thumb_closed_angle)
        self.finger_open_angle_input = QtWidgets.QDoubleSpinBox()
        self.finger_open_angle_input.setRange(0.0, 180.0)
        self.finger_open_angle_input.setDecimals(1)
        self.finger_open_angle_input.setValue(self._settings.finger_open_angle)
        self.finger_closed_angle_input = QtWidgets.QDoubleSpinBox()
        self.finger_closed_angle_input.setRange(0.0, 180.0)
        self.finger_closed_angle_input.setDecimals(1)
        self.finger_closed_angle_input.setValue(self._settings.finger_closed_angle)

        layout.addLayout(self._field_stack("Wrist angle", self.wrist_angle_input))
        layout.addLayout(self._field_stack("Command smoothing", self.command_smoothing_input))
        layout.addLayout(self._field_stack("Open command", self.open_command_input))
        layout.addLayout(self._field_stack("Closed command", self.closed_command_input))
        layout.addLayout(self._field_stack("Thumb open angle", self.thumb_open_angle_input))
        layout.addLayout(self._field_stack("Thumb closed angle", self.thumb_closed_angle_input))
        layout.addLayout(self._field_stack("Finger open angle", self.finger_open_angle_input))
        layout.addLayout(self._field_stack("Finger closed angle", self.finger_closed_angle_input))

        button_row = QtWidgets.QHBoxLayout()
        button_row.setSpacing(8)
        apply_button = QtWidgets.QPushButton("Apply Mapping")
        apply_button.setObjectName("PrimaryButton")
        apply_button.clicked.connect(lambda: self._push_settings("Mapping updated."))
        reset_button = QtWidgets.QPushButton("Reset")
        reset_button.clicked.connect(self._reset_mapping_defaults)
        button_row.addWidget(apply_button)
        button_row.addWidget(reset_button)
        layout.addLayout(button_row)
        return card

    def _build_status_card(self) -> QtWidgets.QWidget:
        card = self._card()
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        layout.addWidget(self._panel_title("Status"))

        self.status_text = QtWidgets.QLabel("Awaiting tracking frames.")
        self.status_text.setObjectName("Muted")
        self.status_text.setWordWrap(True)
        layout.addWidget(self.status_text)

        self.last_event_label = QtWidgets.QLabel("No events yet.")
        self.last_event_label.setObjectName("TinyMuted")
        self.last_event_label.setWordWrap(True)
        layout.addWidget(self.last_event_label)

        self.angle_summary_label = QtWidgets.QLabel("Finger angles will appear here once a hand is detected.")
        self.angle_summary_label.setObjectName("TinyMuted")
        self.angle_summary_label.setWordWrap(True)
        layout.addWidget(self.angle_summary_label)
        return card

    def _build_message_card(self) -> QtWidgets.QWidget:
        card = self._card()
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        layout.addWidget(self._panel_title("Message Structure"))

        helper = QtWidgets.QLabel("Outgoing command payload sent to embedded_control_station.")
        helper.setObjectName("Muted")
        helper.setWordWrap(True)
        layout.addWidget(helper)

        self.message_structure_view = QtWidgets.QPlainTextEdit()
        self.message_structure_view.setReadOnly(True)
        self.message_structure_view.setMinimumHeight(220)
        self.message_structure_view.setPlainText(build_message_preview(self._settings, None, None))
        layout.addWidget(self.message_structure_view)

        footer = QtWidgets.QLabel("The preview updates with the current stream name and the latest detected command values.")
        footer.setObjectName("TinyMuted")
        footer.setWordWrap(True)
        layout.addWidget(footer)
        return card

    def _card(self) -> QtWidgets.QFrame:
        card = QtWidgets.QFrame()
        card.setObjectName("GlassCard")
        return card

    def _panel_title(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("PanelTitle")
        return label

    def _field_label(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("Muted")
        return label

    def _field_stack(self, text: str, widget: QtWidgets.QWidget) -> QtWidgets.QVBoxLayout:
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(6)
        layout.addWidget(self._field_label(text))
        layout.addWidget(widget)
        return layout

    def _browse_source(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Video Source",
            "",
            "Video Files (*.mp4 *.mov *.avi *.mkv);;All Files (*)",
        )
        if path:
            self.source_input.setText(path)
            self._push_settings("Video source selected.")

    def _collect_settings(self) -> BridgeSettings:
        return BridgeSettings(
            source=parse_source(self.source_input.text()),
            camera_enabled=self.camera_switch.isChecked(),
            udp_enabled=self.udp_switch.isChecked(),
            host=self.host_input.text().strip() or "127.0.0.1",
            port=self.port_input.value(),
            stream_name=self.stream_name_input.text().strip() or "finger-angles",
            hand_label=self.hand_label_combo.currentText(),
            max_hands=self.max_hands_input.value(),
            confidence=self.confidence_input.value(),
            fps=self._settings.fps,
            send_rate=self.send_rate_input.value(),
            wrist_angle=self.wrist_angle_input.value(),
            command_smoothing=self.command_smoothing_input.value(),
            open_command=self.open_command_input.value(),
            closed_command=self.closed_command_input.value(),
            thumb_open_angle=self.thumb_open_angle_input.value(),
            thumb_closed_angle=self.thumb_closed_angle_input.value(),
            finger_open_angle=self.finger_open_angle_input.value(),
            finger_closed_angle=self.finger_closed_angle_input.value(),
            use_kalman=self.kalman_switch.isChecked(),
        )

    def _push_settings(self, message: str | None = None) -> None:
        self._settings = self._collect_settings()
        self.worker.update_settings(self._settings)
        if hasattr(self, "message_structure_view"):
            self.message_structure_view.setPlainText(build_message_preview(self._settings, self._snapshot.get("joints"), self._snapshot.get("raw_angles")))
        if message:
            self._append_log(message)

    def _reset_mapping_defaults(self) -> None:
        defaults = BridgeSettings()
        self.wrist_angle_input.setValue(defaults.wrist_angle)
        self.command_smoothing_input.setValue(defaults.command_smoothing)
        self.open_command_input.setValue(defaults.open_command)
        self.closed_command_input.setValue(defaults.closed_command)
        self.thumb_open_angle_input.setValue(defaults.thumb_open_angle)
        self.thumb_closed_angle_input.setValue(defaults.thumb_closed_angle)
        self.finger_open_angle_input.setValue(defaults.finger_open_angle)
        self.finger_closed_angle_input.setValue(defaults.finger_closed_angle)
        self._push_settings("Mapping reset to defaults.")

    def _set_connected_style(self, widget: QtWidgets.QWidget, connected: bool) -> None:
        widget.setProperty("connected", connected)
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _refresh_snapshot(self, snapshot: dict[str, Any]) -> None:
        self._snapshot = snapshot

        camera_connected = bool(snapshot.get("camera_live")) and bool(snapshot.get("camera_enabled"))
        udp_connected = bool(snapshot.get("udp_enabled"))

        self.camera_chip.setText("Live" if camera_connected else "Off")
        self.udp_chip.setText("Sending" if udp_connected else "Paused")
        self.hand_chip.setText(str(snapshot.get("detected_hand", "None")))
        self.fps_chip.setText(f"{snapshot.get('fps', 0.0):.1f}")

        self._set_connected_style(self.camera_chip_frame, camera_connected)
        self._set_connected_style(self.udp_chip_frame, udp_connected)
        self.camera_chip.setProperty("connected", camera_connected)
        self.udp_chip.setProperty("connected", udp_connected)
        self.camera_chip.style().unpolish(self.camera_chip)
        self.camera_chip.style().polish(self.camera_chip)
        self.udp_chip.style().unpolish(self.udp_chip)
        self.udp_chip.style().polish(self.udp_chip)

        hand_text = snapshot.get("detected_hand", "None")
        packets_sent = snapshot.get("packets_sent", 0)
        target = snapshot.get("target", "127.0.0.1:9105")
        self.preview_status_label.setText(f"Hand: {hand_text}  |  Target: {target}")
        self.status_text.setText(
            f"Camera {'live' if camera_connected else 'paused'} • UDP {'active' if udp_connected else 'paused'} • {packets_sent} packets sent"
        )

        joints = snapshot.get("joints") or {}
        raw_angles = snapshot.get("raw_angles") or {}
        if joints:
            self.command_summary_label.setText(
                "Commands: " + "  ".join(f"{joint[:2]}={joints[joint]:3d}" for joint in JOINT_ORDER)
            )
        else:
            self.command_summary_label.setText("No active joint command yet.")

        if raw_angles:
            self.angle_summary_label.setText(
                "Angles: " + "  ".join(f"{finger[:2]}={raw_angles[finger]:5.1f}" for finger in FINGER_NAMES)
            )
        else:
            self.angle_summary_label.setText("Finger angles will appear here once a hand is detected.")

        if hasattr(self, "message_structure_view"):
            self.message_structure_view.setPlainText(build_message_preview(self._settings, joints, raw_angles))

    def _append_log(self, message: str) -> None:
        self.last_event_label.setText(message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream live finger angles to embedded_control_station over UDP")
    parser.add_argument("--source", default=0, type=lambda value: int(value) if str(value).isdigit() else value)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9105)
    parser.add_argument("--stream-name", default="finger-angles")
    parser.add_argument("--hand-label", choices=["Any", "Left", "Right"], default="Right")
    parser.add_argument("--max-hands", type=int, default=1)
    parser.add_argument("--confidence", type=float, default=0.75)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--send-rate", type=float, default=20.0)
    parser.add_argument("--wrist-angle", type=int, default=90)
    parser.add_argument("--command-smoothing", type=float, default=0.35)
    parser.add_argument("--open-command", type=int, default=0)
    parser.add_argument("--closed-command", type=int, default=140)
    parser.add_argument("--thumb-open-angle", type=float, default=150.0)
    parser.add_argument("--thumb-closed-angle", type=float, default=70.0)
    parser.add_argument("--finger-open-angle", type=float, default=170.0)
    parser.add_argument("--finger-closed-angle", type=float, default=70.0)
    parser.add_argument("--camera-disabled", action="store_true")
    parser.add_argument("--udp-disabled", action="store_true")
    parser.add_argument("--no-kalman", action="store_true")
    return parser


def build_initial_settings(args: argparse.Namespace) -> BridgeSettings:
    return BridgeSettings(
        source=args.source,
        camera_enabled=not args.camera_disabled,
        udp_enabled=not args.udp_disabled,
        host=args.host,
        port=args.port,
        stream_name=args.stream_name,
        hand_label=args.hand_label,
        max_hands=args.max_hands,
        confidence=args.confidence,
        fps=args.fps,
        send_rate=args.send_rate,
        wrist_angle=args.wrist_angle,
        command_smoothing=args.command_smoothing,
        open_command=args.open_command,
        closed_command=args.closed_command,
        thumb_open_angle=args.thumb_open_angle,
        thumb_closed_angle=args.thumb_closed_angle,
        finger_open_angle=args.finger_open_angle,
        finger_closed_angle=args.finger_closed_angle,
        use_kalman=not args.no_kalman,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Finger Angle Stream Console")
    app.setOrganizationName("Local")
    apply_theme(app)

    window = MainWindow(build_initial_settings(args))
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())