"""Unified single-camera GUI for MediaPipe hands, face, pose, and holistic."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Sequence

import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from handtrack.tracker import MediaPipeModeTracker, TrackingMode


_MODE_LABELS = {
    "Hands": TrackingMode.HANDS,
    "Face": TrackingMode.FACE,
    "Pose": TrackingMode.POSE,
    "Everything": TrackingMode.HOLISTIC,
}


class MediaPipeTrackerGUI(QMainWindow):
    """A compact preview GUI with runtime-selectable body tracking."""

    def __init__(self, source: int | str = 0, mode: str = "hands") -> None:
        super().__init__()
        self.source = source
        self.capture: cv2.VideoCapture | None = None
        self.tracker = MediaPipeModeTracker(mode)
        self.frames = 0
        self.fps_started = time.perf_counter()

        self.setWindowTitle("HandTrack · MediaPipe Studio")
        self.resize(980, 720)
        self._build_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)
        self._start()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        controls = QHBoxLayout()

        controls.addWidget(QLabel("Tracking mode"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(_MODE_LABELS)
        selected_label = next(
            label
            for label, value in _MODE_LABELS.items()
            if value is self.tracker.mode
        )
        self.mode_combo.setCurrentText(selected_label)
        self.mode_combo.currentTextChanged.connect(self._change_mode)
        controls.addWidget(self.mode_combo)

        self.toggle_button = QPushButton("Stop camera")
        self.toggle_button.clicked.connect(self._toggle)
        controls.addWidget(self.toggle_button)
        controls.addStretch(1)

        self.status = QLabel()
        controls.addWidget(self.status)
        layout.addLayout(controls)

        self.preview = QLabel("Starting camera…")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(640, 480)
        self.preview.setStyleSheet("background: #111; color: #ddd;")
        layout.addWidget(self.preview, 1)
        self.setCentralWidget(root)

    def _start(self) -> None:
        if self.capture is None:
            self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            self.preview.setText(f"Unable to open camera source {self.source!r}")
            self.status.setText("Camera unavailable")
            return
        self.frames = 0
        self.fps_started = time.perf_counter()
        self.timer.start(16)
        self.toggle_button.setText("Stop camera")

    def _stop(self) -> None:
        self.timer.stop()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.toggle_button.setText("Start camera")
        self.status.setText("Stopped")

    def _toggle(self) -> None:
        if self.timer.isActive():
            self._stop()
        else:
            self._start()

    def _change_mode(self, label: str) -> None:
        mode = _MODE_LABELS[label]
        self.tracker.set_mode(mode)
        self.frames = 0
        self.fps_started = time.perf_counter()
        self.status.setText(f"Mode: {label}")

    def _update_frame(self) -> None:
        if self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok:
            self.status.setText("Frame read failed")
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.tracker.process(rgb)
        self.tracker.draw(frame, result)

        self.frames += 1
        elapsed = max(time.perf_counter() - self.fps_started, 1e-6)
        detections = len(result.hands) + len(result.faces) + len(result.poses)
        self.status.setText(
            f"{self.tracker.mode.value.title()} · {detections} detected · "
            f"{self.frames / elapsed:.1f} FPS"
        )

        display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = display.shape
        image = QImage(
            display.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.preview.setPixmap(pixmap)

    def closeEvent(self, event) -> None:
        self._stop()
        self.tracker.close()
        event.accept()


def _parse_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="camera index or video path")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in TrackingMode],
        default=TrackingMode.HANDS.value,
    )
    args = parser.parse_args(argv)

    app = QApplication.instance() or QApplication(sys.argv)
    window = MediaPipeTrackerGUI(_parse_source(args.source), args.mode)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
