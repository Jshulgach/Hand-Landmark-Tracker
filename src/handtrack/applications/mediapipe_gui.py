"""Unified single-camera GUI for MediaPipe hands, face, pose, and holistic."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence

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

from handtrack.tracker import TrackingMode

from ._preview_worker import PreviewWorker

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
        self.worker: PreviewWorker | None = None
        self._mode = TrackingMode.parse(mode)
        self._closing = False
        self.frames = 0
        self.fps_started = time.perf_counter()

        self.setWindowTitle("MAVIS · Tracking Studio")
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
            if value is self._mode
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
        if self.worker is not None:
            return
        self.preview.clear()
        self.preview.setText("Starting camera…")
        self.status.setText("Starting…")
        self.worker = PreviewWorker(self.source, self._mode)
        self.worker.start()
        self.timer.start(16)
        self.toggle_button.setEnabled(True)
        self.toggle_button.setText("Stop camera")

    def _stop(self) -> None:
        if self.worker is not None:
            self.worker.request_stop()
            self.toggle_button.setEnabled(False)
            self.mode_combo.setEnabled(False)
            self.preview.clear()
            self.preview.setText("Closing camera…" if self._closing else "Stopping camera…")
            self.status.setText("Stopping…")

    def _toggle(self) -> None:
        if self.worker is not None:
            self._stop()
        else:
            self._start()

    def _change_mode(self, label: str) -> None:
        self._mode = _MODE_LABELS[label]
        if self.worker is not None:
            self.worker.request_mode(self._mode)
            self.preview.clear()
            self.preview.setText(f"Switching to {label}…")
            self.status.setText(f"Switching to {label}…")

    def _update_frame(self) -> None:
        worker = self.worker
        if worker is None:
            return
        packet, mode, state, error, published_at = worker.poll()
        if packet is not None:
            display = packet.rgb
            height, width, _ = display.shape
            image = QImage(display.data, width, height, display.strides[0], QImage.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(image).scaled(
                self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview.setPixmap(pixmap)
            self.status.setText(error or
                f"{packet.mode.value.title()} · {packet.detections} detected · {packet.fps:.1f} FPS")
        elif state == "running" and published_at is not None and time.perf_counter() - published_at > 1:
            self.preview.clear()
            self.preview.setText("Waiting for new frames…")
            self.status.setText(error or "Waiting for camera/model…")
        if error and state == "running":
            self._mode = mode
            label = next(label for label, value in _MODE_LABELS.items() if value is mode)
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentText(label)
            self.mode_combo.blockSignals(False)
            self.status.setText(error)
        if not worker.is_alive():
            worker.join(timeout=0)
            self.worker = None
            self.timer.stop()
            self.toggle_button.setText("Start camera")
            self.toggle_button.setEnabled(True)
            self.mode_combo.setEnabled(True)
            self.preview.clear()
            message = error or ("Source ended or camera disconnected. Start to retry."
                                if state == "ended" else "Stopped")
            self.preview.setText(message)
            self.status.setText(message)
            if self._closing:
                self.close()

    def closeEvent(self, event) -> None:
        if self.worker is not None:
            self._closing = True
            self._stop()
            event.ignore()
        else:
            self.timer.stop()
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
