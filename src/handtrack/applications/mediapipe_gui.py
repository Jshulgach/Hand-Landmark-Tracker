"""Live preview with selectable MediaPipe or experimental WiLoR inference."""

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
    QSizePolicy,
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

    def __init__(self, source: int | str = 0, mode: str = "hands", *,
                 model: str = "mediapipe", wilor_root=None, wilor_device="auto",
                 wilor_fast=False) -> None:
        super().__init__()
        self.source = source
        if model not in ("mediapipe", "wilor"):
            raise ValueError("model must be mediapipe or wilor")
        if model == "wilor" and TrackingMode.parse(mode) is not TrackingMode.HANDS:
            raise ValueError("WiLoR supports hands only")
        self._model = model
        self._wilor_options = dict(wilor_root=wilor_root, wilor_device=wilor_device,
                                   wilor_fast=wilor_fast)
        self._restart_requested = False
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

        controls.addWidget(QLabel("Model"))
        self.model_combo = QComboBox()
        self.model_combo.addItem("MediaPipe", "mediapipe")
        self.model_combo.addItem("WiLoR", "wilor")
        self.model_combo.setToolTip("WiLoR integration is experimental; see the setup and validation guide.")
        self.model_combo.setCurrentIndex(1 if self._model == "wilor" else 0)
        self.model_combo.currentIndexChanged.connect(self._change_model)
        controls.addWidget(self.model_combo)
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
        self.mode_combo.setEnabled(self._model != "wilor")

        self.toggle_button = QPushButton("Stop camera")
        self.toggle_button.clicked.connect(self._toggle)
        controls.addWidget(self.toggle_button)
        controls.addStretch(1)

        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addLayout(controls)
        layout.addWidget(self.status)

        self.preview = QLabel("Starting camera…")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(240, 180)
        self.preview.setWordWrap(True)
        self.preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.preview.setStyleSheet("background: #111; color: #ddd;")
        layout.addWidget(self.preview, 1)
        self.setCentralWidget(root)

    def _start(self) -> None:
        if self.worker is not None:
            return
        self.preview.clear()
        self.preview.setText("Starting camera…")
        self.status.setText("Starting…")
        self.worker = PreviewWorker(self.source, self._mode, model=self._model, **self._wilor_options)
        self.worker.start()
        self.timer.start(16)
        self.toggle_button.setEnabled(True)
        self.toggle_button.setText("Stop camera")

    def _stop(self) -> None:
        if self.worker is not None:
            self.worker.request_stop()
            self.toggle_button.setEnabled(False)
            self.mode_combo.setEnabled(False)
            self.model_combo.setEnabled(False)
            self.preview.clear()
            self.preview.setText("Closing camera…" if self._closing else "Stopping camera…")
            self.status.setText("Stopping…")

    def _toggle(self) -> None:
        if self.worker is not None:
            self._stop()
        else:
            self._start()

    def _change_model(self, *_):
        self._model = self.model_combo.currentData()
        if self._model == "wilor":
            self._mode = TrackingMode.HANDS
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentText("Hands")
            self.mode_combo.blockSignals(False)
        self.mode_combo.setEnabled(self._model != "wilor")
        if self.worker is not None:
            self._restart_requested = True
            self._stop()

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
                f"{self._model} | {packet.mode.value.title()} · {packet.detections} detected · {packet.fps:.1f} FPS")
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
            self.mode_combo.setEnabled(self._model != "wilor")
            self.model_combo.setEnabled(True)
            self.preview.clear()
            message = error or ("Source ended or camera disconnected. Start to retry."
                                if state == "ended" else "Stopped")
            self.preview.setText(message)
            self.status.setText("WiLoR unavailable" if error and self._model == "wilor" else message)
            if self._closing:
                self.close()
            elif self._restart_requested:
                self._restart_requested = False
                self._start()

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
    parser.add_argument("--model", choices=("mediapipe", "wilor"), default="mediapipe")
    parser.add_argument("--wilor-root", help="official WiLoR source and asset directory")
    parser.add_argument("--wilor-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--wilor-fast", action="store_true", help="CUDA FP16 and depth pruning")
    args = parser.parse_args(argv)
    if args.model == "wilor" and args.mode != "hands":
        parser.error("WiLoR supports hands only; use --mode hands")

    app = QApplication.instance() or QApplication(sys.argv)
    window = MediaPipeTrackerGUI(
        _parse_source(args.source), args.mode, model=args.model,
        wilor_root=args.wilor_root, wilor_device=args.wilor_device, wilor_fast=args.wilor_fast)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
