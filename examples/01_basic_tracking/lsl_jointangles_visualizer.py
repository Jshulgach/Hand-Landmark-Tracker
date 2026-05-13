"""
Standalone LSL JointAngles visualizer.

Brings up a lightweight GUI that subscribes to an LSL stream carrying hand joint
angles and renders a 3D hand model in the same style as the OptiTrack MOCAP GUI
(Top / Front / Side / Isometric views with finger-colored skeleton).
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QImage, QPalette, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    from pylsl import StreamInlet, resolve_byprop

    HAS_LSL = True
except Exception:
    StreamInlet = None
    resolve_byprop = None
    HAS_LSL = False


ANGLE_NAMES = [
    "index_mcp",
    "index_pip",
    "index_dip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "thumb_cmc_mcp",
    "thumb_ip",
]


def _safe_angle_deg(v, default=0.0):
    try:
        x = float(v)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return float(default)


def _angles_to_landmarks_3d(angles14: np.ndarray) -> np.ndarray:
    """
    Convert 14 joint angles into a synthetic 21-point hand model.

    The kinematic model is intentionally simple but stable and interpretable:
    - 4 fingers use MCP/PIP/DIP chain angles in a local Y-Z flexion plane.
    - Thumb uses two provided angles (CMC/MCP combined and IP) mapped across
      thumb joints with mild spread into X/Z for visualization.
    """
    a = np.asarray(angles14, dtype=np.float32).reshape(-1)
    if a.size < 14:
        out = np.zeros(21, dtype=np.float32)
        out[: a.size] = a
        a = out
    else:
        a = a[:14]

    lmk = np.zeros((21, 3), dtype=np.float32)

    # Wrist / palm anchor
    lmk[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # MCP anchor points across palm
    lmk[5] = np.array([-30.0, -5.0, 0.0], dtype=np.float32)  # index
    lmk[9] = np.array([-10.0, -2.0, 0.0], dtype=np.float32)  # middle
    lmk[13] = np.array([12.0, -2.0, 0.0], dtype=np.float32)  # ring
    lmk[17] = np.array([30.0, -5.0, 0.0], dtype=np.float32)  # pinky
    lmk[1] = np.array([-42.0, -18.0, -8.0], dtype=np.float32)  # thumb CMC anchor

    # Finger segment lengths (proximal, middle, distal)
    finger_lengths = {
        "index": (33.0, 23.0, 16.0),
        "middle": (36.0, 26.0, 18.0),
        "ring": (34.0, 24.0, 17.0),
        "pinky": (29.0, 19.0, 14.0),
    }

    # Small per-finger splay into Z so the 3D view has depth
    finger_splay = {
        "index": -0.15,
        "middle": -0.05,
        "ring": 0.08,
        "pinky": 0.18,
    }

    def add_finger_chain(base_idx, out_idxs, mcp_deg, pip_deg, dip_deg, lengths, splay):
        base = lmk[base_idx].copy()
        t1 = np.radians(_safe_angle_deg(mcp_deg))
        t2 = t1 + np.radians(_safe_angle_deg(pip_deg))
        t3 = t2 + np.radians(_safe_angle_deg(dip_deg))
        angles = [t1, t2, t3]
        cur = base.copy()
        for j, (theta, seg_len) in enumerate(zip(angles, lengths)):
            dy = -seg_len * np.cos(theta)
            dz = seg_len * np.sin(theta)
            dx = splay * seg_len
            cur = cur + np.array([dx, dy, dz], dtype=np.float32)
            lmk[out_idxs[j]] = cur

    add_finger_chain(
        5, [6, 7, 8], a[0], a[1], a[2], finger_lengths["index"], finger_splay["index"]
    )
    add_finger_chain(
        9,
        [10, 11, 12],
        a[3],
        a[4],
        a[5],
        finger_lengths["middle"],
        finger_splay["middle"],
    )
    add_finger_chain(
        13, [14, 15, 16], a[6], a[7], a[8], finger_lengths["ring"], finger_splay["ring"]
    )
    add_finger_chain(
        17,
        [18, 19, 20],
        a[9],
        a[10],
        a[11],
        finger_lengths["pinky"],
        finger_splay["pinky"],
    )

    # Thumb (2 DOF provided, spread across 3 chain bends)
    thumb_cmc = np.radians(_safe_angle_deg(a[12]))
    thumb_ip = np.radians(_safe_angle_deg(a[13]))
    t1 = 0.55 * thumb_cmc
    t2 = 0.45 * thumb_cmc
    t3 = thumb_ip
    thumb_lengths = (22.0, 19.0, 15.0)
    cur = lmk[1].copy()
    # Thumb base direction points outward in -X and downward in Y
    thumb_base_dir = np.array([-0.75, -0.65, -0.10], dtype=np.float32)
    thumb_base_dir = thumb_base_dir / (np.linalg.norm(thumb_base_dir) + 1e-8)
    for idx_out, theta, seg_len in zip(
        [2, 3, 4], [t1, t1 + t2, t1 + t2 + t3], thumb_lengths
    ):
        # Bend around local plane to keep motion plausible in 3D
        dy = -seg_len * np.cos(theta)
        dz = seg_len * np.sin(theta)
        step = np.array(
            [
                thumb_base_dir[0] * seg_len,
                0.8 * dy,
                0.7 * dz + thumb_base_dir[2] * seg_len,
            ],
            dtype=np.float32,
        )
        cur = cur + step
        lmk[idx_out] = cur

    return lmk


class LSLJointAnglesVisualizer(QMainWindow):
    def __init__(self, stream_name: str, hand_index: int):
        super().__init__()
        self.stream_name = stream_name
        self.hand_index = int(max(0, hand_index))

        self.inlet = None
        self.last_resolve_t = 0.0
        self.resolve_interval_s = 1.0

        self.latest_angles = np.zeros(14, dtype=np.float32)
        self.latest_ts = 0.0
        self.sample_count = 0

        self.show_top = False
        self.show_front = False
        self.show_side = False
        self.show_iso = True

        self.setWindowTitle("LSL JointAngles 3D Visualizer")
        self.setMinimumSize(1100, 760)

        self._init_ui()
        self._setup_timer()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        self.vis_label = QLabel()
        self.vis_label.setMinimumSize(700, 700)
        self.vis_label.setStyleSheet("background-color: #000; border: 1px solid #555;")
        self.vis_label.setAlignment(Qt.AlignCenter)
        root.addWidget(self.vis_label, stretch=3)

        panel = QFrame()
        panel.setStyleSheet(
            """
            QFrame { background-color: #2d2d2d; border-radius: 6px; }
            QLabel { color: #ffffff; font-size: 11px; }
            QCheckBox { color: #ffffff; font-size: 11px; }
            QComboBox, QPushButton {
                background-color: #3d3d3d; color: #ffffff; border: 1px solid #555;
                padding: 4px; border-radius: 3px;
            }
            QPushButton:hover { background-color: #4a4a4a; }
            """
        )
        panel_layout = QVBoxLayout(panel)

        title = QLabel("JointAngles LSL")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        panel_layout.addWidget(title)

        self.status_label = QLabel("Status: waiting")
        panel_layout.addWidget(self.status_label)

        self.stream_label = QLabel(f"Stream: {self.stream_name}")
        panel_layout.addWidget(self.stream_label)

        self.rate_label = QLabel("Samples: 0")
        panel_layout.addWidget(self.rate_label)

        self.last_ts_label = QLabel("Last sample age: N/A")
        panel_layout.addWidget(self.last_ts_label)

        hand_row = QHBoxLayout()
        hand_row.addWidget(QLabel("Hand:"))
        self.hand_combo = QComboBox()
        self.hand_combo.addItem("Left / first", 0)
        self.hand_combo.addItem("Right / second", 1)
        self.hand_combo.setCurrentIndex(0 if self.hand_index == 0 else 1)
        self.hand_combo.currentIndexChanged.connect(self._on_hand_changed)
        hand_row.addWidget(self.hand_combo)
        panel_layout.addLayout(hand_row)

        self.show_top_cb = QCheckBox("Top")
        self.show_front_cb = QCheckBox("Front")
        self.show_side_cb = QCheckBox("Side")
        self.show_iso_cb = QCheckBox("Isometric")

        self.show_top_cb.setChecked(self.show_top)
        self.show_front_cb.setChecked(self.show_front)
        self.show_side_cb.setChecked(self.show_side)
        self.show_iso_cb.setChecked(self.show_iso)

        self.show_top_cb.toggled.connect(lambda v: setattr(self, "show_top", bool(v)))
        self.show_front_cb.toggled.connect(
            lambda v: setattr(self, "show_front", bool(v))
        )
        self.show_side_cb.toggled.connect(lambda v: setattr(self, "show_side", bool(v)))
        self.show_iso_cb.toggled.connect(lambda v: setattr(self, "show_iso", bool(v)))

        panel_layout.addWidget(self.show_top_cb)
        panel_layout.addWidget(self.show_front_cb)
        panel_layout.addWidget(self.show_side_cb)
        panel_layout.addWidget(self.show_iso_cb)

        refresh_btn = QPushButton("Reconnect stream")
        refresh_btn.clicked.connect(self._force_reconnect)
        panel_layout.addWidget(refresh_btn)

        self.angles_label = QLabel("Angles: --")
        self.angles_label.setWordWrap(True)
        panel_layout.addWidget(self.angles_label)

        panel_layout.addStretch(1)
        root.addWidget(panel, stretch=1)

    def _setup_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(30)

    def _on_hand_changed(self, _):
        self.hand_index = int(self.hand_combo.currentData())

    def _force_reconnect(self):
        self.inlet = None
        self.last_resolve_t = 0.0

    def _resolve_inlet(self):
        if not HAS_LSL or resolve_byprop is None:
            self.status_label.setText("Status: pylsl not available")
            return
        now = time.time()
        if (now - self.last_resolve_t) < self.resolve_interval_s:
            return
        self.last_resolve_t = now
        try:
            streams = resolve_byprop("name", self.stream_name, timeout=0.05)
            if streams:
                self.inlet = StreamInlet(streams[0])
                self.status_label.setText("Status: connected")
            else:
                self.status_label.setText("Status: waiting for stream")
        except Exception as exc:
            self.status_label.setText(f"Status: resolve error ({exc})")

    def _sample_to_angles14(self, sample: list[float]) -> np.ndarray:
        vec = np.asarray(sample, dtype=np.float32).reshape(-1)
        if vec.size == 14:
            return vec
        if vec.size >= 28:
            start = self.hand_index * 14
            end = start + 14
            if vec.size >= end:
                return vec[start:end]
        if vec.size > 14:
            # Fallback: use first 14 and keep GUI running
            return vec[:14]
        out = np.zeros(14, dtype=np.float32)
        out[: vec.size] = vec
        return out

    def _poll_lsl(self):
        if self.inlet is None:
            self._resolve_inlet()
            return
        try:
            samples, timestamps = self.inlet.pull_chunk(timeout=0.0, max_samples=16)
        except Exception:
            self.inlet = None
            return
        if not samples:
            return

        latest = samples[-1]
        self.latest_angles = self._sample_to_angles14(latest)
        if timestamps and len(timestamps) > 0:
            self.latest_ts = float(timestamps[-1])
        else:
            self.latest_ts = time.time()
        self.sample_count += len(samples)

    def _render_3d_view(self):
        hand_landmarks = _angles_to_landmarks_3d(self.latest_angles)

        w, h = 600, 600
        canvas = np.full((h, w, 3), 40, dtype=np.uint8)

        grid_spacing = 50
        for i in range(0, w, grid_spacing):
            cv2.line(canvas, (i, 0), (i, h), (60, 60, 60), 1)
        for i in range(0, h, grid_spacing):
            cv2.line(canvas, (0, i), (w, i), (60, 60, 60), 1)

        half_w, half_h = w // 2, h // 2
        qw, qh = half_w, half_h

        active_views = []
        if self.show_top:
            active_views.append("top")
        if self.show_front:
            active_views.append("front")
        if self.show_side:
            active_views.append("side")
        if self.show_iso:
            active_views.append("iso")

        if not active_views:
            cv2.putText(
                canvas,
                "Enable a 3D view",
                (w // 2 - 100, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (120, 120, 120),
                1,
            )
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            qimg = QImage(
                rgb.data,
                rgb.shape[1],
                rgb.shape[0],
                rgb.shape[1] * 3,
                QImage.Format_RGB888,
            )
            self.vis_label.setPixmap(QPixmap.fromImage(qimg))
            return

        n_views = len(active_views)
        if n_views == 1:
            view_rects = {active_views[0]: (0, 0, w, h)}
        else:
            slots = [(0, 0), (half_w, 0), (0, half_h), (half_w, half_h)]
            view_rects = {}
            for i, vname in enumerate(active_views):
                ox, oy = slots[i]
                view_rects[vname] = (ox, oy, qw, qh)
            cv2.line(canvas, (half_w, 0), (half_w, h), (100, 100, 100), 2)
            if n_views > 2:
                cv2.line(canvas, (0, half_h), (w, half_h), (100, 100, 100), 2)

        view_labels = {
            "top": "Top (XZ)",
            "front": "Front (XY)",
            "side": "Side (ZY)",
            "iso": "Iso",
        }
        for vname, (ox, oy, _, _) in view_rects.items():
            cv2.putText(
                canvas,
                view_labels[vname],
                (ox + 10, oy + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

        valid_points = [p for p in hand_landmarks if np.linalg.norm(p) > 1e-3]
        if not valid_points:
            return
        centroid = np.mean(valid_points, axis=0)
        max_extent = max(np.max(np.abs(p - centroid)) for p in valid_points)
        max_extent = max(max_extent, 1.0)
        ref_size = list(view_rects.values())[0][2]
        scale = (ref_size * 1.755) / max_extent

        iso_angle = np.radians(30)
        cos_a, sin_a = np.cos(iso_angle), np.sin(iso_angle)

        THUMB_COLOR = (0, 200, 255)
        INDEX_COLOR = (0, 255, 100)
        MIDDLE_COLOR = (255, 200, 0)
        RING_COLOR = (255, 0, 150)
        PINKY_COLOR = (100, 100, 255)
        PALM_COLOR = (180, 180, 180)

        colored_connections = [
            (0, 1, THUMB_COLOR),
            (1, 2, THUMB_COLOR),
            (2, 3, THUMB_COLOR),
            (3, 4, THUMB_COLOR),
            (0, 5, INDEX_COLOR),
            (5, 6, INDEX_COLOR),
            (6, 7, INDEX_COLOR),
            (7, 8, INDEX_COLOR),
            (5, 9, PALM_COLOR),
            (9, 13, PALM_COLOR),
            (13, 17, PALM_COLOR),
            (0, 17, PALM_COLOR),
            (9, 10, MIDDLE_COLOR),
            (10, 11, MIDDLE_COLOR),
            (11, 12, MIDDLE_COLOR),
            (13, 14, RING_COLOR),
            (14, 15, RING_COLOR),
            (15, 16, RING_COLOR),
            (17, 18, PINKY_COLOR),
            (18, 19, PINKY_COLOR),
            (19, 20, PINKY_COLOR),
        ]

        def project_point(p, view_type, cx, cy):
            x, y, z = -p[0], -p[1], p[2]
            if view_type == "top":
                u = int(x * scale) + cx
                v = int(z * scale) + cy
            elif view_type == "front":
                u = int(x * scale) + cx
                v = int(-y * scale) + cy
            elif view_type == "side":
                u = int(z * scale) + cx
                v = int(-y * scale) + cy
            else:  # iso
                u = int((x * cos_a - z * cos_a) * scale) + cx
                v = int((-y + x * sin_a + z * sin_a) * scale * 0.8) + cy
            return u, v

        for view_type, (ox, oy, vw, vh) in view_rects.items():
            cx, cy = ox + vw // 2, oy + vh // 2
            for s, e, col in colored_connections:
                p1 = hand_landmarks[s]
                p2 = hand_landmarks[e]
                if np.linalg.norm(p1) < 1e-3 or np.linalg.norm(p2) < 1e-3:
                    continue
                c1 = p1 - centroid
                c2 = p2 - centroid
                u1, v1 = project_point(c1, view_type, cx, cy)
                u2, v2 = project_point(c2, view_type, cx, cy)
                cv2.line(canvas, (u1, v1), (u2, v2), col, 1)
                cv2.circle(canvas, (u1, v1), 2, col, -1)

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        qimg = QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888
        )
        self.vis_label.setPixmap(QPixmap.fromImage(qimg))

    def _tick(self):
        self._poll_lsl()
        self._render_3d_view()

        self.rate_label.setText(f"Samples: {self.sample_count}")
        if self.latest_ts > 0:
            age = time.time() - self.latest_ts
            self.last_ts_label.setText(f"Last sample age: {age:.2f}s")
            if age < 1.0:
                self.status_label.setText("Status: streaming")
            else:
                self.status_label.setText("Status: stale")
        else:
            self.last_ts_label.setText("Last sample age: N/A")

        txt = ", ".join(
            f"{name}={self.latest_angles[i]:.1f}" for i, name in enumerate(ANGLE_NAMES)
        )
        self.angles_label.setText(f"Angles: {txt}")

    def closeEvent(self, event):
        self.timer.stop()
        self.inlet = None
        event.accept()


def main():
    parser = argparse.ArgumentParser(
        description="Standalone LSL JointAngles visualizer"
    )
    parser.add_argument("--stream-name", default="JointAngles", help="LSL stream name")
    parser.add_argument(
        "--hand-index",
        type=int,
        default=0,
        help="0 for first/left hand, 1 for second/right hand when stream has two hands",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = app.palette()
    pal.setColor(QPalette.Window, QColor(30, 30, 30))
    pal.setColor(QPalette.WindowText, Qt.white)
    pal.setColor(QPalette.Base, QColor(45, 45, 45))
    pal.setColor(QPalette.Text, Qt.white)
    pal.setColor(QPalette.Button, QColor(45, 45, 45))
    pal.setColor(QPalette.ButtonText, Qt.white)
    pal.setColor(QPalette.Highlight, QColor(0, 120, 212))
    pal.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(pal)

    w = LSLJointAnglesVisualizer(
        stream_name=args.stream_name, hand_index=args.hand_index
    )
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
