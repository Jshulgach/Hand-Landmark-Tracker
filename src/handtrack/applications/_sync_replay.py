"""Separate review window: one Vicon timeline, with no held missing images."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from mavis_track._sync_analysis import load_session, load_vicon, video_path, file_hash


class ReplayData:
    def __init__(self, session, vicon, *, mapping=None, projection=None):
        self.directory = Path(session).resolve()
        self.manifest, self.frames = load_session(self.directory)
        self.vicon = load_vicon(vicon)
        mapping = Path(mapping) if mapping else self.directory/'alignment'/'frame_mapping.csv'
        report = json.loads((mapping.parent/'alignment.json').read_text(encoding='utf-8'))
        if (report.get('frames_sha256') != file_hash(self.directory/'frames.jsonl') or
                report.get('manifest_sha256') != file_hash(self.directory/'manifest.json') or
                report.get('mapping_sha256') != file_hash(mapping)):
            raise ValueError('Recording or mapping has changed since the alignment report was generated')
        if report.get('vicon', {}).get('sha256') != self.vicon['sha256']:
            raise ValueError('C3D differs from the file used to produce this alignment')
        self.mapping = {camera: {} for camera in self.frames}
        with mapping.open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                if not row['vicon_frame']:
                    continue
                camera, frame, video = int(row['camera_index']), int(row['vicon_frame']), int(row['video_frame_index'])
                if camera not in self.frames or not 0 <= video < len(self.frames[camera]):
                    raise ValueError('Mapping references an unrecorded image')
                if not self.vicon['first_frame'] <= frame <= self.vicon['last_frame']:
                    raise ValueError('Mapping references a frame outside the C3D export')
                if frame in self.mapping[camera]:
                    raise ValueError('Ambiguous video-to-Vicon frame mapping')
                self.mapping[camera][frame] = (video, row['confidence'])
        if not any(self.mapping.values()):
            raise ValueError('No images are mapped to Vicon; create an alignment report with explicit anchors first')
        self.projections = self._load_projection(projection) if projection else {}
        self.readers = {}
        try:
            for camera in self.frames:
                reader = cv2.VideoCapture(str(video_path(self.directory, self.manifest, camera)))
                self.readers[camera] = reader
                if not reader.isOpened():
                    raise ValueError(f'Cannot open camera {camera} video')
        except Exception:
            self.close()
            raise

    def _load_projection(self, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('schema') != 'mavis-vicon-projection-v1' or data.get('coordinate_units') != 'meters':
            raise ValueError('Projection requires mavis-vicon-projection-v1 and meter coordinates')
        result = {}
        for key, entry in data['cameras'].items():
            index = int(key)
            if index not in self.frames:
                raise ValueError('Projection references an unrecorded camera')
            matrix = np.asarray(entry['camera_matrix'], float)
            transform = np.asarray(entry['vicon_to_camera'], float)
            distortion = np.asarray(entry['dist_coeffs'], float).reshape(-1)
            if (matrix.shape != (3, 3) or transform.shape != (4, 4) or
                    not np.isfinite(matrix).all() or not np.isfinite(transform).all() or
                    not np.isfinite(distortion).all() or len(distortion) not in (4, 5, 8, 12, 14) or
                    matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.allclose(matrix[2], [0, 0, 1]) or
                    not np.allclose(transform[3], [0, 0, 0, 1]) or
                    not np.allclose(transform[:3, :3].T@transform[:3, :3], np.eye(3), atol=1e-5) or
                    not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-5)):
                raise ValueError('Invalid camera projection matrices')
            settings = next((s for s in self.manifest.get('camera_settings', []) if s.get('camera_index') == index), {})
            if settings.get('serial') is None or entry.get('serial') != settings['serial']:
                raise ValueError('Projection must match a recorded physical camera serial number')
            if list(entry['image_size']) != [settings.get('width'), settings.get('height')]:
                raise ValueError('Projection resolution does not match recorded camera')
            result[index] = (matrix, distortion, transform, entry['image_size'])
        return result

    def points(self, frame):
        return self.vicon['points'][frame-self.vicon['first_frame']]

    def image(self, camera, frame):
        match = self.mapping[camera].get(frame)
        if match is None:
            return None  # Never substitute the previous or nearest image.
        video_index, confidence = match
        reader = self.readers[camera]
        if int(reader.get(cv2.CAP_PROP_POS_FRAMES)) != video_index:
            reader.set(cv2.CAP_PROP_POS_FRAMES, video_index)
        ok, image = reader.read()
        if not ok:
            return None
        if camera in self.projections:
            matrix, distortion, transform, size = self.projections[camera]
            if (image.shape[1], image.shape[0]) != tuple(size):
                raise ValueError('Decoded video resolution does not match projection')
            points = self.points(frame)
            camera_points = points@transform[:3, :3].T + transform[:3, 3]
            valid = np.isfinite(camera_points).all(axis=1) & (camera_points[:, 2] > 0)
            if valid.any():
                pixels, _ = cv2.projectPoints(camera_points[valid], np.zeros(3), np.zeros(3), matrix, distortion)
                for x, y in pixels.reshape(-1, 2):
                    if np.isfinite([x, y]).all() and 0 <= x < size[0] and 0 <= y < size[1]:
                        cv2.circle(image, (round(x), round(y)), 4, (0, 255, 255), -1)
        cv2.putText(image, confidence, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 220, 255), 1)
        return image

    def close(self):
        for reader in self.readers.values():
            reader.release()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def show_replay(data):
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout, QLabel, QSlider, QPushButton
    app = QApplication.instance() or QApplication([])

    class Window(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(f"Synchronized review — {data.manifest.get('trial_name', 'trial')}")
            self.resize(1100, 800)
            widget = QWidget()
            layout = QVBoxLayout(widget)
            self.status = QLabel('Alignment estimates; missing images remain blank. Vicon marker view: XY, meters.')
            layout.addWidget(self.status)
            grid = QGridLayout()
            self.tiles = {}
            for i, camera in enumerate(data.frames):
                label = QLabel()
                label.setAlignment(Qt.AlignCenter)
                label.setMinimumSize(200, 150)
                grid.addWidget(label, i//2, i%2)
                self.tiles[camera] = label
            self.markers = QLabel()
            self.markers.setAlignment(Qt.AlignCenter)
            grid.addWidget(self.markers, (len(data.frames)+1)//2, 0, 1, 2)
            layout.addLayout(grid, 1)
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setRange(data.vicon['first_frame'], data.vicon['last_frame'])
            self.slider.valueChanged.connect(self.seek)
            layout.addWidget(self.slider)
            self.play = QPushButton('Play')
            self.play.clicked.connect(self.toggle)
            layout.addWidget(self.play)
            self.setCentralWidget(widget)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.tick)
            finite = data.vicon['points'][:, :, :2].reshape(-1, 2)
            finite = finite[np.isfinite(finite).all(axis=1)]
            self.origin = finite.min(axis=0) if len(finite) else np.zeros(2)
            extent = finite.max(axis=0)-self.origin if len(finite) else np.ones(2)
            self.scale = 220/max(float(extent.max()), .001)
            self.render()

        def set_image(self, label, image):
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            converted = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
            label.setPixmap(QPixmap.fromImage(converted).scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        def render(self):
            frame = self.slider.value()
            self.status.setText(f'Vicon frame {frame} | {(frame-data.vicon["first_frame"])/data.vicon["rate"]:.3f} s | estimated mapping; XY marker view')
            for camera, label in self.tiles.items():
                image = data.image(camera, frame)
                if image is None:
                    label.clear()
                    label.setText(f'Camera {camera}: no mapped/decodable image')
                else:
                    self.set_image(label, image)
            canvas = np.zeros((270, 700, 3), np.uint8)
            for name, point in zip(data.vicon['markers'], data.points(frame)):
                if np.isfinite(point).all():
                    x, y = ((point[:2]-self.origin)*self.scale + [20, 20]).astype(int)
                    cv2.circle(canvas, (x, 250-y), 3, (80, 230, 80), -1)
                    cv2.putText(canvas, name, (x+4, 250-y), cv2.FONT_HERSHEY_SIMPLEX, .3, (200, 200, 200), 1)
            self.set_image(self.markers, canvas)

        def seek(self):
            if self.timer.isActive():
                self.started, self.start_frame = time.perf_counter(), self.slider.value()
            self.render()

        def toggle(self):
            if self.timer.isActive():
                self.timer.stop()
                self.play.setText('Play')
            else:
                if self.slider.value() == self.slider.maximum():
                    self.slider.setValue(self.slider.minimum())
                self.started, self.start_frame = time.perf_counter(), self.slider.value()
                self.timer.start(16)
                self.play.setText('Pause')

        def tick(self):
            frame = min(self.slider.maximum(), self.start_frame + int((time.perf_counter()-self.started)*data.vicon['rate']))
            if frame != self.slider.value():
                self.slider.blockSignals(True)
                self.slider.setValue(frame)
                self.slider.blockSignals(False)
                self.render()
            if frame == self.slider.maximum():
                self.toggle()

        def closeEvent(self, event):
            self.timer.stop()
            event.accept()

    window = Window()
    window.show()
    return app.exec_()
