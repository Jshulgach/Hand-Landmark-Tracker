"""Minimal DPI-aware calibration preview; exposure is controlled from the keyboard."""
import sys
import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QSpinBox
from ._display import CameraGrid, ImagePreview, configure_high_dpi, fit_window_to_screen, apply_text_font


class CalibrationPreview(QMainWindow):
    def __init__(self, controls, title):
        configure_high_dpi()
        app = QApplication.instance() or QApplication(sys.argv[:1])
        super().__init__()
        self.app = app
        self.controls = controls
        self._initial_geometry_applied = False
        self._closing = False
        self._base_font = QFont(self.app.font())
        self._base_font.setPointSizeF(max(11.,self._base_font.pointSizeF()))
        self.setWindowTitle(title)
        self.setFont(self._base_font)
        self.resize(1280,800)
        root = QWidget()
        root.setStyleSheet('QWidget { background: #181818; color: #eee; } QSpinBox { background: #292929; }')
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(8,8,8,8)
        layout.setSpacing(6)
        header = QHBoxLayout()
        self.status = QLabel('Starting cameras...')
        self.status.setWordWrap(True)
        font = QFont(self._base_font);font.setBold(True);self.status.setFont(font)
        header.addWidget(self.status,1)
        header.addWidget(QLabel('Text'))
        self.text_size = QSpinBox()
        self.text_size.setRange(100,180)
        self.text_size.setSuffix('%')
        self.text_size.setSingleStep(10)
        self.text_size.setValue(100)
        self.text_size.setFocusPolicy(Qt.NoFocus)
        self.text_size.valueChanged.connect(self._change_text_size)
        header.addWidget(self.text_size)
        layout.addLayout(header)
        self.exposure_status = QLabel()
        self.exposure_status.setWordWrap(True)
        layout.addWidget(self.exposure_status)
        self.camera_grid = CameraGrid()
        grid = self.camera_grid.reset(len(controls.exposures))
        self.labels = []
        for index in range(len(controls.exposures)):
            label = ImagePreview()
            label.setStyleSheet('background: #111; border: 1px solid #333;')
            grid.addWidget(label,index//self.camera_grid.columns,index%self.camera_grid.columns)
            self.labels.append(label)
        layout.addWidget(self.camera_grid,1)
        self.help = QLabel('\n'.join(controls.instructions))
        self.help.setWordWrap(True)
        self.help.setStyleSheet('color: #ccc;')
        layout.addWidget(self.help)
        apply_text_font(root, self._base_font)
        self.setFocusPolicy(Qt.StrongFocus)
        self.show()
        self.setFocus()

    def showEvent(self,event):
        super().showEvent(event)
        if not self._initial_geometry_applied:
            self._initial_geometry_applied=True
            fit_window_to_screen(self)
            if self.windowHandle() is not None:
                self.windowHandle().screenChanged.connect(self._screen_changed)

    def _screen_changed(self,*args):
        self.camera_grid.refresh_geometry()
        for label in self.labels:
            label.refresh_image()

    def _change_text_size(self,value):
        font = QFont(self._base_font)
        font.setPointSizeF(self._base_font.pointSizeF()*value/100)
        self.setFont(font)
        apply_text_font(self.centralWidget(),font)
        bold = QFont(font);bold.setBold(True);self.status.setFont(bold)

    def keyPressEvent(self,event):
        keys = {Qt.Key_Up:'up',Qt.Key_Down:'down',Qt.Key_A:'a',
                Qt.Key_Space:'space',Qt.Key_Q:'q',Qt.Key_Escape:'escape'}
        key = keys.get(event.key(),event.text().lower())
        if key and self.controls.handle_key(key,bool(event.modifiers() & Qt.ShiftModifier)):
            self._update_exposure_text()
            event.accept()
        else:
            super().keyPressEvent(event)

    def _update_exposure_text(self):
        self.exposure_status.setText(self.controls.last_error or self.controls.description())
        self.exposure_status.setStyleSheet('color: #ffcc66;' if self.controls.last_error else 'color: #ddd;')

    def show_frames(self,frames,status):
        self.status.setText(status)
        self._update_exposure_text()
        for index,(frame,label) in enumerate(zip(frames,self.labels)):
            rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            height,width = rgb.shape[:2]
            if index == 0:
                self.camera_grid.set_image_aspect(width/height)
            image = QImage(rgb.data,width,height,rgb.strides[0],QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(image))
        self.app.processEvents()

    def closeEvent(self,event):
        if not self._closing:
            self.controls.cancelled=True
        event.accept()

    def shutdown(self):
        self._closing=True
        self.close()
        self.app.processEvents()
