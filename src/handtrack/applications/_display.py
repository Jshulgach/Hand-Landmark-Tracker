"""Qt display helpers shared by live tracking and calibration windows."""
from PyQt5.QtCore import Qt, QSize, QRect
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QGridLayout, QSizePolicy


def configure_high_dpi():
    """Configure Qt before QApplication, leaving an embedding host in control."""
    if QApplication.instance() is None:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)


def apply_text_font(root, font):
    """Apply point sizing explicitly through Qt stylesheet widget boundaries."""
    for widget in [root, *root.findChildren(QWidget)]:
        current = QFont(widget.font())
        current.setFamily(font.family())
        current.setPointSizeF(font.pointSizeF())
        widget.setFont(current)


def fit_window_to_screen(window, preferred=(1280, 800)):
    handle = window.windowHandle()
    screen = handle.screen() if handle is not None else QApplication.primaryScreen()
    if screen is None:
        return
    available = screen.availableGeometry().adjusted(12, 12, -12, -12)
    frame = window.frameGeometry()
    width = max(1, available.width() - max(0, frame.width() - window.width()))
    height = max(1, available.height() - max(0, frame.height() - window.height()))
    window.setMinimumSize(min(640, width), min(360, height))
    window.resize(min(preferred[0], width), min(preferred[1], height))
    frame = window.frameGeometry()
    frame.moveCenter(available.center())
    window.move(frame.topLeft())


class ImagePreview(QLabel):
    """Preserve full frames by default; optional fill mode crops only the display."""
    def __init__(self):
        super().__init__()
        self._source_pixmap = None
        self._fill = False
        self.setMinimumSize(1, 1)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setAlignment(Qt.AlignCenter)

    def setPixmap(self, pixmap):
        self._source_pixmap = pixmap
        self.refresh_image()

    def set_fill_mode(self, fill):
        self._fill = bool(fill)
        self.refresh_image()

    def clear(self):
        self._source_pixmap = None
        super().clear()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_image()

    def refresh_image(self):
        source = self._source_pixmap
        size = self.contentsRect().size()
        if source is None or source.isNull() or size.isEmpty():
            return
        ratio = self.devicePixelRatioF()
        pixels = QSize(max(1, round(size.width()*ratio)),
                       max(1, round(size.height()*ratio)))
        mode = Qt.KeepAspectRatioByExpanding if self._fill else Qt.KeepAspectRatio
        pixmap = source.scaled(pixels, mode, Qt.SmoothTransformation)
        if self._fill:
            pixmap = pixmap.copy((pixmap.width()-pixels.width())//2,
                                 (pixmap.height()-pixels.height())//2,
                                 pixels.width(), pixels.height())
        pixmap.setDevicePixelRatio(ratio)
        super().setPixmap(pixmap)


class CameraGrid(QWidget):
    """Pack camera tiles at image aspect ratio, keeping gaps outside the grid."""
    def __init__(self):
        super().__init__()
        self._content = None
        self._aspect = 1280/1024
        self._fill = False
        self.setMinimumSize(1, 1)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.reset(0)

    def reset(self, count):
        if self._content is not None:
            self._content.hide()
            self._content.deleteLater()
        self._count = count
        self.columns = 2 if count <= 4 else 3
        self._content = QWidget(self)
        self.grid = QGridLayout(self._content)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(3)
        self._content.show()
        self.refresh_geometry()
        return self.grid

    def set_image_aspect(self, aspect):
        if aspect > 0 and abs(self._aspect-aspect) > .001:
            self._aspect = aspect
            self.refresh_geometry()

    def set_fill_mode(self, fill):
        self._fill = bool(fill)
        self.refresh_geometry()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_geometry()

    def refresh_geometry(self):
        area = self.contentsRect()
        if not self._count:
            self._content.setGeometry(area)
            return
        gap = self.grid.spacing()
        def image_area(columns):
            rows = (self._count+columns-1)//columns
            width = min(max(1, (area.width()-gap*(columns-1))/columns),
                        max(1, (area.height()-gap*(rows-1))/rows)*self._aspect)
            return width*width/self._aspect*self._count
        columns = max(range(1,self._count+1),key=image_area)
        if columns != self.columns:
            widgets = []
            while self.grid.count():
                item = self.grid.takeAt(0)
                widgets.append(item.widget())
            self.columns = columns
            for index,widget in enumerate(widgets):
                self.grid.addWidget(widget,index//columns,index%columns)
        rows = (self._count+columns-1)//columns
        for index in range(self.grid.columnCount()):
            self.grid.setColumnStretch(index,1 if index < columns else 0)
        for index in range(self.grid.rowCount()):
            self.grid.setRowStretch(index,1 if index < rows else 0)
        if self._fill:
            self._content.setGeometry(area)
            return
        tile_width = min(max(1, (area.width()-gap*(columns-1))/columns),
                         max(1, (area.height()-gap*(rows-1))/rows)*self._aspect)
        width = max(1, round(tile_width*columns+gap*(columns-1)))
        height = max(1, round(tile_width/self._aspect*rows+gap*(rows-1)))
        rect = QRect(0, 0, width, height)
        rect.moveCenter(area.center())
        self._content.setGeometry(rect)
