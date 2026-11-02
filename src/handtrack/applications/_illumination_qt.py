"""Compact illumination panel shared by tracking and calibration."""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGroupBox, QGridLayout, QComboBox, QSpinBox, QPushButton, QCheckBox, QLabel
from ._illumination_controls import IlluminationControls


class IlluminationPanel(QGroupBox):
    def __init__(self, controls=None, keyboard_focus=True):
        super().__init__('IR illumination')
        self.controls = controls or IlluminationControls(None)
        grid = QGridLayout(self)
        grid.setContentsMargins(6,6,6,6)
        grid.setSpacing(4)
        self.target = QComboBox()
        self.target.setToolTip('Change all cameras or one camera')
        self.level = QSpinBox()
        self.level.setToolTip('SDK IR intensity; exposure is a separate setting')
        self.apply_button = QPushButton('Apply IR')
        self.high_power = QCheckBox('High power')
        self.high_power.setToolTip('Enable only when the camera and powered hub report support/availability')
        self.max_button = QPushButton('Max available')
        self.max_button.setToolTip('Maximum reported intensity; enable high power where supported and available')
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        if not keyboard_focus:
            for widget in (self.target,self.level,self.apply_button,self.high_power,self.max_button):
                widget.setFocusPolicy(Qt.NoFocus)
        grid.addWidget(self.target,0,0)
        grid.addWidget(self.level,0,1)
        grid.addWidget(self.apply_button,0,2)
        grid.addWidget(self.high_power,1,0,1,2)
        grid.addWidget(self.max_button,1,2)
        grid.addWidget(self.status,2,0,1,3)
        grid.setColumnStretch(0,1)
        self.target.currentIndexChanged.connect(self.refresh)
        self.apply_button.clicked.connect(lambda:self._apply(intensity=self.level.value()))
        self.max_button.clicked.connect(lambda:self._apply(maximize=True))
        self.high_power.stateChanged.connect(self._power_changed)
        self.bind_manager(self.controls.manager)

    def bind_manager(self, manager):
        self.controls.manager = manager
        self.controls.errors = []
        self.target.blockSignals(True)
        self.target.clear()
        self.target.addItem('All cameras',None)
        count = getattr(manager,'num_cameras',0) if manager is not None else 0
        for index in range(count):
            self.target.addItem(f'Camera {index}',index)
        self.target.blockSignals(False)
        self.refresh()

    def indices(self):
        count = self.target.count()-1
        selected = self.target.currentData()
        return list(range(count)) if selected is None else [selected]

    def set_target(self, selected):
        index = 0 if selected is None else selected+1
        if self.target.currentIndex()!=index:
            self.target.blockSignals(True)
            self.target.setCurrentIndex(index)
            self.target.blockSignals(False)
            self.refresh()

    def refresh(self,*args):
        indices = self.indices()
        states = [self.controls.read(index) for index in indices]
        levels = [state for state in states if state.intensity_supported]
        low = max((s.minimum for s in levels),default=0)
        high = min((s.maximum for s in levels),default=0)
        editable = bool(levels) and low<=high
        self.level.blockSignals(True)
        self.level.setRange(low,high if high>=low else low)
        if editable:
            self.level.setValue(max(low,min(high,levels[0].intensity)))
        self.level.blockSignals(False)
        self.level.setEnabled(editable)
        self.apply_button.setEnabled(editable)
        self.target.setEnabled(bool(states))
        power = [s.high_power for s in states if s.high_power_supported]
        mixed = len(set(power))>1
        self.high_power.blockSignals(True)
        self.high_power.setTristate(mixed)
        self.high_power.setCheckState(Qt.PartiallyChecked if mixed else Qt.Checked if power and all(power) else Qt.Unchecked)
        self.high_power.setEnabled(any(s.high_power_supported and (s.high_power_available or s.high_power) for s in states))
        self.high_power.blockSignals(False)
        self.max_button.setEnabled(any(s.intensity_supported or s.high_power_supported and s.high_power_available for s in states))
        lines = [f'C{index}: {state.describe()}' for index,state in zip(indices,states)]
        lines.extend(self.controls.errors)
        if states and all(not s.intensity_supported and not s.high_power_supported and s.message==states[0].message for s in states):
            lines = [states[0].message, *self.controls.errors]
        self.status.setText('\n'.join(lines) if lines else 'Start cameras to read illumination capabilities.')
        self.status.setStyleSheet('color: #ffcc66;' if self.controls.errors or any(s.message for s in states) else 'color: #aaa;')

    def _power_changed(self,state):
        if state != Qt.PartiallyChecked:
            self._apply(high_power=state==Qt.Checked)

    def _apply(self,**settings):
        self.controls.apply(self.indices(),**settings)
        self.refresh()
