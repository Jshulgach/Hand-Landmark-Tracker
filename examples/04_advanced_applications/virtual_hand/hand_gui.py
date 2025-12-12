import sys
import torch
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSlider
from PyQt5.QtCore import Qt
from manopth.manolayer import ManoLayer
from manopth import demo

model_path = r"C:\Users\HP\Documents\Github\Hand-Landmark-Tracker\manopth\mano\models"

# FINGER_INDICES = {
#     "thumb": 3,
#     "index": 12,
#     "middle": 21,
#     "ring": 30,
#     "pinky": 39
# }

FINGER_INDICES = {
    "thumb": 0,
    "index": 9,
    "middle": 18,
    "ring": 27,
    "pinky": 36
}

class HandPoseGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MANO Finger Control")
        self.layout = QVBoxLayout()

        self.mano_layer = ManoLayer(mano_root=model_path, use_pca=False, flat_hand_mean=True)
        #self.mano_layer = ManoLayer(mano_root=model_path, use_pca=False)
        self.pose = torch.zeros(1, 39)
        self.shape = torch.zeros(1, 10)

        self.sliders = {}

        for finger in FINGER_INDICES:
            label = QLabel(f"{finger.title()} Flex (degrees)")
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(90)
            slider.setValue(0)
            slider.valueChanged.connect(self.update_pose)
            self.layout.addWidget(label)
            self.layout.addWidget(slider)
            self.sliders[finger] = slider

        self.setLayout(self.layout)
        self.update_pose()  # Initial display

    def update_pose(self):
        self.pose.zero_()
        for finger, start_idx in FINGER_INDICES.items():
            angle_deg = self.sliders[finger].value()
            angle_rad = -angle_deg * 3.1416 / 180.0
            for j in range(3):  # 3 joints
                joint_z_idx = start_idx + j * 3 + 2
                if joint_z_idx < 39:
                    self.pose[0, joint_z_idx] = angle_rad

        full_pose = torch.cat([torch.zeros(1, 3), self.pose], dim=1)

        verts, joints = self.mano_layer(full_pose, self.shape)
        demo.display_hand({'verts': verts, 'joints': joints}, mano_faces=self.mano_layer.th_faces)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandPoseGUI()
    window.show()
    sys.exit(app.exec_())
