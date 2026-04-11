# Advanced Applications

Complex examples demonstrating robot control, 3D rendering, and interactive applications using hand tracking.

## Projects

### `stream_finger_angles_udp.py` ⭐
Drive the printed hand through `embedded_control_station` using live webcam hand tracking.

**Features:**
- Live MediaPipe hand tracking inside a desktop GUI with an embedded camera view
- MCP finger-angle extraction using the package kinematics helpers
- Safe UDP broadcast to `embedded_control_station` using the arm JSON protocol
- Configurable open/closed calibration and command smoothing
- Right-side control panel for camera enable or disable, UDP routing, and mapping controls
- Visual overlay with commanded finger positions

**Usage:**
```bash
python stream_finger_angles_udp.py --host 127.0.0.1 --port 9105 --stream-name finger-angles
```

The script now opens a polished control console inspired by `embedded_control_station`, with the live camera preview on the left and camera or UDP controls on the right.

**Recommended Workflow:**
1. Start `embedded_control_station` and connect the Pico.
2. Route the `finger-angles` stream in the Arm Console.
3. Run this bridge script.
4. Tune `--thumb-open-angle`, `--thumb-closed-angle`, `--finger-open-angle`, and `--finger-closed-angle` until the printed hand tracks comfortably.

**Notes:**
- The bridge sends `wrist`, `thumb`, `index`, `middle`, `ring`, and `pinky` targets over UDP.
- Default command range is `0..140` for safer first motion. Increase `--closed-command` if the hand needs more travel.

---

### 📁 Miniarm
Control a Mini-Arm robot gripper using hand tracking as a virtual spacemouse.

**Features:**
- 3D position control from hand centroid
- Pinch gesture detection for gripper control
- Real-time trajectory visualization
- Serial communication with robot

**Usage:**
```bash
cd Miniarm
python miniarm.py --port COM5
```

**Hardware Required:**
- [Mini-Arm robot](https://github.com/Jshulgach/Mini-Arm)
- USB serial connection

---

### 📁 InMoov-Arm-Demo
Demo for controlling InMoov robotic arm with hand tracking.

**Features:**
- Full arm inverse kinematics
- Gesture-based mode switching
- Hand pose mirroring to robot
- Pico microcontroller integration

**Usage:**
```bash
cd InMoov-Arm-Demo
python hand-arm-tracker.py
```

**Files:**
- `hand-arm-tracker.py` - Main tracking and control
- `inmoov.py` - InMoov robot class
- `pico/code.py` - Raspberry Pi Pico firmware

---

### 📁 virtual_hand
Render virtual 3D hand models driven by tracked landmarks.

**Features:**
- OpenGL 3D rendering
- MANO hand model support
- Real-time pose updates
- Texture mapping

**Usage:**
```bash
cd virtual_hand
python hand_gui.py
```

**Requirements:**
- PyOpenGL
- trimesh
- MANO hand model files

**Use Cases:**
- VR hand avatars
- Animation
- Hand pose visualization
- Research demonstrations

---

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install pyserial PyOpenGL trimesh
   ```

2. **Test basic robot control:**
   ```bash
   cd Miniarm
   python miniarm.py --port COM5
   ```

3. **Check serial ports:**
   ```bash
   python -c "import serial.tools.list_ports; print([p.device for p in serial.tools.list_ports.comports()])"
   ```

## Tips

- **Serial Connection:** Ensure correct COM port and baud rate
- **Robot Safety:** Test with small motions first
- **Gesture Calibration:** Adjust thresholds for your hand size
- **Coordinate Mapping:** Tune scale factors for workspace size
