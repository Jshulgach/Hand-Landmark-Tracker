# Advanced Applications

Complex examples demonstrating robot control, 3D rendering, and interactive applications using hand tracking.

## Projects

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
