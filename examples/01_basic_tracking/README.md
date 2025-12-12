# Basic Hand Tracking Examples

Simple examples to get started with real-time hand tracking using MediaPipe.

## Examples

### 1. `handtrack_webcam.py`
Basic webcam hand tracking with live visualization of 21 hand landmarks.

**Usage:**
```bash
python handtrack_webcam.py
```

**Features:**
- Real-time hand detection from webcam
- 21 landmark visualization
- Kalman filtering for smooth tracking
- Press ESC to quit

---

### 2. `demo_gui.py`
Interactive PyQt5 GUI for passive circular marker tracking with adjustable detection parameters.

**Usage:**
```bash
python demo_gui.py
```

**Features:**
- Load images or videos
- Adjust detection sensitivity
- Tune minimum distance between markers
- Real-time parameter updates

**Use Case:** Tracking passive reflective markers for motion capture.

---

### 3. `passive_marker_track.py`
Track passive circular markers in video streams or images.

**Usage:**
```bash
python passive_marker_track.py
```

**Features:**
- Hough circle detection
- Configurable marker size and sensitivity
- Works with reflective markers
- Outputs marker positions

**Use Case:** Low-cost motion capture with passive markers.

---

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install opencv-python mediapipe numpy
   ```

2. **Run basic tracking:**
   ```bash
   python handtrack_webcam.py
   ```

3. **Check camera index if webcam not detected:**
   ```bash
   # Edit source parameter in script or try:
   python handtrack_webcam.py --source 1
   ```

## Tips

- **Good Lighting:** Ensure even lighting without harsh shadows
- **Camera Distance:** Keep hands 1-2 feet from camera
- **Plain Background:** Avoid cluttered backgrounds for better detection
- **FPS:** Use 30+ FPS camera for smooth tracking
