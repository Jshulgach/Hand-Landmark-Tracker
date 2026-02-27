# Basic Hand & Face Tracking Examples

Simple examples to get started with real-time hand and face landmark tracking using MediaPipe and PyQt5.

## Installation

Before running any examples, install the required dependencies:

```bash
pip install -r ../../requirements.txt
```

Or manually install:
```bash
pip install opencv-python mediapipe PyQt5 numpy scipy
```

---

## Examples

### OptiTrack Multi-Camera (Flagship)

#### `optitrack_mocap_gui.py` ⭐ **Flagship Demo**
**Stereo OptiTrack hand tracking GUI with 3D triangulation and streaming**

```bash
python optitrack_mocap_gui.py
```

#### `optitrack_calibration.py`
**OptiTrack multi-camera ChArUco calibration launcher**

```bash
python optitrack_calibration.py
```

This wraps the package calibration implementation with the same dependency
preflight and MINGW-safe child-process environment handling as the GUI launcher.

**Why this is the showcase app:**
- Multi-camera 3D triangulation workflow with calibration support
- Real-time hand kinematics and smoothing in one GUI
- UDP + LSL streaming output for downstream integrations
- Designed for demos where robust 3D tracking matters most

Legacy Unity-oriented scripts were moved to:

- `examples/01_basic_tracking/unity_hand_tracking/mono/`
- `examples/01_basic_tracking/unity_hand_tracking/stereo/`
- `examples/01_basic_tracking/unity_hand_tracking/optitrack_tools/`

Use these as reference tools; package entrypoints remain the recommended runtime path.

---

### Hand Tracking

#### 1. `handtrack_webcam.py`
**Basic webcam hand tracking with live visualization**

```bash
python handtrack_webcam.py
```

**Features:**
- Real-time hand detection from webcam
- 21 landmark visualization per hand
- Supports up to 2 hands simultaneously
- Press ESC to quit

**Use Case:** Quick testing of hand tracking on your webcam.

---

#### 2. `handtrack_gui.py` ⭐ **Recommended**
**Interactive PyQt5 GUI for hand tracking with advanced controls**

```bash
python handtrack_gui.py
```

**Features:**
- 📹 Switch between webcam and video file input
- 🎯 Configurable camera ID and max number of hands
- 🔧 **Kalman filter smoothing** for noise reduction
- 💾 **Record and save landmark data** (CSV/NPZ format)
- 🌐 **UDP broadcasting** for real-time streaming to other applications
- Dark theme UI with intuitive control panel
- Scrollable control panel for all parameters
- Real-time frame counter and hand detection display

**Controls:**
- Adjust max hands to detect (1-4)
- Enable/disable Kalman smoothing
- Set save directory and record landmarks
- Configure UDP broadcast IP and port
- Monitor tracking status in real-time

**Example Workflow:**
```bash
# 1. Run the GUI
python handtrack_gui.py

# 2. In the GUI:
# - Select "Webcam" or "Video File" as source
# - Check "Enable Kalman Filter Smoothing" for smoother tracking
# - Set a save directory and enable recording
# - Enable UDP broadcasting to stream data to other apps
# - Click "Start Tracking" to begin

# 3. The landmarks will be:
# - Displayed in real-time on screen
# - Saved to CSV/NPZ when recording is enabled
# - Broadcast via UDP when enabled
```

---

### Face Tracking

#### 3. `facetrack_webcam.py`
**Basic webcam face tracking with live visualization**

```bash
python facetrack_webcam.py
```

**Features:**
- Real-time face mesh detection
- 468 facial landmarks including iris refinement
- Single face tracking
- Press ESC to quit

**Use Case:** Quick testing of face tracking on your webcam.

---

#### 4. `facetrack_gui.py` ⭐ **Recommended**
**Interactive PyQt5 GUI for face tracking with advanced controls**

```bash
python facetrack_gui.py
```

**Features:**
- 📹 Switch between webcam and video file input
- 🔧 **Kalman filter smoothing** for face landmark stabilization
- 💾 **Record and save face landmark data** (CSV/NPZ format)
- 🌐 **UDP broadcasting** for real-time face data streaming
- Dark theme UI with intuitive control panel
- Scrollable control panel for all parameters
- Real-time frame counter and face detection display

**Controls:**
- Enable/disable Kalman smoothing
- Set save directory and record landmarks
- Configure UDP broadcast IP and port
- Monitor tracking status in real-time

---

### Marker Tracking

#### 5. `passive_marker_track.py`
**Track passive circular markers in video streams or images**

```bash
python passive_marker_track.py
```

**Features:**
- Hough circle detection
- Configurable marker size and sensitivity
- Works with reflective markers
- Outputs marker positions

**Use Case:** Low-cost motion capture with passive circular markers.

---

#### 6. `demo_gui.py`
**Interactive GUI for passive circular marker tracking**

```bash
python demo_gui.py
```

**Features:**
- Load images or videos
- Adjust detection sensitivity
- Tune minimum distance between markers
- Real-time parameter updates

**Use Case:** Tuning marker detection parameters before running motion capture.

---

## UDP Broadcasting Guide

Both `handtrack_gui.py` and `facetrack_gui.py` support UDP broadcasting for real-time data streaming:

### How to Use:

1. **In the tracker GUI:**
   - Check "Enable UDP Broadcasting"
   - Set the **IP Address** (use `255.255.255.255` for local broadcast, or a subnet like `192.168.1.255`)
   - Set the **Port** (default: 5005, valid range: 1024-65535)
   - Start tracking

2. **Data Format (JSON over UDP):**

   **Hand Landmarks:**
   ```json
   {
     "frame": 120,
     "num_hands": 2,
     "timestamp": 1705324800.123,
     "hands": [
       {
         "hand_index": 0,
         "landmarks": [[0.5, 0.3, 0.1], [0.51, 0.31, 0.09], ...]
       }
     ]
   }
   ```

   **Face Landmarks:**
   ```json
   {
     "frame": 120,
     "num_faces": 1,
     "timestamp": 1705324800.123,
     "landmarks": [[0.5, 0.3, 0.1], [0.51, 0.31, 0.09], ...]
   }
   ```

3. **Receive Data (Python Example):**
   ```python
   import socket
   import json
   
   sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
   sock.bind(('', 5005))  # Listen on port 5005
   
   while True:
       data, addr = sock.recvfrom(65535)
       landmarks = json.loads(data.decode('utf-8'))
       print(f"Frame: {landmarks['frame']}, Hands: {landmarks['num_hands']}")
   ```

---

## Tips for Best Results

- **Good Lighting:** Ensure even lighting without harsh shadows
- **Camera Distance:** Keep hands/face 1-2 feet from camera
- **Plain Background:** Avoid cluttered backgrounds for better detection
- **Frame Rate:** Use 30+ FPS camera for smooth tracking
- **Kalman Filtering:** Enable for real-time applications to reduce jitter
- **UDP Network:** Ensure your network allows UDP broadcast traffic

---

## Troubleshooting

### Webcam not detected
- Try changing the camera ID (0, 1, 2, etc.)
- Check camera permissions in your OS settings
- Ensure the camera is not in use by another application

### Poor hand/face detection
- Improve lighting conditions
- Bring hand/face closer to camera (1-2 feet)
- Clean camera lens
- Try a different background

### UDP not working
- Check firewall settings (may block UDP)
- Use specific IP instead of broadcast if on restricted network
- Verify port is not in use: `netstat -an | grep 5005` (Windows) or `lsof -i :5005` (macOS/Linux)

---

## Performance Notes

- **CPU Usage:** ~10-20% on modern CPUs for single tracking
- **FPS:** Typically 20-30 FPS depending on hardware
- **GPU Support:** Use GPU-enabled MediaPipe for faster inference
- **UDP Broadcasting:** Minimal overhead (~1-2% CPU), depends on network bandwidth
