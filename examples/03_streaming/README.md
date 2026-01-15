# Streaming Examples

Real-time landmark streaming for integration with other applications.

## Quick Start: UDP Broadcasting (Recommended)

The easiest way to stream landmarks is using the GUI applications with built-in UDP broadcasting:

```bash
# Hand tracking with UDP streaming
python ../01_basic_tracking/handtrack_gui.py

# Face tracking with UDP streaming
python ../01_basic_tracking/facetrack_gui.py
```

**In the GUI:**
1. Check "Enable UDP Broadcasting"
2. Set IP address (use `255.255.255.255` for local broadcast)
3. Set port (default: 5005)
4. Start tracking

**Receive Data (Python):**
```python
import socket
import json

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('', 5005))

while True:
    data, addr = sock.recvfrom(65535)
    landmarks = json.loads(data.decode('utf-8'))
    print(f"Frame: {landmarks['frame']}, Hands: {landmarks.get('num_hands', landmarks.get('num_faces'))}")
```

**Advantages:**
- No external dependencies (pure socket)
- Works across networks
- Low latency
- Simple JSON format
- Cross-platform

---

## LSL Streaming (Advanced)
Stream hand landmarks to LSL in real-time from webcam.

**Basic Usage:**
```bash
# Stream from default webcam with Kalman filtering
python stream_landmarks_lsl.py
```

**Advanced Options:**
```bash
# Track two hands simultaneously
python stream_landmarks_lsl.py --max_hands 2

# Use secondary camera
python stream_landmarks_lsl.py --source 1

# Custom stream name (useful for multiple streams)
python stream_landmarks_lsl.py --stream_name LeftHand

# Disable Kalman smoothing for raw landmarks
python stream_landmarks_lsl.py --no_kalman

# Run headless (no visualization window)
python stream_landmarks_lsl.py --no_visualize

# Change target FPS
python stream_landmarks_lsl.py --fps 60
```

**LSL Stream Specifications:**
- **Name:** `HandLandmarks` (customizable)
- **Type:** `Mocap` (Motion Capture)
- **Channels:** 63 (21 landmarks × 3 coordinates)
- **Format:** `float32`
- **Sample Rate:** 30 Hz (configurable)
- **Channel Labels:** `WRIST_x`, `WRIST_y`, `WRIST_z`, `THUMB_CMC_x`, ...

**Metadata:**
- Landmark names for each channel
- Units (normalized for x/y, depth for z)
- MediaPipe model version
- Stream source ID

---

### 2. `lsl_emg_predict_angles.py`
Predict joint angles from EMG signals streamed via LSL using a trained model.

**Usage:**
```bash
# Basic usage with trained model
python lsl_emg_predict_angles.py --model path/to/model.pkl

# Connect to a specific named EMG stream
python lsl_emg_predict_angles.py --emg_stream MyEMGStream --model ./model.pkl

# Adjust prediction window size (samples to buffer)
python lsl_emg_predict_angles.py --model ./model.pkl --window_size 100
```

**Requirements:**
- `pip install pylsl joblib`
- A trained model file (see `examples/Joint_Kinematics_from_EMG/` for training)
- An active LSL stream of EMG data

**Workflow:**
1. Train a model using `examples/Joint_Kinematics_from_EMG/`
2. Start your EMG streaming device
3. Run this script to get real-time joint angle predictions

---

## Typical Workflow: Stream and Log

### Terminal 1 - Start Streaming
```bash
cd examples/03_streaming
python stream_landmarks_lsl.py --stream_name MyHandTracking
```

You should see:
```
============================================================
STREAMING HAND LANDMARKS TO LSL
============================================================
Stream Name: MyHandTracking
Kalman Filtering: Enabled
Max Hands: 1
Visualization: Enabled

Press ESC to stop streaming...
============================================================

[LSL] Created outlet: 'MyHandTracking' with 63 channels @ 30 Hz
[Tracker] Initializing HandTracker (source=0, max_hands=1)
```

### Terminal 2 - Log the Stream
```bash
nml-wtf-exo-logger
```

1. Click **"Refresh Streams"**
2. Select **"MyHandTracking"** from dropdown
3. Choose log directory (default: `landmarks/`)
4. Click **"Start Logging"**

Logs will be saved as CSV files with timestamps.

---

## Compatible Applications

### Python
```python
from pylsl import StreamInlet, resolve_stream

# Find and connect to stream
streams = resolve_stream('name', 'HandLandmarks')
inlet = StreamInlet(streams[0])

# Pull samples
while True:
    sample, timestamp = inlet.pull_sample()
    # sample is a list of 63 floats
    # Reshape to (21, 3) for landmarks
    landmarks = np.array(sample).reshape(21, 3)
```

### MATLAB
```matlab
% Add LSL to path
addpath('path/to/liblsl-Matlab')

% Resolve stream
lib = lsl_loadlib();
result = lsl_resolve_byprop(lib, 'name', 'HandLandmarks');
inlet = lsl_inlet(result{1});

% Pull samples
while true
    [sample, timestamp] = inlet.pull_sample();
    landmarks = reshape(sample, [3, 21])';  % 21x3 matrix
end
```

### Unity (C#)
```csharp
using LSL;

// Resolve stream
var results = LSL.resolve_stream("name", "HandLandmarks");
var inlet = new StreamInlet(results[0]);

// Pull samples
float[] sample = new float[63];
while (true) {
    inlet.pull_sample(sample);
    // Parse into Vector3[] for landmarks
}
```

### Other Tools
- **LabRecorder** - General-purpose LSL recording
- **nml-wtf-exo-viewer** - Real-time landmark trajectory visualization
- **OpenViBE** - Brain-computer interface platform
- **Bonsai** - Visual programming for neuroscience

---

## Coordinate System

LSL stream uses MediaPipe's coordinate system:
- **Channels 0-2:** WRIST (x, y, z)
- **Channels 3-5:** THUMB_CMC (x, y, z)
- **Channels 6-8:** THUMB_MCP (x, y, z)
- ...and so on for all 21 landmarks

**x, y:** Normalized [0.0, 1.0] (image coordinates)
**z:** Depth relative to wrist (negative = closer to camera)

---

## Performance Tips

### Low Latency
- Use wired Ethernet (not WiFi) for LSL
- Reduce `--fps` if stream drops samples
- Disable visualization: `--no_visualize`
- Use single hand tracking: `--max_hands 1`

### High Accuracy
- Enable Kalman filtering (default)
- Good lighting conditions
- Stable camera mount
- Plain background

### Multiple Streams
Stream from multiple cameras simultaneously:
```bash
# Terminal 1
python stream_landmarks_lsl.py --source 0 --stream_name Camera1

# Terminal 2
python stream_landmarks_lsl.py --source 1 --stream_name Camera2
```

---

## Troubleshooting

**Stream not found:**
```bash
# Check available LSL streams
python -c "from pylsl import resolve_streams; print(resolve_streams())"
```

**Firewall blocking LSL:**
- Allow Python through Windows Firewall
- LSL uses UDP multicast (port 16571)

**Dropped samples:**
- Lower FPS: `--fps 15`
- Reduce max hands: `--max_hands 1`
- Close other applications

**Import error for pylsl:**
```bash
pip install pylsl
```

---

## Use Cases

- **Robot Control:** Real-time hand tracking for teleoperation
- **VR/AR:** Low-latency hand pose input
- **Research:** Synchronized multi-modal data collection
- **Rehabilitation:** Motion tracking for therapy
- **Gaming:** Gesture-based control
- **Data Collection:** Large-scale hand motion datasets
