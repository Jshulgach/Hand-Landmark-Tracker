# Hand Landmark Tracker Examples

This directory contains organized examples demonstrating various capabilities of the Hand Landmark Tracker package.

## Directory Structure

```
examples/
├── 01_basic_tracking/          # Simple real-time tracking demos
├── 02_data_extraction/         # Extract and save landmark data
├── 03_streaming/               # LSL streaming and real-time data sharing
├── 04_advanced_applications/   # Robot control and complex applications
├── Joint_Kinematics_from_EMG/  # EMG-based joint angle prediction
└── Smoothing_Motion_with_RNN/  # Neural network smoothing techniques
```

---

## 📁 01_basic_tracking

**Simple examples to get started with hand tracking**

### `handtrack_webcam.py`
Basic webcam hand tracking with live visualization.
```bash
python 01_basic_tracking/handtrack_webcam.py
```

### `demo_gui.py`
Interactive GUI for passive circular marker tracking with adjustable parameters.
```bash
python 01_basic_tracking/demo_gui.py
```

### `passive_marker_track.py`
Track passive circular markers in images or video streams.
```bash
python 01_basic_tracking/passive_marker_track.py
```

**Use Cases:**
- Testing camera setup
- Real-time hand gesture recognition
- Quick prototyping

---

## 📁 02_data_extraction

**Extract and save landmark data from videos for offline analysis**

### `extract_hand_landmarks.py`
Extract 21 hand landmarks from video files with optional Kalman filtering.
```bash
python 02_data_extraction/extract_hand_landmarks.py --video_path path/to/video.mp4 --visualize --save_video
```

**Output:** `.npz` file containing:
- `landmarks`: (N, 21, 3) array of 3D positions
- `labels`: Landmark names
- `sampling_rate`: Video FPS
- `time_vector`: Timestamps

### `extract_face_landmarks.py`
Extract facial landmarks using MediaPipe Face Mesh.
```bash
python 02_data_extraction/extract_face_landmarks.py
```

### `run_landmark_selector.py`
Interactive tool to select and annotate specific landmarks.
```bash
python 02_data_extraction/run_landmark_selector.py
```

**Use Cases:**
- Creating training datasets
- Offline motion analysis
- Gait/gesture studies
- Synchronizing with other sensors

---

## 📁 03_streaming

**Real-time data streaming via Lab Streaming Layer (LSL)**

### `stream_landmarks_lsl.py` ⭐
Stream hand landmarks to LSL for real-time integration with other applications.

```bash
# Basic streaming from webcam
python 03_streaming/stream_landmarks_lsl.py

# Advanced options
python 03_streaming/stream_landmarks_lsl.py \
    --source 0 \
    --max_hands 2 \
    --stream_name HandLandmarks \
    --fps 30 \
    --no_kalman
```

**LSL Stream Details:**
- **Channels:** 63 (21 landmarks × 3 coords)
- **Type:** Mocap
- **Format:** float32
- **Rate:** 30 Hz (configurable)

**Compatible Tools:**
- `nml-wtf-exo-logger` - Record LSL streams to CSV
- `nml-wtf-exo-viewer` - Visualize landmark trajectories
- LabRecorder - General LSL recording
- MATLAB, Unity, Python LSL clients

### `lsl_emg_predict_angles.py`
Predict joint angles from EMG signals streamed via LSL.
```bash
python 03_streaming/lsl_emg_predict_angles.py
```

**Use Cases:**
- Multi-modal data collection
- Real-time robot control
- VR/AR hand tracking
- Synchronized physiological recordings

---

## 📁 04_advanced_applications

**Complex applications: robot control, virtual hands, and interactive demos**

### `Miniarm/`
Control a Mini-Arm robot gripper using hand tracking as a virtual spacemouse.

```bash
cd 04_advanced_applications/Miniarm
python miniarm.py --port COM5
python virtualspacemouse.py
```

**Features:**
- 3D position control from hand centroid
- Pinch gesture for gripper open/close
- Real-time trajectory visualization

### `InMoov-Arm-Demo/`
Demo for controlling InMoov robotic arm with hand tracking.

```bash
cd 04_advanced_applications/InMoov-Arm-Demo
python hand-arm-tracker.py
```

**Features:**
- Full arm inverse kinematics
- Gesture-based mode switching
- Serial communication with Pico controller

### `virtual_hand/`
Render virtual 3D hand models driven by tracked landmarks.

```bash
cd 04_advanced_applications/virtual_hand
python hand_gui.py
```

**Features:**
- OpenGL 3D hand rendering
- MANO hand model support
- Real-time pose estimation

---

## 📁 Joint_Kinematics_from_EMG

**Train regression models to predict continuous finger joint angles from EMG signals**

This pipeline demonstrates **EMG-to-angles regression** using PyTorch for prosthetic control and motion analysis.

### What It Does:
- Synchronizes video landmarks with EMG recordings
- Extracts EMG features (RMS, MAV, WL, ZC, SSC)
- Trains PyTorch neural networks (regression, not classification)
- Predicts 5 continuous joint angles (thumb, index, middle, ring, pinky)
- Real-time LSL streaming prediction

### Quick Start:
```bash
# 1. Calculate sync offset
python calculate_sync_offset.py --root_dir /data --label MySession

# 2. Create training dataset
python create_feature_dataset.py --root_dir /data --label MySession

# 3. Train regression model
python train_model.py --root_dir /data --label MySession

# 4. Evaluate predictions
python example_predict_angles.py --root_dir /data --label MySession

# 5. Real-time prediction
python lsl_emg_predict_angles.py
```

### Key Differences:
- **This example:** EMG → Continuous angles (regression, PyTorch)
- **python-intan gesture_classifier:** EMG → Discrete gestures (classification, sklearn)

See [`README.md`](Joint_Kinematics_from_EMG/README.md) and [`PACKAGE_COMPARISON.md`](Joint_Kinematics_from_EMG/PACKAGE_COMPARISON.md) for full documentation.

---

## 📁 Smoothing_Motion_with_RNN

**Use recurrent neural networks to smooth noisy landmark trajectories**

### Workflow:

1. **Extract Features** (`1_feature_extraction.py`)
   ```bash
   python 1_feature_extraction.py --video_path path/to/video.mp4
   ```

2. **Train GRU Model** (`2_train_gru.py`)
   ```bash
   python 2_train_gru.py --data_path features.npz
   ```

3. **Predict** (`3_predict.py`)
   ```bash
   python 3_predict.py --model_path model.h5
   ```

4. **Evaluate** (`4_evaluate.py`)
   ```bash
   python 4_evaluate.py --predictions results.npz
   ```

**Comparison:**
- `kalman_filter.py` - Traditional Kalman smoothing (baseline)
- GRU outperforms Kalman for complex hand motions

---

## Quick Start Guide

### 1. Install Dependencies
```bash
cd Hand-Landmark-Tracker
pip install -e .
pip install pylsl  # For LSL streaming examples
```

### 2. Test Basic Tracking
```bash
python examples/01_basic_tracking/handtrack_webcam.py
```

### 3. Stream to LSL and Log
**Terminal 1:**
```bash
python examples/03_streaming/stream_landmarks_lsl.py
```

**Terminal 2:**
```bash
nml-wtf-exo-logger
# Click "Refresh Streams" → Select "HandLandmarks" → Start Logging
```

### 4. Extract Data from Video
```bash
python examples/02_data_extraction/extract_hand_landmarks.py \
    --video_path my_video.mp4 \
    --visualize \
    --save_video
```

---

## Requirements by Category

**Basic Tracking:**
- opencv-python
- mediapipe
- numpy

**Data Extraction:**
- scipy (for Kalman filtering)
- pandas
- tqdm

**Streaming:**
- pylsl
- pyqtgraph (for viewer)

**Advanced Applications:**
- pyserial (robot communication)
- PyOpenGL (3D rendering)
- trimesh (MANO hand model)

**Machine Learning:**
- tensorflow / pytorch
- scikit-learn

---

## Tips & Best Practices

### Camera Setup
- Use good lighting (avoid shadows)
- Position camera 1-2 feet from hands
- Avoid busy backgrounds
- 30+ FPS camera recommended

### Kalman Filtering
- Enable for smoother tracking: `--apply_kalman`
- Tune process_noise and measurement_noise for your application
- Disable for raw MediaPipe output: `--no_kalman`

### LSL Streaming
- Use wired connections for low latency
- Check stream with `pylsl.resolve_streams()` before connecting
- Name streams descriptively when running multiple: `--stream_name LeftHand`

### Performance
- Lower resolution for higher FPS: `img_size=(640, 480)`
- Reduce `max_hands` if only tracking one hand
- Disable visualization for headless operation: `--no_visualize`

---

## Troubleshooting

**Camera not detected:**
```bash
# Try different camera indices
python handtrack_webcam.py --source 0  # or 1, 2, etc.
```

**LSL stream not found:**
```bash
# Check available streams
python -c "from pylsl import resolve_streams; print(resolve_streams())"
```

**Import errors:**
```bash
# Reinstall package
cd Hand-Landmark-Tracker
pip install -e .
```

**Slow performance:**
- Reduce camera resolution
- Close other applications
- Use GPU-accelerated OpenCV build

---

## Contributing

When adding new examples:
1. Place in appropriate category directory
2. Add command-line argument parsing
3. Include docstring with usage example
4. Update this README
5. Test with default parameters

---

## Citation

If you use these examples in your research, please cite:
```
@software{hand_landmark_tracker,
  author = {Shulgach, Jonathan},
  title = {Hand Landmark Tracker},
  year = {2025},
  url = {https://github.com/Jshulgach/Hand-Landmark-Tracker}
}
```
