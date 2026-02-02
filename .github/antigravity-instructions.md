# AI Agent Instructions for Hand-Landmark-Tracker

## Project Overview
This project is a Python-based library and application suite for real-time hand and face landmark tracking. It leverages **Google MediaPipe** for inference and **OpenCV** for image processing. It includes a core library (`handtrack`) and interactive GUI examples (`examples/`).

## Architecture & Core Components

### 1. Core Library (`src/handtrack/`)
- **`tracker/`**: Contains the main tracking logic.
    - **`HandTracker`** (`_hand_tracker.py`): The primary class for detecting hands. It wraps MediaPipe, handles video sources (webcam/file), and manages `Kalman3D` filters.
    - **`Kalman3D`** (`processing/_kalman_filter.py`): Used to smooth landmark positions and reduce jitter.
- **`processing/`**:
    - **`_joint_angles.py`**: Computes biomechanically relevant angles from raw landmarks.
- **`ml/`**: Machine learning components for gesture recognition (EMG integration logic exists in `pipeline.py`).

### 2. Examples & Applications (`examples/`)
The `examples` directory enables immediate usage:
- **`01_basic_tracking/`**: Contains the most varied use-cases.
    - **`handtrack_gui.py`**: A PyQt5-based GUI for monocular tracking.
    - **`unity_hand_tracking/stereo/`**: Tools for **stereo camera setup**, including calibration (`calibration.py`) and stereo tracking (`stereo_handtrack_gui.py`).

### 3. Data Flow
1. **Input**: Webcam feed (`cv2.VideoCapture`) or Video File.
2. **Inference**: MediaPipe Hands detects 21 3D landmarks per hand.
3. **Filtering**: (Optional) Kalman filters smooth the raw coordinate data.
4. **Calculations**: Joint angles are computed from smoothed landmarks.
5. **Output**:
    - **Visual**: OpenCV overlay or PyQt5 GUI.
    - **Network**: UDP broadcast of JSON data (Ports 5005 for landmarks, 5010 for angles).
    - **File**: CSV/NPZ recording.

## Critical Developer Workflows

### 1. Basic Monocular Tracking
To run the main GUI with configurable options:
```bash
python examples/01_basic_tracking/handtrack_gui.py
```
*Key features: toggle Kalman filter, record data, enable UDP.*

### 2. Stereo Camera Setup (Advanced)
Stereo tracking requires calibration:
1.  **Test Cameras**: `python examples/01_basic_tracking/unity_hand_tracking/stereo/test_cameras.py`
2.  **Calibration**: `python examples/01_basic_tracking/unity_hand_tracking/stereo/calibration.py` (Requires checkerboard).
3.  **Run Stereo Tracker**: `python examples/01_basic_tracking/unity_hand_tracking/stereo/stereo_handtrack_gui.py`

### 3. UDP Broadcasting
The system is designed to stream data to external apps (e.g., Unity, Robot OS).
- **Config**: IP/Port settable in GUIs or `config.py`.
- **Format**: JSON strings.
- **Troubleshooting**: If UDP fails, check firewalls and ensure the target application is listening on the correct port (default 5005/5010).

## Dependencies & Environment
- **Python**: >= 3.10
- **Package Manager**: `pip` (standard `pyproject.toml` setup).
- **Key Libraries**:
    - `mediapipe`: Core ML inference.
    - `opencv-python`: Image manipulation.
    - `PyQt5`: GUI components.
    - `numpy/scipy`: Math and filtering.
    - `pyqtgraph`: Real-time plotting (used in some advanced examples).

## Project Conventions
- **Code Style**: Standard Python PEP8.
- **Imports**: Absolute imports from `src` (e.g., `from handtrack.tracker import HandTracker`).
- **Configuration**: `pyproject.toml` defines build system and dependencies.
- **Root Directory**: Scripts often assume execution from their specific directory or require `PYTHONPATH` adjustments if run from root without installation.
