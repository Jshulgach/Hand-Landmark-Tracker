<div align="center">
  <img src="assets/hand-demo.gif" width="70%">
</div>

<!-- ![](assets/hand-demo.gif) -->

# Hand Landmark Tracker &nbsp;[![](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
Hand Landmark Tracker bring a new perspective to human-computer interaction using hands. The hand is the most versatile and intuitive controller someone can use, so it makes sense to see if there is a way to design an interface that takes advantage of the hands without requiring them to touch anything.
This code uses the amazing features of Google's machine learning suite [MediaPipe](https://developers.google.com/mediapipe), a media-based ML package for classification and recognition with neural networks.

<!-- This program uses openCV and mediapipe to acquire hand landmarks and post/gesture tracking commands to stream to a [Robot Web Server]().  -->

<!-- <figure> -->
<!--  <img src="./assets/hand-demo.gif" alt="Hand" width="500" height="400"><br> -->
<!--  <figcaption>Landmark tracking. Multi-hand classification and landmark identification.</figcaption> -->
<!-- </figure> -->

## Installation

1. Create a virtual environment using [Anaconda](https://www.anaconda.com/products/distribution) or Python's virtualenv.
   - Using Anaconda:
      ~~~
      conda create -n hand-tracker python=3.10
      conda activate hand-tracker
      ~~~
   - Using Python's virtualenv:
     ~~~
     python3 -m venv hand-tracker
     source hand-tracker/bin/activate # Linux/macOS
     call hand-tracker/Scripts/activate # Windows
     ~~~
2. Clone the repository and navigate to the project directory.
   ~~~
   git clone https://github.com/Jshulgach/Hand-Landmark-Tracker/unity_tree.git
   cd Hand-Landmark-Tracker
   ~~~
3. Install the package (recommended for production-style usage).
   ~~~
   pip install -e .
   ~~~

  Optional (OptiTrack-focused extras):
  ~~~
  pip install -e .[optitrack]
  ~~~


## Getting Started

### Flagship Demo (OptiTrack Multi-Camera GUI)

Run the full OptiTrack GUI using the package entrypoint:

```bash
handtrack-optitrack-gui
```

Related operational commands:

```bash
handtrack-doctor --optitrack
handtrack-optitrack-board
handtrack-optitrack-calibrate
handtrack-optitrack-cameras
handtrack-optitrack-test-sender
```

### Production Architecture

- `handtrack`: reusable core APIs (tracking primitives, calibration, processing, IO/broadcast)
- `unity_hand_tracking.optitrack_cam_py`: flagship OptiTrack application layer and camera backend integration
- `examples/`: scenario scripts, legacy demos, and utility tools

This separation keeps demo velocity high while preserving a stable API surface for downstream tools.

### Legacy Mono/Stereo Demos

Legacy script-style demos were moved out of `src/` and are now located in:

```bash
examples/01_basic_tracking/unity_hand_tracking/mono/
examples/01_basic_tracking/unity_hand_tracking/stereo/
```

These are kept for reference and backward-compatible experimentation, while active development focuses on the OptiTrack package entrypoints.

#### UDP Data Format

##### Landmarks (Port 5005)
```json
{
  "frame": 123,
  "num_hands": 1,
  "timestamp": 1234567890.123,
  "hands": [
    {
      "hand_index": 0,
      "landmarks": [[x1, y1, z1], [x2, y2, z2], ...]
    }
  ]
}
```

21 landmarks per hand in millimeters (world coordinates).

##### Joint Angles (Port 5010)
```json
{
  "frame": 123,
  "timestamp": 1234567890.123,
  "hands": [
    {
      "hand_index": 0,
      "angles": {
        "index_mcp": 12.34,
        "index_pip": 45.67,
        "index_dip": 23.89,
        "middle_mcp": 10.12,
        ...
      }
    }
  ]
}
```
14 angles per hand in degrees:
- Fingers: `{finger}_mcp`, `{finger}_pip`, `{finger}_dip` (index, middle, ring, pinky)
- Thumb: `thumb_cmc_mcp`, `thumb_ip`


**Features:**
- Real-time tracking from webcam or video file
- Kalman filter smoothing for stability
- Record landmarks to CSV/NPZ files
- **UDP broadcasting** for real-time streaming to other apps
- Dark theme UI with intuitive controls

<img src="docs/source/_static/stereo_hand_track.gif" alt="Stereo" width="500">

**Figure:** Example of a stereo hand tracking with unity. 

## Acknowledgement

The project was inspired by xenon-19's [Gesture Controlled Virtual Mouse](https://github.com/xenon-19/Gesture-Controlled-Virtual-Mouse) project.
 Thanks Google for the amazing [MediaPipe](https://developers.google.com/mediapipe) library.


