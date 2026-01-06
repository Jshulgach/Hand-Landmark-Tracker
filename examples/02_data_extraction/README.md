# Data Extraction Examples

Extract hand and face landmarks from videos and save for offline analysis.

## Examples

### 1. `extract_hand_landmarks.py`
Extract 21 hand landmarks from video files with optional Kalman filtering and visualization.

**Usage:**
```bash
# Basic extraction
python extract_hand_landmarks.py --video_path path/to/video.mp4

# With visualization and video output
python extract_hand_landmarks.py \
    --video_path path/to/video.mp4 \
    --visualize \
    --save_video

# Without Kalman filtering (raw landmarks)
python extract_hand_landmarks.py \
    --video_path path/to/video.mp4 \
    --no_kalman

# Custom output directory
python extract_hand_landmarks.py \
    --video_path path/to/video.mp4 \
    --output_dir ./my_landmarks
```

**Output:** `.npz` file containing:
- `landmarks`: (N, 21, 3) NumPy array of normalized 3D positions
- `labels`: Landmark names (WRIST, THUMB_TIP, INDEX_TIP, etc.)
- `sampling_rate`: Video FPS
- `time_vector`: Frame timestamps in seconds

**Features:**
- Kalman filtering for smooth trajectories (enabled by default)
- Progress bar with frame count
- Optional video output with landmarks overlaid
- Saves to `landmarks/` directory by default

---

### 2. `extract_face_landmarks.py`
Extract facial landmarks using MediaPipe Face Mesh (468 landmarks).

**Usage:**
```bash
# Basic extraction
python extract_face_landmarks.py --video_path path/to/video.mp4

# With visualization
python extract_face_landmarks.py \
    --video_path path/to/video.mp4 \
    --visualize

# Save labeled video
python extract_face_landmarks.py \
    --video_path path/to/video.mp4 \
    --visualize \
    --save_video

# Without Kalman filtering
python extract_face_landmarks.py \
    --video_path path/to/video.mp4 \
    --no_kalman

# Custom output directory
python extract_face_landmarks.py \
    --video_path path/to/video.mp4 \
    --output_dir ./my_landmarks
```

**Output:** `.npz` file containing:
- `landmarks`: (N, 468, 3) NumPy array of normalized 3D positions
- `num_landmarks`: Number of landmarks (468)
- `sampling_rate`: Video FPS
- `time_vector`: Frame timestamps in seconds

**Features:**
- Full face mesh (468 landmarks)
- Eyes, nose, mouth, face contour
- 3D depth estimation
- Kalman filtering for smooth trajectories

**Use Case:** Facial expression analysis, gaze tracking, emotion recognition.

---

### 3. `run_landmark_selector.py`
Interactive tool to select and annotate specific landmarks of interest.

**Usage:**
```bash
python run_landmark_selector.py
```

**Features:**
- Click to select landmarks
- Label and save selections
- Export custom landmark subsets

**Use Case:** Creating custom landmark configurations for specific applications.

---

## Loading Extracted Data

```python
import numpy as np

# Load saved landmarks
data = np.load('landmarks/video_name_landmarks.npz')

landmarks = data['landmarks']  # Shape: (num_frames, 21, 3)
labels = data['labels']        # Landmark names
fps = data['sampling_rate']    # Video frame rate
time = data['time_vector']     # Time in seconds

# Access specific landmark across all frames
wrist_positions = landmarks[:, 0, :]  # WRIST is index 0
index_tip = landmarks[:, 8, :]        # INDEX_FINGER_TIP is index 8

# Get x, y, z separately
x_coords = landmarks[:, :, 0]
y_coords = landmarks[:, :, 1]
z_coords = landmarks[:, :, 2]  # Depth relative to wrist
```

## MediaPipe Landmark Indices

```
0:  WRIST
1:  THUMB_CMC
2:  THUMB_MCP
3:  THUMB_IP
4:  THUMB_TIP
5:  INDEX_FINGER_MCP
6:  INDEX_FINGER_PIP
7:  INDEX_FINGER_DIP
8:  INDEX_FINGER_TIP
9:  MIDDLE_FINGER_MCP
10: MIDDLE_FINGER_PIP
11: MIDDLE_FINGER_DIP
12: MIDDLE_FINGER_TIP
13: RING_FINGER_MCP
14: RING_FINGER_PIP
15: RING_FINGER_DIP
16: RING_FINGER_TIP
17: PINKY_MCP
18: PINKY_PIP
19: PINKY_DIP
20: PINKY_TIP
```

## Coordinate System

- **x, y:** Normalized to image dimensions [0.0, 1.0]
  - Origin at **top-left** corner
  - x increases → right
  - y increases → down
- **z:** Depth relative to wrist (negative = closer to camera)

To convert to pixel coordinates:
```python
img_width, img_height = 1920, 1080
x_pixels = landmarks[:, :, 0] * img_width
y_pixels = landmarks[:, :, 1] * img_height
```

## Tips

- **Video Format:** MP4, AVI, MOV supported
- **Kalman Filtering:** Enabled by default, reduces jitter
- **Frame Rate:** Higher FPS = smoother tracking
- **Missing Hands:** Landmarks will be zeros for frames without hands
- **Multiple Hands:** Only first detected hand is tracked
