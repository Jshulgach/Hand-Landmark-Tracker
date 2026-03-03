import os
import sys

# Try to import compiled OptiTrack SDK module.
# Search order:
#   1) Existing PYTHONPATH / environment
#   2) Environment variable OPTITRACK_CAM_PY_PATH
#   3) Local ./Release folder next to this config file
_sdk_extra_paths = []
_env_sdk_path = os.environ.get("OPTITRACK_CAM_PY_PATH", "").strip()
if _env_sdk_path:
    _sdk_extra_paths.append(_env_sdk_path)

_local_release = os.path.join(os.path.dirname(__file__), "Release")
_sdk_extra_paths.append(_local_release)

for _p in _sdk_extra_paths:
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import optitrack_cam  # noqa: F401 - Exported for other modules to import
except Exception:
    optitrack_cam = None

# ==================== CAMERA SETTINGS ====================

# OptiTrack Specific Settings
CAMERA_EXPOSURE = 15  # Low exposure for maximum FPS
MJPEG_MODE = 6  # Core::MJPEGMode``
DISPLAY_SCALE = 1.0  # Global scaling for UI
GRID_COLS = 3  # Grid layout for visualization

# Primary camera index
PRIMARY_CAMERA_INDEX = 0

# Cameras that are physically mounted upside down (rotated 180 degrees).
# Their images will be rotated before MediaPipe detection, and the resulting
# 2D landmarks will be rotated back to match the calibration coordinate space.
UPSIDE_DOWN_CAMERAS = [2, 3]

# Camera resolution (will be set by actual hardware at runtime)
CAMERA_WIDTH = None  # Populated at runtime from camera
CAMERA_HEIGHT = None  # Populated at runtime from camera

# Camera count (will be set at runtime)
NUM_CAMERAS = None  # Populated at runtime
CAMERA_IDS = None  # Populated at runtime

# ==================== CALIBRATION SETTINGS ====================

# ChArUco Board Settings
CHARUCO_SQUARES_X = 6  # Number of squares horizontally (width)
CHARUCO_SQUARES_Y = 5  # Number of squares vertically (height)
CHARUCO_SQUARE_LENGTH = 0.040  # Size of each square in meters (40mm)
CHARUCO_MARKER_LENGTH = 0.030  # Size of each marker in meters (30mm)
ARUCO_DICT = "DICT_5X5_250"  # ArUco dictionary type

# Calibration data paths
CALIBRATION_DIR = os.path.join(os.path.dirname(__file__), "calibration_data")
CALIBRATION_FILE = os.path.join(CALIBRATION_DIR, "multi_camera_calib_latest.npz")

# Number of calibration images to capture per camera
NUM_CALIBRATION_IMAGES = 36  # 6 for each camera 6*n

# Calibration flags
CALIBRATION_FLAGS = 0  # Use default OpenCV calibration flags

# ==================== MEDIAPIPE SETTINGS ====================

# Hand detection
MAX_HANDS = 1
MODEL_COMPLEXITY = 1
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6

# Number of hand landmarks
NUM_LANDMARKS = 21

# ==================== KALMAN FILTER SETTINGS ====================

# 3D landmark filtering
KALMAN_3D_PROCESS_NOISE = 5e-3
KALMAN_3D_MEASUREMENT_NOISE = 2e-2

# 1D angle filtering
KALMAN_1D_PROCESS_NOISE = 0.3
KALMAN_1D_MEASUREMENT_NOISE = 3.0

# Smoothing backend
# Options: "kalman" (adaptive Kalman), "ema" (low-latency exponential smoothing)
SMOOTHING_METHOD = "kalman"

# EMA smoothing (used when SMOOTHING_METHOD="ema")
# Higher alpha = less smoothing / lower lag.
EMA_3D_ALPHA = 0.45
EMA_1D_ALPHA = 0.35

# ==================== UDP BROADCASTING ====================

# UDP settings
UDP_IP = "127.0.0.1"  # Localhost by default
UDP_PORT_LANDMARKS = 5005  # Port for landmark data
UDP_PORT_ANGLES = 5010  # Port for joint angle data

# ==================== GUI SETTINGS ====================

# Window size (automatically adjusts based on number of cameras)
GUI_WIDTH = 1600
GUI_HEIGHT = 900

# Primary video display size
VIDEO_PRIMARY_WIDTH = 960
VIDEO_PRIMARY_HEIGHT = 720

# Secondary video display size (for additional cameras)
VIDEO_SECONDARY_WIDTH = 480
VIDEO_SECONDARY_HEIGHT = 360

# Update rate (milliseconds)
FRAME_UPDATE_INTERVAL = 33  # ~30 FPS

# ==================== MODE SETTINGS ====================

# Operating mode (will be set at runtime)
SINGLE_CAMERA_MODE = None
MULTI_CAMERA_MODE = None

# ==================== JOINT ANGLE NAMES ====================

# All 14 joint angles tracked (3 per finger, 2 for thumb)
# Note: thumb_cmc_mcp matches the LSL broadcast stream channel naming
ANGLE_NAMES = [
    "index_mcp",
    "index_pip",
    "index_dip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "thumb_cmc_mcp",
    "thumb_ip",
]

# # --- Splay angles (comment out to disable splay Kalman filtering) --- it was so bad for now the calculation is very off it looks like
# ANGLE_NAMES += [
#     "index_splay",
#     "middle_splay",
#     "ring_splay",
#     "pinky_splay",
# ]
# # --- End splay ---

# ==================== TRIANGULATION SETTINGS ====================

# Minimum number of cameras that must see a landmark for triangulation
MIN_CAMERAS_FOR_TRIANGULATION = 2

# Maximum reprojection error for valid triangulation (in pixels)
# Used by RANSAC to classify inlier vs outlier cameras, and to detect
# occluded thumbs (if error > this, the camera is dropped for that landmark)
MAX_REPROJECTION_ERROR = 17.5

# Method for combining multiple camera views
# 'ransac' - RANSAC outlier rejection + N-view DLT (RECOMMENDED for 3+ cameras)
# 'n_view_dlt' - Triangulate using all available cameras simultaneously (no outlier rejection)
# 'simple_average' - Average all triangulated points
# 'weighted_average' - Weight by detection confidence
# 'reprojection' - Weight by camera confidence
# 'weighted_error' - Find the best pair of cameras with lowest reprojection error
# 'best_triplet' - Find the best 3 cameras using N-view DLT triangulation
TRIANGULATION_METHOD = "ransac"

# ==================== COORDINATE SYSTEM ====================

# Coordinate system for triangulated points
# 'camera0' - Use primary camera (CAMERA_IDS[0]) as world origin
# 'centroid' - Use centroid of all cameras as origin
WORLD_COORDINATE_SYSTEM = "camera0"

# ==================== HAND MATCHING SETTINGS ====================

# Maximum distance (normalized coordinates) for matching hands across cameras
HAND_MATCH_THRESHOLD = 0.3

# Use appearance matching in addition to position
USE_APPEARANCE_MATCHING = False

# ==================== VISUALIZATION SETTINGS ====================

# Draw camera frustums in 3D visualization
DRAW_CAMERA_FRUSTUMS = True

# Draw triangulated points
DRAW_TRIANGULATED_POINTS = True

# Color scheme for different cameras (BGR format)
CAMERA_COLORS = [
    (255, 0, 0),  # Blue for camera 0
    (0, 255, 0),  # Green for camera 1
    (0, 0, 255),  # Red for camera 2
    (255, 255, 0),  # Cyan for camera 3
    (255, 0, 255),  # Magenta for camera 4
    (0, 255, 255),  # Yellow for camera 5
]

# Extend colors if more cameras than predefined colors (done at runtime)
