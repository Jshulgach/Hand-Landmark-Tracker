import os
import sys

# Add path to the compiled OptiTrack module
sys.path.insert(
    0,
    r"c:\Users\NML\Desktop\Lokesh\Hand-Landmark-Tracker\src\unity_hand_tracking\optitrack_cam_py\Release",
)
import optitrack_cam  # noqa: F401 - Exported for other modules to import

# ==================== CAMERA SETTINGS ====================

# OptiTrack Specific Settings
CAMERA_EXPOSURE = 15  # Low exposure for maximum FPS
MJPEG_MODE = 6  # Core::MJPEGMode``
DISPLAY_SCALE = 1.0  # Global scaling for UI
GRID_COLS = 3  # Grid layout for visualization

# Primary camera index
PRIMARY_CAMERA_INDEX = 0

# Camera resolution (will be set by actual hardware at runtime)
CAMERA_WIDTH = None  # Populated at runtime from camera
CAMERA_HEIGHT = None  # Populated at runtime from camera

# Camera count (will be set at runtime)
NUM_CAMERAS = None  # Populated at runtime
CAMERA_IDS = None  # Populated at runtime

# ==================== CALIBRATION SETTINGS ====================

# Checkerboard pattern (inner corners)
CHECKERBOARD_ROWS = 4  # Number of inner corners vertically
CHECKERBOARD_COLS = 5  # Number of inner corners horizontally
CHECKERBOARD_SQUARE_SIZE = 40  # Size of each square

# ArUco marker settings (if board has markers)
USE_ARUCO_BOARD = True
ARUCO_DICT = "DICT_5X5_250"  # ArUco dictionary type
ARUCO_MARKER_SIZE = 30  # Marker size in mm

# Calibration data paths
CALIBRATION_DIR = "calibration_data"
CALIBRATION_FILE = os.path.join(CALIBRATION_DIR, "multi_camera_calib_latest.npz")

# Number of calibration images to capture per camera
NUM_CALIBRATION_IMAGES = 36  # 6 for each camera 6*n

# Calibration flags
CALIBRATION_FLAGS = 0  # Use default OpenCV calibration flags

# ==================== MEDIAPIPE SETTINGS ====================

# Hand detection
MAX_HANDS = 1
MODEL_COMPLEXITY = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Number of hand landmarks
NUM_LANDMARKS = 21

# ==================== KALMAN FILTER SETTINGS ====================

# 3D landmark filtering
KALMAN_3D_PROCESS_NOISE = 5e-3
KALMAN_3D_MEASUREMENT_NOISE = 5e-4

# 1D angle filtering
KALMAN_1D_PROCESS_NOISE = 0.3
KALMAN_1D_MEASUREMENT_NOISE = 3.0

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
# Used to detect occluded thumbs (if error > this, the thumb is dropped)
MAX_REPROJECTION_ERROR = 50.0

# Method for combining multiple camera views
# 'simple_average' - Average all triangulated points
# 'weighted_average' - Weight by detection confidence
# 'reprojection' - Weight by reprojection confidence
# 'weighted_error' - Find the best pair of cameras with lowest reprojection error
# 'best_triplet' - Find the best 3 cameras using N-view DLT triangulation
TRIANGULATION_METHOD = "best_triplet"

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
