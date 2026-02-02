"""
Configuration file for multi-camera hand tracking system.
Supports any number of cameras for improved 3D tracking.
"""

import os

# ==================== CAMERA SETTINGS ====================

# Camera IDs - ADD OR REMOVE CAMERAS HERE
# Example: [0, 1] for 2 cameras, [0, 1, 2] for 3 cameras, etc.
CAMERA_IDS = [0, 1]  # Modify this list to add/remove cameras

# Number of cameras (automatically calculated)
NUM_CAMERAS = len(CAMERA_IDS)

# Camera resolution (applied to all cameras)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Frame rate target
CAMERA_FPS = 30

# Primary camera index (used as reference coordinate system)
PRIMARY_CAMERA_INDEX = 0

# ==================== CALIBRATION SETTINGS ====================

# Checkerboard pattern (inner corners)
CHECKERBOARD_ROWS = 5     # Number of inner corners vertically
CHECKERBOARD_COLS = 7      # Number of inner corners horizontally
CHECKERBOARD_SQUARE_SIZE = 32.0  # Size of each square in mm

# ArUco marker settings (if board has markers)
USE_ARUCO_BOARD = False
ARUCO_DICT = "DICT_6X6_250"  # ArUco dictionary type
ARUCO_MARKER_SIZE = 15.0     # Marker size in mm

# Calibration data paths
CALIBRATION_DIR = "calibration_data"
CALIBRATION_FILE = os.path.join(CALIBRATION_DIR, "multi_camera_calib_latest.npz")

# Number of calibration images to capture per camera
NUM_CALIBRATION_IMAGES = 20

# Calibration flags
CALIBRATION_FLAGS = 0  # Use default OpenCV calibration flags

# ==================== MEDIAPIPE SETTINGS ====================

# Hand detection
MAX_HANDS = 2
MODEL_COMPLEXITY = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

# Number of hand landmarks
NUM_LANDMARKS = 21

# ==================== KALMAN FILTER SETTINGS ====================

# 3D landmark filtering
KALMAN_3D_PROCESS_NOISE = 1e-3
KALMAN_3D_MEASUREMENT_NOISE = 2e-4

# 1D angle filtering
KALMAN_1D_PROCESS_NOISE = 0.1
KALMAN_1D_MEASUREMENT_NOISE = 4.0

# ==================== UDP BROADCASTING ====================

# UDP settings
UDP_IP = "127.0.0.1"       # Localhost by default
UDP_PORT_LANDMARKS = 5005  # Port for landmark data
UDP_PORT_ANGLES = 5010     # Port for joint angle data

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

# Operating mode
SINGLE_CAMERA_MODE = (NUM_CAMERAS == 1)
MULTI_CAMERA_MODE = (NUM_CAMERAS > 1)

# ==================== JOINT ANGLE NAMES ====================

# All 14 joint angles tracked
ANGLE_NAMES = [
    "index_mcp", "index_pip", "index_dip",
    "middle_mcp", "middle_pip", "middle_dip",
    "ring_mcp", "ring_pip", "ring_dip",
    "pinky_mcp", "pinky_pip", "pinky_dip",
    "thumb_cmc_mcp", "thumb_ip"
]

# ==================== TRIANGULATION SETTINGS ====================

# Minimum number of cameras that must see a landmark for triangulation
MIN_CAMERAS_FOR_TRIANGULATION = 2

# Maximum reprojection error for valid triangulation (in pixels)
MAX_REPROJECTION_ERROR = 10.0

# Method for combining multiple camera views
# 'simple_average' - Average all triangulated points
# 'weighted_average' - Weight by detection confidence
# 'ransac' - Use RANSAC to reject outliers
TRIANGULATION_METHOD = 'simple_average'

# ==================== COORDINATE SYSTEM ====================

# Coordinate system for triangulated points
# 'camera0' - Use primary camera (CAMERA_IDS[0]) as world origin
# 'centroid' - Use centroid of all cameras as origin
WORLD_COORDINATE_SYSTEM = 'camera0'

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
    (255, 0, 0),      # Blue for camera 0
    (0, 255, 0),      # Green for camera 1
    (0, 0, 255),      # Red for camera 2
    (255, 255, 0),    # Cyan for camera 3
    (255, 0, 255),    # Magenta for camera 4
    (0, 255, 255),    # Yellow for camera 5
]

# Extend colors if more cameras than predefined colors
while len(CAMERA_COLORS) < NUM_CAMERAS:
    import random
    CAMERA_COLORS.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
