"""
Hand Tracker package for tracking hand landmarks and computing joint angles
"""

__version__ = "0.0.3.10"
__author__ = "Jonathan Shulgach"
__email__ = "jshulgac@andrew.cmu.edu"
__license__ = "MIT"
__url__ = "https://github.com/jshulgach/Hand-Landmark-Tracker"
__description__ = "Python package for hand landmark tracking and kinematics suite"

__all__ = [
    "MultiCameraTracker",
    "MultiCameraCalibrator",
    "UDPBroadcaster",
    "LSLBroadcaster",
]


def __getattr__(name):
    if name == "MultiCameraTracker":
        from .tracker.stereo import MultiCameraTracker

        return MultiCameraTracker
    if name == "MultiCameraCalibrator":
        from .calibration import MultiCameraCalibrator

        return MultiCameraCalibrator
    if name == "UDPBroadcaster":
        from .io.broadcast import UDPBroadcaster

        return UDPBroadcaster
    if name == "LSLBroadcaster":
        from .io.broadcast import LSLBroadcaster

        return LSLBroadcaster
    raise AttributeError(f"module 'handtrack' has no attribute '{name}'")
