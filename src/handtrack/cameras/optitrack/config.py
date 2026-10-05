"""Alias the established backend so configuration and monkeypatches are shared."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("unity_hand_tracking.optitrack_cam_py.config")
