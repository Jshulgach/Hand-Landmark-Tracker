"""Webcam multi-camera tracking application and utilities."""

__all__ = ["run_webcam_gui"]


def run_webcam_gui(*args, **kwargs):
    from .mocap_handracker_gui import main

    return main(*args, **kwargs)


# Backward-compatible alias for older imports.
run_optitrack_gui = run_webcam_gui
