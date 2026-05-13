"""OptiTrack multi-camera tracking application and utilities."""

__all__ = ["run_optitrack_gui"]


def run_optitrack_gui(*args, **kwargs):
    from .mocap_handtrack_gui import main

    return main(*args, **kwargs)
