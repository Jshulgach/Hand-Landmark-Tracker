"""Compatibility entrypoint for the established OptiTrack application."""

def main():
    from unity_hand_tracking.optitrack_cam_py.mocap_handtrack_gui import main as run
    return run()
