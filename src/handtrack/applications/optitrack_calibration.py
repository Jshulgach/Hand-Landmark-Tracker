"""OptiTrack entrypoint for shared calibration."""

def main():
    from ._calibration_runtime import main as run
    return run("optitrack")
