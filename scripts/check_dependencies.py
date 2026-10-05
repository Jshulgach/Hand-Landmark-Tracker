"""Check dependencies, including MediaPipe's known macOS wheel tag defect."""
import platform
import subprocess
import sys
from importlib.metadata import version


def known_macos_metadata_warning(output, system, machine, mediapipe_version):
    return (system == "darwin" and machine.lower() == "arm64"
            and mediapipe_version == "0.10.14"
            and output.strip() == "mediapipe 0.10.14 is not supported on this platform")


def verify_native_tracking():
    import numpy as np

    from mavis_track import HandTracker

    with HandTracker(max_hands=2) as tracker:
        result = tracker.process(np.zeros((240, 320, 3), np.uint8))
        assert result.hands == () and result.image_size == (320, 240)


def main():
    check = subprocess.run([sys.executable, "-m", "pip", "check"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           check=False)
    print(check.stdout, end="")
    if check.returncode == 0:
        return
    # The universal2 wheel ships both CPU architectures, but its WHEEL file
    # advertises only x86_64. Do not suppress any additional dependency errors.
    # https://github.com/google-ai-edge/mediapipe/issues/6030
    if check.returncode == 1 and known_macos_metadata_warning(
            check.stdout, sys.platform, platform.machine(), version("mediapipe")):
        verify_native_tracking()  # Import/model failures must still fail CI.
        print("Known MediaPipe wheel metadata defect; native ARM tracking passed.")
        return
    raise SystemExit(check.returncode)


if __name__ == "__main__":
    main()
