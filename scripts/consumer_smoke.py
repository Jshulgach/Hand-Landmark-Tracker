"""Exercise an installed wheel from outside the source checkout."""
import argparse
import json
import subprocess
import sys
import sysconfig
import tempfile
from importlib.metadata import distribution
from pathlib import Path

import cv2
import numpy as np

from mavis_track import HandTracker, __version__, load_session


def verify_positive_video(tracker, source):
    capture = cv2.VideoCapture(str(source))
    present = 0
    try:
        assert capture.isOpened(), "The positive-hand video could not be opened"
        for _ in range(12):
            ok, frame = capture.read()
            assert ok, "The positive-hand video ended early"
            result = tracker.process(frame)
            for hand in result.hands:
                assert hand.landmarks.shape == (21, 3)
                assert np.isfinite(hand.landmarks).all()
                present += 1
        assert present > 0, "No hands were detected in the positive-hand video"
    finally:
        capture.release()
    print(f"Positive video inference passed: {present} detected hand slots across 12 frames")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, help="Existing positive-hand demo video for CI")
    args = parser.parse_args()
    installed = distribution("mavis-track")
    assert installed.metadata["Name"] == "mavis-track"
    assert installed.version == __version__
    command = Path(sysconfig.get_path("scripts")) / (
        "mavis-track.exe" if sys.platform == "win32" else "mavis-track"
    )
    assert command.is_file(), "The installed mavis-track command is missing"
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        subprocess.run([sys.executable, "-m", "mavis_track", "--help"],
                       check=True, cwd=folder)
        with HandTracker(max_hands=2) as tracker:
            result = tracker.process(np.zeros((240, 320, 3), np.uint8))
            assert result.hands == () and result.image_size == (320, 240)
            if args.video:
                verify_positive_video(tracker, args.video.resolve())
        clip = folder / "blank.avi"
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 29.97, (320, 240))
        assert writer.isOpened()
        for _ in range(3):
            writer.write(np.zeros((240, 320, 3), np.uint8))
        writer.release()
        capture = cv2.VideoCapture(str(clip))
        try:
            assert capture.isOpened(), "The generated video could not be opened"
            source_rate = float(capture.get(cv2.CAP_PROP_FPS))
            assert np.isfinite(source_rate) and source_rate > 0
        finally:
            capture.release()
        def cli(*args):
            subprocess.run([str(command), *map(str, args)],
                           check=True, cwd=folder)
        cli("doctor", "--backend", "webcam")
        cli("record", "--source", clip, "--max-hands", "2", "--output-dir", folder,
            "--session-name", "roundtrip")
        session = load_session(folder / "roundtrip")
        assert session.landmarks.shape == (3, 2, 21, 3)
        assert not session.valid.any()
        # Some native video backends round the requested 29.97 Hz to 30 Hz.
        # Recording must preserve the source's reported rate in either case.
        assert abs(session.sampling_rate - source_rate) < 0.001
        cli("replay", folder / "roundtrip", "--no-display")
        cli("export", folder / "roundtrip")
        manifest = json.loads((folder / "roundtrip" / "exports" / "export_manifest.json").read_text())
        assert manifest
        print("Installed mavis-track wheel: metadata, command, API and record/replay/export passed")

if __name__ == "__main__":
    main()
