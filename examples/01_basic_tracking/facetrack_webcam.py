import argparse
import sys
import os

# Add parent directory to path if handtrack is not installed
try:
    from handtrack.tracker import FaceTracker
except ModuleNotFoundError:
    # Try adding src directory to path
    src_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
    sys.path.insert(0, os.path.abspath(src_path))
    from handtrack.tracker import FaceTracker

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Tracker with MediaPipe Face Mesh")
    parser.add_argument('--camera_id', type=str, default='0', help='Camera ID or video file path')
    parser.add_argument('--img_size', type=int, nargs=2, default=(1080, 720), help='Image size (width height)')
    parser.add_argument('--max_faces', type=int, default=1, help='Maximum number of faces to detect')
    parser.add_argument('--confidence', type=float, default=0.5, help='Minimum detection confidence')
    parser.add_argument('--verbose', action='store_true', help='Print landmark info to terminal')
    args = parser.parse_args()

    # Convert camera input
    camera_id = int(args.camera_id) if args.camera_id.isdigit() else args.camera_id

    # Start the tracker
    tracker = FaceTracker(
        source=camera_id,
        img_size=tuple(args.img_size),
        max_faces=args.max_faces,
        confidence=args.confidence,
        verbose=args.verbose
    )
    tracker.run()
