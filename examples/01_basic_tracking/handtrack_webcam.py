import argparse
import sys
import os

# Add parent directory to path if handtrack is not installed
try:
    from handtrack.tracker import HandTracker
except ModuleNotFoundError:
    # Try adding src directory to path
    src_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
    sys.path.insert(0, os.path.abspath(src_path))
    from handtrack.tracker import HandTracker

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand Tracker with Optional 3D Visualization and Angle Logging")
    parser.add_argument('--camera_id', type=str, default='0', help='Camera ID or video file path')
    parser.add_argument('--img_size', type=int, nargs=2, default=(1080, 720), help='Image size (width height)')
    parser.add_argument('--save_angles', action='store_true', help='Save joint angle log to CSV')
    parser.add_argument('--out_path', type=str, default='angles.csv', help='Output path for saved angle log')
    parser.add_argument('--verbose', action='store_true', help='Print joint angles to terminal')
    args = parser.parse_args()

    # Convert camera input
    camera_id = int(args.camera_id) if args.camera_id.isdigit() else args.camera_id

    # Start the tracker
    tracker = HandTracker(
        source=camera_id,
        img_size=tuple(args.img_size),
        save_angles=args.save_angles,
        out_path=args.out_path,
        verbose=args.verbose
    )
    tracker.run()
