import argparse
from src.old_utils.hand_tracker import HandTracker

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand Tracker with Optional 3D Visualization and Angle Logging")
    parser.add_argument('--camera_id', type=str, default='0', help='Camera ID or video file path')
    parser.add_argument('--img_size', type=int, nargs=2, default=(1080, 720), help='Image size (width height)')
    #parser.add_argument('--update_rate', type=int, default=5, help='Prediction update rate (in frames)')
    #parser.add_argument('--visualize_3d', action='store_true', help='Enable 3D landmark visualization')
    parser.add_argument('--save_angles', action='store_true', help='Save joint angle log to CSV')
    parser.add_argument('--out_path', type=str, default='angles.csv', help='Output path for saved angle log')
    parser.add_argument('--verbose', action='store_true', help='Print joint angles to terminal')
    args = parser.parse_args()

    # Convert camera input
    camera_id = int(args.camera_id) if args.camera_id.isdigit() else args.camera_id
    camera_id = r"G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\Open_Ephys\Jonathan\2025_05_07\media\HandDynamic.mp4"
    # Start the tracker
    tracker = HandTracker(
        source=camera_id,
        img_size=tuple(args.img_size),
        #update_rate=args.update_rate,
        #visualize_3d=args.visualize_3d,
        save_angles=args.save_angles,
        out_path=args.out_path,
        verbose=args.verbose
    )
    tracker.run()
