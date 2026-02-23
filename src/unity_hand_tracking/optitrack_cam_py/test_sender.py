"""
Test Sender - Broadcasts simulated hand tracking data via UDP and LSL

This script simulates hand tracking data and broadcasts it using the same
format as mocap_handtrack_gui.py. Useful for testing receivers without cameras.

Usage:
    python test_sender.py                    # Send at 30 Hz with both UDP and LSL
    python test_sender.py --rate 60          # Send at 60 Hz
    python test_sender.py --udp-only         # Only send via UDP
    python test_sender.py --lsl-only         # Only send via LSL
    python test_sender.py --hands 2          # Simulate 2 hands
"""

import argparse
import sys
import time

import numpy as np

# Add parent directory to path for handtrack imports
sys.path.insert(0, "../../src/")

try:
    from broadcast import LSLBroadcaster, UDPBroadcaster
except ImportError:
    print("Error: Could not import from handtrack.io.broadcast")
    print("Make sure you're running from the correct directory")
    sys.exit(1)


class HandDataSimulator:
    """Simulates realistic hand tracking data."""

    def __init__(self, num_hands=1):
        self.num_hands = num_hands
        self.frame_count = 0
        self.time_offset = time.time()

        # Angle names matching the broadcast format
        self.angle_keys = [
            "thumb_cmc_mcp",
            "thumb_ip",
            "index_mcp",
            "index_pip",
            "index_dip",
            "middle_mcp",
            "middle_pip",
            "middle_dip",
            "ring_mcp",
            "ring_pip",
            "ring_dip",
            "pinky_mcp",
            "pinky_pip",
            "pinky_dip",
        ]

    def generate_landmarks(self, hand_idx=0):
        """
        Generate 21 hand landmarks with realistic positions.
        Returns: list of 21 [x, y, z] coordinates
        """
        t = time.time() - self.time_offset

        # Base position (oscillates slightly for animation)
        base_x = 0.0 + 0.02 * np.sin(t * 0.5) + hand_idx * 0.3
        base_y = 0.0 + 0.02 * np.cos(t * 0.7)
        base_z = 0.5 + 0.01 * np.sin(t * 0.3)

        landmarks = []

        # Wrist (landmark 0)
        landmarks.append([base_x, base_y, base_z])

        # Thumb: 1(CMC), 2(MCP), 3(IP), 4(tip)
        for i in range(4):
            offset = (i + 1) * 0.025
            landmarks.append(
                [
                    base_x - 0.05 + offset * 0.5,
                    base_y + offset * 0.3,
                    base_z + offset * 0.2,
                ]
            )

        # Index finger: 5(MCP), 6(PIP), 7(DIP), 8(tip)
        for i in range(4):
            offset = (i + 1) * 0.03
            flex = np.sin(t * 2.0) * 0.01  # Animated flexion
            landmarks.append([base_x + 0.02, base_y + offset, base_z + flex])

        # Middle finger: 9(MCP), 10(PIP), 11(DIP), 12(tip)
        for i in range(4):
            offset = (i + 1) * 0.035
            landmarks.append([base_x, base_y + offset, base_z])

        # Ring finger: 13(MCP), 14(PIP), 15(DIP), 16(tip)
        for i in range(4):
            offset = (i + 1) * 0.032
            landmarks.append([base_x - 0.02, base_y + offset, base_z])

        # Pinky finger: 17(MCP), 18(PIP), 19(DIP), 20(tip)
        for i in range(4):
            offset = (i + 1) * 0.025
            landmarks.append([base_x - 0.04, base_y + offset, base_z - 0.01])

        return landmarks

    def generate_angles(self, hand_idx=0):
        """
        Generate realistic joint angles (in degrees).
        Returns: dict of angle_name -> angle_value
        """
        t = time.time() - self.time_offset

        # Simulate finger flexion with oscillation
        base_flex = 45.0 + 30.0 * np.sin(t * 1.5)  # Oscillate between 15-75 degrees

        angles = {}

        # Thumb (less flexion)
        angles["thumb_cmc_mcp"] = base_flex * 0.6 + np.random.randn() * 2.0
        angles["thumb_ip"] = base_flex * 0.5 + np.random.randn() * 2.0

        # Index finger
        angles["index_mcp"] = base_flex + np.random.randn() * 3.0
        angles["index_pip"] = base_flex * 1.2 + np.random.randn() * 3.0
        angles["index_dip"] = base_flex * 0.8 + np.random.randn() * 3.0

        # Middle finger (slightly different phase)
        phase_offset = 0.3
        mid_flex = 45.0 + 30.0 * np.sin(t * 1.5 + phase_offset)
        angles["middle_mcp"] = mid_flex + np.random.randn() * 3.0
        angles["middle_pip"] = mid_flex * 1.2 + np.random.randn() * 3.0
        angles["middle_dip"] = mid_flex * 0.8 + np.random.randn() * 3.0

        # Ring finger
        angles["ring_mcp"] = base_flex + np.random.randn() * 3.0
        angles["ring_pip"] = base_flex * 1.1 + np.random.randn() * 3.0
        angles["ring_dip"] = base_flex * 0.9 + np.random.randn() * 3.0

        # Pinky (slightly more flexion)
        angles["pinky_mcp"] = base_flex * 1.1 + np.random.randn() * 3.0
        angles["pinky_pip"] = base_flex * 1.3 + np.random.randn() * 3.0
        angles["pinky_dip"] = base_flex * 1.0 + np.random.randn() * 3.0

        return angles

    def get_frame_data(self):
        """Generate a complete frame of data for all hands."""
        self.frame_count += 1
        timestamp = time.time()

        # Generate landmark data
        landmarks_data = []
        for hand_idx in range(self.num_hands):
            landmarks = self.generate_landmarks(hand_idx)
            landmarks_data.append({"hand_index": hand_idx, "landmarks": landmarks})

        # Generate angle data
        angles_data = []
        for hand_idx in range(self.num_hands):
            angles = self.generate_angles(hand_idx)
            angles_data.append({"hand_index": hand_idx, "angles": angles})

        return self.frame_count, timestamp, landmarks_data, angles_data


def main():
    parser = argparse.ArgumentParser(description="Test hand tracking data sender")
    parser.add_argument(
        "--rate", type=float, default=30.0, help="Broadcast rate in Hz (default: 30)"
    )
    parser.add_argument(
        "--hands",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of hands to simulate (default: 1)",
    )
    parser.add_argument(
        "--udp-only", action="store_true", help="Only use UDP broadcasting"
    )
    parser.add_argument(
        "--lsl-only", action="store_true", help="Only use LSL broadcasting"
    )
    parser.add_argument(
        "--duration", type=float, default=0, help="Duration in seconds (0 = infinite)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("HAND TRACKING DATA SENDER (Test Mode)")
    print("=" * 70)
    print(f"Simulating {args.hands} hand(s) at {args.rate} Hz")
    print("Press Ctrl+C to stop\n")

    # Initialize simulator
    simulator = HandDataSimulator(num_hands=args.hands)

    # Initialize broadcasters
    udp_broadcaster = None
    lsl_broadcaster = None

    if not args.lsl_only:
        try:
            udp_broadcaster = UDPBroadcaster(
                ip="127.0.0.1", port_landmarks=5005, port_angles=5010
            )
            print("✓ UDP broadcaster initialized")
        except Exception as e:
            print(f"✗ UDP broadcaster failed: {e}")

    if not args.udp_only:
        try:
            lsl_broadcaster = LSLBroadcaster(
                stream_name="StereoHandTracker", source_id="test_sender_01"
            )
            print("✓ LSL broadcaster initialized")
        except Exception as e:
            print(f"✗ LSL broadcaster failed: {e}")

    if not udp_broadcaster and not lsl_broadcaster:
        print("\nError: No broadcasters initialized. Exiting.")
        return

    print("\n" + "-" * 70)
    print(f"{'Frame':<10} {'Time':<12} {'Landmarks':<15} {'Angles':<15} {'Hz':<10}")
    print("-" * 70)

    # Timing
    interval = 1.0 / args.rate
    start_time = time.time()
    last_report = start_time
    frames_since_report = 0

    try:
        while True:
            loop_start = time.time()

            # Generate data
            frame_count, timestamp, landmarks_data, angles_data = (
                simulator.get_frame_data()
            )

            # Broadcast landmarks
            if udp_broadcaster:
                udp_broadcaster.send_landmarks(frame_count, timestamp, landmarks_data)
            if lsl_broadcaster:
                lsl_broadcaster.send_landmarks(frame_count, timestamp, landmarks_data)

            # Broadcast angles
            if udp_broadcaster:
                udp_broadcaster.send_angles(frame_count, timestamp, angles_data)
            if lsl_broadcaster:
                lsl_broadcaster.send_angles(frame_count, timestamp, angles_data)

            frames_since_report += 1

            # Report every second
            elapsed = time.time() - last_report
            if elapsed >= 1.0:
                actual_rate = frames_since_report / elapsed
                print(
                    f"{frame_count:<10} "
                    f"{timestamp - start_time:<12.2f} "
                    f"{len(landmarks_data)} hands      "
                    f"{len(angles_data)} hands      "
                    f"{actual_rate:<10.1f}"
                )
                last_report = time.time()
                frames_since_report = 0

            # Check duration
            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                print(f"\nDuration of {args.duration}s reached. Stopping.")
                break

            # Sleep to maintain rate
            elapsed = time.time() - loop_start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\nStopping sender...")

    # Cleanup
    if udp_broadcaster:
        udp_broadcaster.close()
        print("✓ UDP broadcaster closed")

    print(f"\nTotal frames sent: {frame_count}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
