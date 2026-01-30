import csv
import glob
import json
import os
import re
import socket
import time
from datetime import datetime

# --- Config ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5010  # Tracker sends here
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Allow port reuse
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(1.0)

UNITY_IP = "127.0.0.1"
UNITY_PORT = 5015  # Unity listens here
unity_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Unity 26-joint hierarchy mapping
# 0: Wrist, 1: Palm (We skip these)
# Fingers start at index 2
finger_starts = {"index": 2, "middle": 7, "ring": 12, "little": 17, "thumb": 22}

# Tracker angle names per finger
finger_angle_names = {
    "index": ["index_mcp", "index_pip", "index_dip"],
    "middle": ["middle_mcp", "middle_pip", "middle_dip"],
    "ring": ["ring_mcp", "ring_pip", "ring_dip"],
    "little": ["pinky_mcp", "pinky_pip", "pinky_dip"],
    "thumb": ["thumb_cmc_mcp", "thumb_ip"],
}

# --- Recording State ---
is_recording = False
current_csv_file = None
csv_writer = None
session_number = None
row_count = 0

print(f"FORWARDER: Listening on {UDP_IP}:{UDP_PORT}")
print(f"FORWARDER: Sending FINGERS ONLY to Unity {UNITY_IP}:{UNITY_PORT}")
print("Waiting for tracker data...\n")


def get_next_dataset_number():
    """Scan data directory for existing dataset files and return next number."""

    data_dir = "data"
    pattern = os.path.join(data_dir, "dataset_*_*.csv")
    existing_files = glob.glob(pattern)

    if not existing_files:
        return 1

    numbers = []
    for filepath in existing_files:
        filename = os.path.basename(filepath)
        match = re.match(r"dataset_(\d+)_", filename)
        if match:
            numbers.append(int(match.group(1)))

    return max(numbers) + 1 if numbers else 1


def start_recording_session(timestamp):
    """Initialize a new recording session and create CSV file."""
    global is_recording, current_csv_file, csv_writer, session_number, row_count
    import os

    is_recording = True
    row_count = 0

    # Get next dataset number
    session_number = get_next_dataset_number()

    # Format timestamp as MM_DD_YYYY_HHmm
    dt = datetime.fromtimestamp(timestamp)
    time_str = dt.strftime("%m_%d_%Y_%H%M")

    # Create data directory if it doesn't exist
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # Create filename
    filename = os.path.join(data_dir, f"dataset_{session_number}_{time_str}.csv")

    # Open CSV file
    current_csv_file = open(filename, "w", newline="")
    csv_writer = csv.writer(current_csv_file)

    # Write header
    header = ["timestamp"]
    # EMG columns
    header.extend([f"emg_ch{i + 1}" for i in range(8)])
    # Unity values (78 columns)
    header.extend([f"unity_val{i}" for i in range(78)])

    csv_writer.writerow(header)
    current_csv_file.flush()  # Ensure header is written immediately

    print(f"\n[RECORDING STARTED] Session {session_number} → {filename}")


def stop_recording_session():
    """Close current recording session and save CSV file."""
    global is_recording, current_csv_file, csv_writer, session_number, row_count

    if not is_recording or current_csv_file is None:
        return

    is_recording = False

    # Close file
    current_csv_file.close()

    print(f"\n[RECORDING STOPPED] Session {session_number} saved: {row_count} rows")

    # Reset state
    current_csv_file = None
    csv_writer = None
    session_number = None
    row_count = 0


def map_tracker_to_unity(tracker_angles):
    # Start with a fresh list of nans every frame
    data_list = ["nan"] * 78

    for finger, start_idx in finger_starts.items():
        angles_for_finger = finger_angle_names[finger]

        if finger != "thumb":
            # Non-thumb fingers: MCP is the 2nd bone in the 5-bone chain
            # hierarchy: Metacarpal(0), MCP(1), PIP(2), DIP(3), Tip(4)
            for i, angle_name in enumerate(angles_for_finger):
                if angle_name in tracker_angles:
                    # (start_idx + bone_offset) * 3 axes
                    # i+1 skips the metacarpal bone
                    joint_idx = (start_idx + i + 1) * 3
                    val = tracker_angles[angle_name]
                    data_list[joint_idx] = str(round(val, 4))
        else:
            # Thumb logic
            for i, angle_name in enumerate(angles_for_finger):
                if angle_name in tracker_angles:
                    joint_idx = (start_idx + i) * 3
                    val = tracker_angles[angle_name]
                    data_list[joint_idx] = str(round(val, 4))

    return ",".join(data_list)


# --- Main Loop ---
frame_count = 0
first_packet = True

try:
    while True:
        try:
            data, addr = sock.recvfrom(65536)
            if not data:
                continue

            hand_data = json.loads(data.decode("utf-8"))

            # Extract metadata
            timestamp = hand_data.get("timestamp", time.time())
            recording_flag = hand_data.get("recording", False)
            hands = hand_data.get("hands", [])

            # Debug: Print first packet to verify format
            if first_packet:
                print("First packet received from tracker:")
                print(f"  Timestamp: {timestamp}")
                print(f"  Recording flag: {recording_flag}")
                print(f"  Hands detected: {len(hands)}")
                if hands:
                    print(
                        f"  Angles in first hand: {list(hands[0].get('angles', {}).keys())}"
                    )
                print()
                first_packet = False

            # Handle recording state transitions
            if recording_flag and not is_recording:
                start_recording_session(timestamp)
            elif not recording_flag and is_recording:
                stop_recording_session()

            # Process first hand if available
            if hands and len(hands) > 0:
                first_hand = hands[0]
                angles = first_hand.get("angles", {})

                # Map to Unity format (existing function)
                unity_message = map_tracker_to_unity(angles)

                # Forward to Unity
                unity_sock.sendto(unity_message.encode("utf-8"), (UNITY_IP, UNITY_PORT))

                # Record data if recording is active
                if is_recording and csv_writer is not None:
                    # Build CSV row
                    row = [timestamp]

                    # Add EMG placeholders (8 NaN values)
                    row.extend(["nan"] * 8)

                    # Add Unity values (78 values from the string)
                    unity_values = unity_message.split(",")
                    row.extend(unity_values)

                    # Write row
                    csv_writer.writerow(row)
                    row_count += 1

                    # Flush periodically for crash safety
                    if row_count % 30 == 0:
                        current_csv_file.flush()

                frame_count += 1
                if frame_count % 30 == 0:  # Update every 30 frames
                    status = "RECORDING" if is_recording else "Streaming"
                    print(
                        f"\r[{status}] Frame: {frame_count} | Hands: {len(hands)}",
                        end="",
                        flush=True,
                    )

        except socket.timeout:
            continue
        except json.JSONDecodeError as e:
            print(f"\nJSON decode error: {e}")
            continue
        except Exception as e:
            print(f"\nError: {e}")
            import traceback

            traceback.print_exc()
            continue

except KeyboardInterrupt:
    print("\n\nStopping forwarder...")
    if is_recording:
        stop_recording_session()
    sock.close()
    unity_sock.close()
    print("Forwarder stopped cleanly")
