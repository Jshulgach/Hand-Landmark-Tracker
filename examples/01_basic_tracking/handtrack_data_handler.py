import socket
import time
import json

# --- Config ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5010  # Tracker sends here
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(1.0)

UNITY_IP = "127.0.0.1"
UNITY_PORT = 5015  # Unity listens here
unity_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Unity 26-joint hierarchy mapping
# 0: Wrist, 1: Palm (We skip these)
# Fingers start at index 2
finger_starts = {
    "index": 2,
    "middle": 7,
    "ring": 12,
    "little": 17,
    "thumb": 22
}

# Tracker angle names per finger
finger_angle_names = {
    "index": ["index_mcp", "index_pip", "index_dip"],
    "middle": ["middle_mcp", "middle_pip", "middle_dip"],
    "ring": ["ring_mcp", "ring_pip", "ring_dip"],
    "little": ["pinky_mcp", "pinky_pip", "pinky_dip"],
    "thumb": ["thumb_cmc_mcp", "thumb_ip"]
}

print(f"FORWARDER: Sending FINGERS ONLY to Unity {UNITY_IP}:{UNITY_PORT}")

# Initialize with 'nan' so Unity ignores indices we don't send
previous_data = ["nan"] * 78 

def map_tracker_to_unity(tracker_angles):
    global previous_data
    # Start with a fresh list of nans every frame to ensure we only 
    # affect what the tracker is currently seeing
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
try:
    while True:
        try:
            data, addr = sock.recvfrom(65536)
            if not data:
                continue

            hand_data = json.loads(data.decode("utf-8"))
            if hand_data.get("hands") and len(hand_data["hands"]) > 0:
                # We take the first hand detected
                first_hand = hand_data["hands"][0]
                angles = first_hand.get("angles", {})

                unity_message = map_tracker_to_unity(angles)
                unity_sock.sendto(unity_message.encode("utf-8"), (UNITY_IP, UNITY_PORT))

                frame_count += 1
                if frame_count % 10 == 0:
                    print(f"\rSending Finger Data | Frame: {frame_count}", end="", flush=True)

        except socket.timeout:
            continue
        except Exception as e:
            print(f"\nError: {e}")
            continue

except KeyboardInterrupt:
    print("\nStopping sender...")
    sock.close()
    unity_sock.close()