"""
Angle Forwarder: Receives joint angles from the hand tracker's UDP broadcaster
and forwards them to Unity in a 78-element comma-separated format.
"""

import json
import socket

# --- Config ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5010  # Tracker broadcasts angles here
UNITY_IP = "127.0.0.1"
UNITY_PORT = 5015  # Unity listens here

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(1.0)

unity_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Unity 26-joint hierarchy mapping
# 0: Wrist, 1: Palm (skipped)
# Fingers start at index 2
finger_starts = {"index": 2, "middle": 7, "ring": 12, "little": 17, "thumb": 22}

# Tracker angle names per finger
# These must match what finger_bend_angles() in the GUI actually produces:
#   index_mcp, index_pip, index_dip
#   middle_mcp, middle_pip, middle_dip
#   ring_mcp, ring_pip, ring_dip
#   pinky_mcp, pinky_pip, pinky_dip
#   thumb_mcp, thumb_ip
finger_angle_names = {
    "index": ["index_mcp", "index_pip", "index_dip"],
    "middle": ["middle_mcp", "middle_pip", "middle_dip"],
    "ring": ["ring_mcp", "ring_pip", "ring_dip"],
    "little": ["pinky_mcp", "pinky_pip", "pinky_dip"],
    "thumb": ["thumb_cmc_mcp", "thumb_ip"],
}

# # (comment out this entire block to disable splay forwarding)
# # --- Splay angles ---
# finger_splay_names = {
#     "index": "index_splay",
#     "middle": "middle_splay",
#     "ring": "ring_splay",
#     "little": "pinky_splay",
# }
# # --- End splay ---


print(f"FORWARDER: Listening on {UDP_IP}:{UDP_PORT}")
print(f"FORWARDER: Sending FINGERS ONLY to Unity {UNITY_IP}:{UNITY_PORT}")
print("Waiting for tracker data...\n")


def map_tracker_to_unity(tracker_angles):
    """Map tracker angle dict to Unity's 78-element (26 joints × 3 axes) string."""
    data_list = ["nan"] * 78

    for finger, start_idx in finger_starts.items():
        angles_for_finger = finger_angle_names[finger]
        # All fingers: skip bone 0 (metacarpal), angles map to bones 1, 2, 3
        # Thumb: 22+1=23 (Proximal), 22+2=24 (Distal)
        # Index: 2+1=3 (Proximal), 2+2=4 (Intermediate), 2+3=5 (Distal)
        for i, angle_name in enumerate(angles_for_finger):
            if angle_name in tracker_angles:
                joint_idx = (start_idx + i + 1) * 3
                val = tracker_angles[angle_name]
                data_list[joint_idx] = str(round(val, 4))

        # # --- Splay: write to Y slot of metacarpal (comment out to disable) ---
        # if finger in finger_splay_names:
        #     splay_name = finger_splay_names[finger]
        #     if splay_name in tracker_angles:
        #         metacarpal_y_slot = start_idx * 3 + 1  # Y slot of bone 0
        #         data_list[metacarpal_y_slot] = str(round(tracker_angles[splay_name], 4))
        # # --- End splay ---

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

            packet = json.loads(data.decode("utf-8"))

            # Packet format compatibility:
            #   Old: {"frame": int, "ts": float, "angles": [{"hand_index": 0, "angles": {...}}, ...]}
            #   New: {"frame": int, "timestamp": float, "hands": [{"hand_index": 0, "angles": {...}}, ...]}
            hand_list = packet.get("angles")
            if hand_list is None:
                hand_list = packet.get("hands", [])

            # Debug: print first packet
            if first_packet and hand_list:
                print("First packet received from tracker:")
                print(f"  Frame: {packet.get('frame')}")
                print(f"  Timestamp: {packet.get('ts', packet.get('timestamp'))}")
                print(f"  Hands: {len(hand_list)}")
                if hand_list:
                    angles = hand_list[0].get("angles", {})
                    print(f"  Angle names ({len(angles)}): {list(angles.keys())}")
                    # Show thumb values
                    thumb_keys = [k for k in angles if k.startswith("thumb")]
                    print(
                        f"  Thumb angles: { {k: round(angles[k], 2) for k in thumb_keys} }"
                    )
                    # Show mapped positions
                    for i, angle_name in enumerate(["thumb_cmc_mcp", "thumb_ip"]):
                        slot = (22 + i + 1) * 3
                        val = angles.get(angle_name, "MISSING")
                        print(
                            f"    {angle_name} -> Unity slot {slot} (joint {22 + i + 1}): {val}"
                        )
                print()
                first_packet = False

            if hand_list:
                # Take the first hand (hand_index 0, or whatever is first)
                first_hand = hand_list[0]
                angles = first_hand.get("angles", {})

                unity_message = map_tracker_to_unity(angles)
                unity_sock.sendto(unity_message.encode("utf-8"), (UNITY_IP, UNITY_PORT))

                frame_count += 1
                if frame_count % 30 == 0:
                    print(
                        f"\rSending Finger Data | Frame: {frame_count} | Hands: {len(hand_list)}",
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
    sock.close()
    unity_sock.close()
    print("Forwarder stopped cleanly")
