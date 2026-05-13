import socket
import time
import math

# --- Config---
UDP_IP = "127.0.0.1"
UDP_PORT = 5015
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Timing parameters
move_duration = 2.0
hold_duration = 1.0
total_cycle = (move_duration * 2) + (hold_duration * 2)

print(f"Starting Left Hand 'Thumbs Up' Sequence...")
print("Action: Curling all fingers except the thumb.")

try:
    while True:
        # Initialize 78 values (26 joints * 3 axes) as "nan"
        data_list = ["nan"] * 78
        
        elapsed = time.time() % total_cycle
        test_angle = 0.0
        phase_name = ""

        # --- Timing Logic ---
        if elapsed < move_duration:
            test_angle = (elapsed / move_duration) * 75  # Slightly deeper curl for a fist
            phase_name = "CLOSING FIST "
        elif elapsed < (move_duration + hold_duration):
            test_angle = 75.0
            phase_name = "THUMBS UP!   "
        elif elapsed < (move_duration * 2 + hold_duration):
            progress = (elapsed - move_duration - hold_duration) / move_duration
            test_angle = 75 - (progress * 75)
            phase_name = "OPENING HAND "
        else:
            test_angle = 0.0
            phase_name = "READY        "

        # Finger starts in Unity Hierarchy: 
        # Index=Element 2, Middle=Element 7, Ring=Element 12, Little=Element 17
        finger_starts = [2, 7, 12, 17]
        
        for start in finger_starts:
            # curl the Proximal (start+1), Intermediate (start+2), and Distal (start+3)
            # leave the Metacarpal (start) at
            for offset in [1,2,3]:
                joint_idx = (start + offset) * 3
                data_list[joint_idx] = f"{test_angle:.2f}"

        # THUMB (Elements 22-25) is not changed

        # Package and Send
        message = ",".join(data_list)
        sock.sendto(bytes(message, "utf-8"), (UDP_IP, UDP_PORT))

        print(f"\rCurrent Action: {phase_name} | Finger Angle: {test_angle:5.2f}°", end="")
        time.sleep(0.02)

except KeyboardInterrupt:
    print("\n\nSocket Closed.")
    sock.close()