import socket
import time
import math

# --- Configuration ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005 
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Timing parameters (in seconds)
move_duration = 2.0
hold_duration = 1.0
# Total cycle: 2s (up) + 1s (hold) + 2s (down) + 1s (hold) = 6 seconds
total_cycle = (move_duration * 2) + (hold_duration * 2)

# The Exact Order based on your Unity Bone Hierarchy (26 Joints)
joint_names = [
    "L_Wrist", "L_Palm", 
    "L_IndexMeta", "L_IndexProx", "L_IndexInter", "L_IndexDist", "L_IndexTip",
    "L_MiddleMeta", "L_MiddleProx", "L_MiddleInter", "L_MiddleDist", "L_MiddleTip",
    "L_RingMeta", "L_RingProx", "L_RingInter", "L_RingDist", "L_RingTip",
    "L_LittleMeta", "L_LittleProx", "L_LittleInter", "L_LittleDist", "L_LittleTip",
    "L_ThumbMeta", "L_ThumbProx", "L_ThumbDist", "L_ThumbTip"
]

print(f"Starting Left Hand Socket...")
print(f"Targeting: L_IndexProximal (Unity Element 3)")
print("Press Ctrl+C to stop.")

try:
    while True:
        # 26 joints * 3 axes = 78 values. Default all to "nan"
        data_list = ["nan"] * 78
        
        # Get position within the 6-second cycle
        elapsed = time.time() % total_cycle
        test_angle = 0.0
        phase_name = ""

        if elapsed < move_duration:
            # Phase 1: Moving 0 -> 60 (2 seconds)
            progress = elapsed / move_duration
            test_angle = progress * 60
            phase_name = "MOVING UP  "
        
        elif elapsed < (move_duration + hold_duration):
            # Phase 2: Hold at 60 (1 second)
            test_angle = 60.0
            phase_name = "HOLD (60°) "
            
        elif elapsed < (move_duration * 2 + hold_duration):
            # Phase 3: Moving 60 -> 0 (2 seconds)
            progress = (elapsed - move_duration - hold_duration) / move_duration
            test_angle = 60 - (progress * 60)
            phase_name = "MOVING DOWN"
            
        else:
            # Phase 4: Hold at 0 (1 second)
            test_angle = 0.0
            phase_name = "HOLD (0°)  "

        # --- MAPPING LOGIC ---
        # Unity Element 0 (Wrist)  -> Indices 0,1,2
        # Unity Element 1 (Palm)   -> Indices 3,4,5
        # Unity Element 2 (IdxMet) -> Indices 6,7,8
        # Unity Element 3 (IdxPrx) -> Indices 9,10,11
        
        # Set the X-rotation (index 9) for L_IndexProximal
        data_list[9] = f"{test_angle:.2f}" 

        # Package and Send
        message = ",".join(data_list)
        sock.sendto(bytes(message, "utf-8"), (UDP_IP, UDP_PORT))

        # Terminal Dashboard
        print(f"\rAngle: {test_angle:5.2f}° | Phase: {phase_name} | Sending to Port {UDP_PORT}", end="")
        
        time.sleep(0.02) # ~50 FPS

except KeyboardInterrupt:
    print("\n\nSocket Closed. Stopping sender...")
    sock.close()