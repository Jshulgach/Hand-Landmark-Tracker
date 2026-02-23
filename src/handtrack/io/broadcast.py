"""
Broadcasting utilities for sending tracking data via UDP or LSL.
"""

import json
import socket
import time
import threading
import numpy as np

# Try importing pysl, but make it optional
try:
    from pylsl import StreamInfo, StreamOutlet
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False


class DataBroadcaster:
    """Base class for data broadcasting."""
    def __init__(self):
        pass

    def send_landmarks(self, frame_count, timestamp, hands_data):
        raise NotImplementedError

    def send_angles(self, frame_count, timestamp, hands_data):
        raise NotImplementedError

    def close(self):
        pass


class UDPBroadcaster(DataBroadcaster):
    """Broadcasts data via UDP packets (JSON)."""
    
    def __init__(self, ip="127.0.0.1", port_landmarks=5005, port_angles=5010):
        super().__init__()
        self.ip = ip
        self.port_landmarks = port_landmarks
        self.port_angles = port_angles
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        print(f"UDP Broadcaster initialized on {self.ip} (Landmarks: {self.port_landmarks}, Angles: {self.port_angles})")

    def send_landmarks(self, frame_count, timestamp, hands_data):
        """
        Broadcast hand landmarks.
        hands_data: list of dicts with 'hand_index' and 'landmarks' (list of [x,y,z])
        """
        if not hands_data:
            return

        payload = {
            "frame": frame_count,
            "timestamp": timestamp,
            "num_hands": len(hands_data),
            "hands": hands_data
        }

        try:
            msg = json.dumps(payload).encode('utf-8')
            self.socket.sendto(msg, (self.ip, self.port_landmarks))
        except Exception as e:
            print(f"UDP Landmark Error: {e}")

    def send_angles(self, frame_count, timestamp, hands_data):
        """
        Broadcast joint angles.
        hands_data: list of dicts with 'hand_index' and 'angles' (dict of name->angle)
        """
        if not hands_data:
            return
            
        payload = {
            "frame": frame_count,
            "timestamp": timestamp,
            "hands": hands_data
        }

        try:
            msg = json.dumps(payload).encode('utf-8')
            self.socket.sendto(msg, (self.ip, self.port_angles))
        except Exception as e:
            print(f"UDP Angle Error: {e}")

    def close(self):
        if self.socket:
            self.socket.close()
            self.socket = None


class LSLBroadcaster(DataBroadcaster):
    """Broadcasts data via Lab Streaming Layer (LSL)."""
    
    def __init__(self, stream_name="HandTracker", source_id="hand_tracker_01"):
        super().__init__()
        if not LSL_AVAILABLE:
            print("Warning: pylsl not installed. LSL broadcasting will be disabled.")
            self.outlet_landmarks = None
            self.outlet_angles = None
            return

        # 1. Landmarks Stream
        # 63 channels (21 landmarks * 3 coords) per hand. 
        # For simplicity, we'll assume max 2 hands for now -> 126 channels.
        # Format: Hand1_L0_X, Hand1_L0_Y, Hand1_L0_Z, ... Hand2_L0_X...
        # We will use variable sampling rate (irregular).
        n_channels_lm = 2 * 21 * 3
        info_lm = StreamInfo(f"{stream_name}_Landmarks", 'MOCAP', n_channels_lm, 0, 'float32', f"{source_id}_lm")
        
        # Add channel metadata
        channels = info_lm.desc().append_child("channels")
        for hand_idx in range(2):
            for i in range(21):
                for axis in ['x', 'y', 'z']:
                    chan = channels.append_child("channel")
                    chan.append_child_value("label", f"Hand{hand_idx}_L{i}_{axis}")
                    chan.append_child_value("unit", "meters")
                    chan.append_child_value("type", "position")

        self.outlet_landmarks = StreamOutlet(info_lm)
        
        # 2. Angles Stream
        # Fixed channel count based on canonical angle order below.
        self.angle_keys = [
            "thumb_cmc_mcp", "thumb_ip",
            "index_mcp", "index_pip", "index_dip",
            "middle_mcp", "middle_pip", "middle_dip",
            "ring_mcp", "ring_pip", "ring_dip",
            "pinky_mcp", "pinky_pip", "pinky_dip",
        ]
        n_channels_ang = 2 * len(self.angle_keys)
        info_ang = StreamInfo(f"{stream_name}_Angles", 'MOCAP', n_channels_ang, 0, 'float32', f"{source_id}_ang")
        self.outlet_angles = StreamOutlet(info_ang)
        
        print(f"LSL Broadcaster initialized streams: {stream_name}_Landmarks, {stream_name}_Angles")

    def send_landmarks(self, frame_count, timestamp, hands_data):
        if not self.outlet_landmarks:
            return

        # Flatten data for LSL: [Hand0_..., Hand1_...]
        # Initialize with NaNs
        sample = [np.nan] * (2 * 21 * 3)
        
        for hand in hands_data:
            idx = hand.get('hand_index', 0)
            if idx >= 2: continue # Limit to 2 hands for fixed stream
            
            landmarks = hand.get('landmarks', [])
            flat_lm = np.array(landmarks).flatten()
            
            start_pos = idx * (21 * 3)
            end_pos = start_pos + len(flat_lm)
            
            if end_pos <= len(sample):
                sample[start_pos:end_pos] = flat_lm.tolist()
        
        self.outlet_landmarks.push_sample(sample, timestamp)

    def send_angles(self, frame_count, timestamp, hands_data):
        if not self.outlet_angles:
            return

        # Initialize with NaNs (2 hands * len(angle_keys))
        sample = [np.nan] * (2 * len(self.angle_keys))
        
        for hand in hands_data:
            idx = hand.get('hand_index', 0)
            if idx >= 2: continue
            
            angles = hand.get('angles', {})
            
            for i, key in enumerate(self.angle_keys):
                if key in angles:
                    sample[idx * len(self.angle_keys) + i] = angles[key]
                    
        self.outlet_angles.push_sample(sample, timestamp)
