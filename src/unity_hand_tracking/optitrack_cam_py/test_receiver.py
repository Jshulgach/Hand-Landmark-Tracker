"""
Test Receiver - Receives hand tracking data via UDP and LSL

This script receives and displays hand tracking data from both UDP and LSL sources.
Can be used to test the mocap_handtrack_gui.py or test_sender.py output.

Usage:
    python test_receiver.py                  # Monitor both UDP and LSL
    python test_receiver.py --udp-only       # Only monitor UDP
    python test_receiver.py --lsl-only       # Only monitor LSL
    python test_receiver.py --verbose        # Show detailed packet data
"""

import argparse
import json
import socket
import sys
import threading
import time
from collections import deque
from datetime import datetime

try:
    from pylsl import StreamInlet, resolve_byprop
    LSL_AVAILABLE = True
except ImportError:
    print("Warning: pylsl not available. LSL receiving disabled.")
    LSL_AVAILABLE = False


class UDPReceiver:
    """Receives hand tracking data via UDP."""
    
    def __init__(self, port_landmarks=5005, port_angles=5010):
        self.port_landmarks = port_landmarks
        self.port_angles = port_angles
        self.running = False
        
        # Create sockets
        self.sock_landmarks = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_landmarks.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock_landmarks.bind(('', port_landmarks))
        self.sock_landmarks.settimeout(0.1)  # Non-blocking with timeout
        
        self.sock_angles = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_angles.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock_angles.bind(('', port_angles))
        self.sock_angles.settimeout(0.1)
        
        # Stats
        self.landmarks_count = 0
        self.angles_count = 0
        self.last_landmark_data = None
        self.last_angle_data = None
        self.landmark_timestamps = deque(maxlen=100)
        self.angle_timestamps = deque(maxlen=100)
        
        print(f"UDP Receiver initialized (Landmarks: {port_landmarks}, Angles: {port_angles})")
    
    def start(self):
        """Start receiver threads."""
        self.running = True
        
        self.thread_landmarks = threading.Thread(target=self._receive_landmarks, daemon=True)
        self.thread_angles = threading.Thread(target=self._receive_angles, daemon=True)
        
        self.thread_landmarks.start()
        self.thread_angles.start()
    
    def stop(self):
        """Stop receiver threads."""
        self.running = False
        if hasattr(self, 'thread_landmarks'):
            self.thread_landmarks.join(timeout=1)
        if hasattr(self, 'thread_angles'):
            self.thread_angles.join(timeout=1)
        
        self.sock_landmarks.close()
        self.sock_angles.close()
    
    def _receive_landmarks(self):
        """Thread to receive landmark packets."""
        while self.running:
            try:
                data, addr = self.sock_landmarks.recvfrom(65536)
                packet = json.loads(data.decode('utf-8'))
                self.landmarks_count += 1
                self.last_landmark_data = packet
                self.landmark_timestamps.append(time.time())
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"UDP Landmark receive error: {e}")
    
    def _receive_angles(self):
        """Thread to receive angle packets."""
        while self.running:
            try:
                data, addr = self.sock_angles.recvfrom(65536)
                packet = json.loads(data.decode('utf-8'))
                self.angles_count += 1
                self.last_angle_data = packet
                self.angle_timestamps.append(time.time())
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"UDP Angle receive error: {e}")
    
    def get_stats(self):
        """Get reception statistics."""
        # Calculate rates
        lm_rate = 0
        if len(self.landmark_timestamps) > 1:
            span = self.landmark_timestamps[-1] - self.landmark_timestamps[0]
            if span > 0:
                lm_rate = (len(self.landmark_timestamps) - 1) / span
        
        ang_rate = 0
        if len(self.angle_timestamps) > 1:
            span = self.angle_timestamps[-1] - self.angle_timestamps[0]
            if span > 0:
                ang_rate = (len(self.angle_timestamps) - 1) / span
        
        return {
            'landmarks_count': self.landmarks_count,
            'angles_count': self.angles_count,
            'landmarks_rate': lm_rate,
            'angles_rate': ang_rate,
            'last_landmark': self.last_landmark_data,
            'last_angle': self.last_angle_data
        }


class LSLReceiver:
    """Receives hand tracking data via LSL."""
    
    def __init__(self, stream_name="StereoHandTracker"):
        self.stream_name = stream_name
        self.inlet_landmarks = None
        self.inlet_angles = None
        self.running = False
        
        # Stats
        self.landmarks_count = 0
        self.angles_count = 0
        self.last_landmark_sample = None
        self.last_angle_sample = None
        self.landmark_timestamps = deque(maxlen=100)
        self.angle_timestamps = deque(maxlen=100)
        
        # Try to resolve streams
        print(f"Looking for LSL streams: {stream_name}_Landmarks, {stream_name}_Angles")
        
        try:
            streams = resolve_byprop("name", f"{stream_name}_Landmarks", timeout=5)
            if streams:
                self.inlet_landmarks = StreamInlet(streams[0])
                print(f"  ✓ Connected to {stream_name}_Landmarks")
            else:
                print(f"  ✗ {stream_name}_Landmarks not found")
        except Exception as e:
            print(f"  ✗ Error connecting to landmarks stream: {e}")
        
        try:
            streams = resolve_byprop("name", f"{stream_name}_Angles", timeout=5)
            if streams:
                self.inlet_angles = StreamInlet(streams[0])
                print(f"  ✓ Connected to {stream_name}_Angles")
            else:
                print(f"  ✗ {stream_name}_Angles not found")
        except Exception as e:
            print(f"  ✗ Error connecting to angles stream: {e}")
    
    def start(self):
        """Start receiver threads."""
        self.running = True
        
        if self.inlet_landmarks:
            self.thread_landmarks = threading.Thread(target=self._receive_landmarks, daemon=True)
            self.thread_landmarks.start()
        
        if self.inlet_angles:
            self.thread_angles = threading.Thread(target=self._receive_angles, daemon=True)
            self.thread_angles.start()
    
    def stop(self):
        """Stop receiver threads."""
        self.running = False
    
    def _receive_landmarks(self):
        """Thread to receive landmark samples."""
        while self.running:
            try:
                sample, timestamp = self.inlet_landmarks.pull_sample(timeout=0.1)
                if sample:
                    self.landmarks_count += 1
                    self.last_landmark_sample = (sample, timestamp)
                    self.landmark_timestamps.append(time.time())
            except Exception as e:
                if self.running:
                    print(f"LSL Landmark receive error: {e}")
    
    def _receive_angles(self):
        """Thread to receive angle samples."""
        while self.running:
            try:
                sample, timestamp = self.inlet_angles.pull_sample(timeout=0.1)
                if sample:
                    self.angles_count += 1
                    self.last_angle_sample = (sample, timestamp)
                    self.angle_timestamps.append(time.time())
            except Exception as e:
                if self.running:
                    print(f"LSL Angle receive error: {e}")
    
    def get_stats(self):
        """Get reception statistics."""
        # Calculate rates
        lm_rate = 0
        if len(self.landmark_timestamps) > 1:
            span = self.landmark_timestamps[-1] - self.landmark_timestamps[0]
            if span > 0:
                lm_rate = (len(self.landmark_timestamps) - 1) / span
        
        ang_rate = 0
        if len(self.angle_timestamps) > 1:
            span = self.angle_timestamps[-1] - self.angle_timestamps[0]
            if span > 0:
                ang_rate = (len(self.angle_timestamps) - 1) / span
        
        return {
            'landmarks_count': self.landmarks_count,
            'angles_count': self.angles_count,
            'landmarks_rate': lm_rate,
            'angles_rate': ang_rate,
            'last_landmark': self.last_landmark_sample,
            'last_angle': self.last_angle_sample
        }


def print_landmark_detail(data, source="UDP"):
    """Print detailed landmark information."""
    if source == "UDP":
        print(f"\n  [UDP LANDMARKS] Frame: {data.get('frame', 'N/A')}")
        print(f"    Timestamp: {data.get('timestamp', 0):.6f}")
        print(f"    Hands: {data.get('num_hands', 0)}")
        
        for hand in data.get('hands', []):
            idx = hand.get('hand_index', 0)
            landmarks = hand.get('landmarks', [])
            print(f"\n    Hand {idx}: {len(landmarks)} landmarks")
            if landmarks:
                print(f"      Wrist (L0):       [{landmarks[0][0]:.4f}, {landmarks[0][1]:.4f}, {landmarks[0][2]:.4f}]")
                print(f"      Index tip (L8):   [{landmarks[8][0]:.4f}, {landmarks[8][1]:.4f}, {landmarks[8][2]:.4f}]")
                print(f"      Thumb tip (L4):   [{landmarks[4][0]:.4f}, {landmarks[4][1]:.4f}, {landmarks[4][2]:.4f}]")
    
    else:  # LSL
        sample, timestamp = data
        print(f"\n  [LSL LANDMARKS] Timestamp: {timestamp:.6f}")
        print(f"    Channels: {len(sample)}")
        # Parse Hand 0 data (first 63 channels)
        if len(sample) >= 63:
            wrist = sample[0:3]
            index_tip = sample[24:27]  # L8 * 3
            thumb_tip = sample[12:15]  # L4 * 3
            
            print(f"\n    Hand 0:")
            print(f"      Wrist (L0):       [{wrist[0]:.4f}, {wrist[1]:.4f}, {wrist[2]:.4f}]")
            print(f"      Index tip (L8):   [{index_tip[0]:.4f}, {index_tip[1]:.4f}, {index_tip[2]:.4f}]")
            print(f"      Thumb tip (L4):   [{thumb_tip[0]:.4f}, {thumb_tip[1]:.4f}, {thumb_tip[2]:.4f}]")


def print_angle_detail(data, source="UDP"):
    """Print detailed angle information."""
    if source == "UDP":
        print(f"\n  [UDP ANGLES] Frame: {data.get('frame', 'N/A')}")
        print(f"    Timestamp: {data.get('timestamp', 0):.6f}")
        
        for hand in data.get('hands', []):
            idx = hand.get('hand_index', 0)
            angles = hand.get('angles', {})
            print(f"\n    Hand {idx} Angles (degrees):")
            print(f"      Thumb CMC-MCP: {angles.get('thumb_cmc_mcp', 0):.2f}°")
            print(f"      Index MCP:     {angles.get('index_mcp', 0):.2f}°")
            print(f"      Index PIP:     {angles.get('index_pip', 0):.2f}°")
            print(f"      Middle MCP:    {angles.get('middle_mcp', 0):.2f}°")
    
    else:  # LSL
        sample, timestamp = data
        print(f"\n  [LSL ANGLES] Timestamp: {timestamp:.6f}")
        print(f"    Channels: {len(sample)}")
        # First 14 channels are Hand 0 angles
        angle_names = [
            "thumb_cmc_mcp", "thumb_ip",
            "index_mcp", "index_pip", "index_dip",
            "middle_mcp", "middle_pip", "middle_dip",
            "ring_mcp", "ring_pip", "ring_dip",
            "pinky_mcp", "pinky_pip", "pinky_dip",
        ]
        
        print(f"\n    Hand 0 Angles (degrees):")
        for i, name in enumerate(angle_names[:5]):  # Show first 5
            if i < len(sample):
                print(f"      {name:15s}: {sample[i]:.2f}°")


def main():
    parser = argparse.ArgumentParser(description="Test hand tracking data receiver")
    parser.add_argument('--udp-only', action='store_true',
                        help='Only receive via UDP')
    parser.add_argument('--lsl-only', action='store_true',
                        help='Only receive via LSL')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed packet information')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='Stats reporting interval in seconds (default: 1.0)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("HAND TRACKING DATA RECEIVER (Test Mode)")
    print("=" * 70)
    print("Monitoring incoming data streams...")
    print("Press Ctrl+C to stop\n")
    
    # Initialize receivers
    udp_receiver = None
    lsl_receiver = None
    
    if not args.lsl_only:
        try:
            udp_receiver = UDPReceiver(port_landmarks=5005, port_angles=5010)
            udp_receiver.start()
            print("✓ UDP receiver started")
        except Exception as e:
            print(f"✗ UDP receiver failed: {e}")
    
    if not args.udp_only and LSL_AVAILABLE:
        try:
            lsl_receiver = LSLReceiver(stream_name="StereoHandTracker")
            lsl_receiver.start()
            print("✓ LSL receiver started")
        except Exception as e:
            print(f"✗ LSL receiver failed: {e}")
    
    if not udp_receiver and not lsl_receiver:
        print("\nError: No receivers initialized. Exiting.")
        return
    
    print("\n" + "-" * 70)
    
    last_report = time.time()
    last_verbose = time.time()
    
    try:
        while True:
            time.sleep(0.1)
            
            now = time.time()
            elapsed = now - last_report
            
            # Regular stats report
            if elapsed >= args.interval:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Reception Statistics:")
                
                if udp_receiver:
                    stats = udp_receiver.get_stats()
                    print(f"  UDP:")
                    print(f"    Landmarks: {stats['landmarks_count']} packets ({stats['landmarks_rate']:.1f} Hz)")
                    print(f"    Angles:    {stats['angles_count']} packets ({stats['angles_rate']:.1f} Hz)")
                
                if lsl_receiver:
                    stats = lsl_receiver.get_stats()
                    print(f"  LSL:")
                    print(f"    Landmarks: {stats['landmarks_count']} samples ({stats['landmarks_rate']:.1f} Hz)")
                    print(f"    Angles:    {stats['angles_count']} samples ({stats['angles_rate']:.1f} Hz)")
                
                print("-" * 70)
                last_report = now
            
            # Verbose output (every 5 seconds)
            if args.verbose and (now - last_verbose) >= 5.0:
                if udp_receiver:
                    stats = udp_receiver.get_stats()
                    if stats['last_landmark']:
                        print_landmark_detail(stats['last_landmark'], "UDP")
                    if stats['last_angle']:
                        print_angle_detail(stats['last_angle'], "UDP")
                
                if lsl_receiver:
                    stats = lsl_receiver.get_stats()
                    if stats['last_landmark']:
                        print_landmark_detail(stats['last_landmark'], "LSL")
                    if stats['last_angle']:
                        print_angle_detail(stats['last_angle'], "LSL")
                
                last_verbose = now
    
    except KeyboardInterrupt:
        print("\n\nStopping receiver...")
    
    # Cleanup
    if udp_receiver:
        udp_receiver.stop()
        print("✓ UDP receiver stopped")
    
    if lsl_receiver:
        lsl_receiver.stop()
        print("✓ LSL receiver stopped")
    
    # Final stats
    print("\nFinal Statistics:")
    if udp_receiver:
        stats = udp_receiver.get_stats()
        print(f"  UDP - Landmarks: {stats['landmarks_count']}, Angles: {stats['angles_count']}")
    
    if lsl_receiver:
        stats = lsl_receiver.get_stats()
        print(f"  LSL - Landmarks: {stats['landmarks_count']}, Angles: {stats['angles_count']}")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
