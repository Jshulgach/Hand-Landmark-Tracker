"""
Stream hand landmarks to Lab Streaming Layer (LSL) in real-time.

This example demonstrates how to:
1. Capture hand landmarks from a webcam using MediaPipe
2. Apply Kalman filtering for smooth tracking
3. Stream the 3D landmark positions via LSL for other applications to consume

The LSL stream contains 63 channels (21 landmarks × 3 coordinates: x, y, z)
and can be logged using tools like nml-wtf-exo-logger or consumed by other
real-time applications.

Usage:
    python stream_landmarks_lsl.py
    python stream_landmarks_lsl.py --source 0              # Use default webcam
    python stream_landmarks_lsl.py --source 1              # Use secondary webcam
    python stream_landmarks_lsl.py --max_hands 2           # Track two hands
    python stream_landmarks_lsl.py --no_kalman             # Disable smoothing
    python stream_landmarks_lsl.py --stream_name MyHands   # Custom stream name

Requirements:
    pip install pylsl
"""

import argparse
import sys
import os
import time
import cv2
import numpy as np

# Add parent directory to path if handtrack is not installed
try:
    from handtrack.tracker import HandTracker
except ModuleNotFoundError:
    # Try adding src directory to path
    src_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
    sys.path.insert(0, os.path.abspath(src_path))
    from handtrack.tracker import HandTracker

from pylsl import StreamInfo, StreamOutlet
import mediapipe as mp


def create_lsl_outlet(stream_name="HandLandmarks", num_landmarks=21, fps=30):
    """
    Create an LSL outlet for streaming hand landmark data.
    
    Args:
        stream_name (str): Name of the LSL stream
        num_landmarks (int): Number of landmarks per hand (default: 21)
        fps (float): Expected sampling rate in Hz
        
    Returns:
        StreamOutlet: LSL outlet ready to push samples
    """
    n_channels = num_landmarks * 3  # x, y, z for each landmark
    
    # Create stream info with detailed metadata
    info = StreamInfo(
        name=stream_name,
        type='Mocap',  # Motion capture type
        channel_count=n_channels,
        nominal_srate=fps,
        channel_format='float32',
        source_id='handtrack_' + str(int(time.time()))
    )
    
    # Add channel metadata (labels for each coordinate)
    channels = info.desc().append_child("channels")
    landmark_names = [lm.name for lm in mp.solutions.hands.HandLandmark]
    
    for i, lm_name in enumerate(landmark_names):
        for coord in ['x', 'y', 'z']:
            ch = channels.append_child("channel")
            ch.append_child_value("label", f"{lm_name}_{coord}")
            ch.append_child_value("unit", "normalized" if coord in ['x', 'y'] else "depth")
            ch.append_child_value("type", "Position")
    
    # Add additional metadata
    info.desc().append_child_value("manufacturer", "MediaPipe")
    info.desc().append_child_value("model", "HandLandmarker")
    
    # Create and return the outlet
    outlet = StreamOutlet(info, chunk_size=1, max_buffered=360)
    print(f"[LSL] Created outlet: '{stream_name}' with {n_channels} channels @ {fps} Hz")
    print(f"[LSL] Stream source_id: {info.source_id()}")
    
    return outlet


def stream_landmarks(source=0, stream_name="HandLandmarks", max_hands=1, 
                     apply_kalman=True, confidence=0.8, fps=30, visualize=True):
    """
    Stream hand landmarks to LSL in real-time.
    
    Args:
        source (int): Camera index (0 for default webcam)
        stream_name (str): Name for the LSL stream
        max_hands (int): Maximum number of hands to track
        apply_kalman (bool): Apply Kalman filtering for smoothing
        confidence (float): Minimum detection confidence (0.0 to 1.0)
        fps (int): Target frames per second
        visualize (bool): Show live video feed with landmarks
    """
    # Initialize hand tracker
    print(f"[Tracker] Initializing HandTracker (source={source}, max_hands={max_hands})")
    tracker = HandTracker(
        source=source,
        video_fps=fps,
        max_hands=max_hands,
        confidence=confidence,
        apply_kalman=apply_kalman,
        verbose=False
    )
    
    # Create LSL outlet
    outlet = create_lsl_outlet(stream_name=stream_name, num_landmarks=21, fps=fps)
    
    print("\n" + "="*60)
    print("STREAMING HAND LANDMARKS TO LSL")
    print("="*60)
    print(f"Stream Name: {stream_name}")
    print(f"Kalman Filtering: {'Enabled' if apply_kalman else 'Disabled'}")
    print(f"Max Hands: {max_hands}")
    print(f"Visualization: {'Enabled' if visualize else 'Disabled'}")
    print("\nPress ESC to stop streaming...")
    print("="*60 + "\n")
    
    frame_count = 0
    start_time = time.time()
    last_fps_time = start_time
    fps_counter = 0
    current_fps = 0
    
    try:
        while tracker.cap.isOpened():
            # Get frame from camera
            frame = tracker.get_image(flip_frame=True)
            if frame is None:
                print("[Warning] Failed to read frame")
                break
            
            # Detect hands
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = tracker.hands.process(rgb)
            
            # Extract and stream landmarks
            if results and results.multi_hand_landmarks:
                for hand_idx, landmarks in enumerate(results.multi_hand_landmarks):
                    # Extract raw landmark array
                    landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
                    
                    # Apply Kalman filtering if enabled
                    if apply_kalman:
                        filtered_landmarks = np.array([
                            tracker.kalman_filters[i].update(landmark_array[i]) 
                            for i in range(21)
                        ])
                    else:
                        filtered_landmarks = landmark_array
                    
                    # Flatten to 1D array (63 values: 21 landmarks × 3 coords)
                    sample = filtered_landmarks.flatten().tolist()
                    
                    # Push to LSL
                    outlet.push_sample(sample)
                    
                    # Visualize if enabled
                    if visualize:
                        # Draw raw landmarks
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                            mp.solutions.drawing_styles.get_default_hand_connections_style()
                        )
                        
                        # Draw filtered landmarks in white if Kalman is enabled
                        if apply_kalman:
                            for x, y, _ in filtered_landmarks:
                                cv2.circle(frame, 
                                         (int(x * frame.shape[1]), int(y * frame.shape[0])), 
                                         4, (255, 255, 255), -1)
            else:
                # No hands detected - push zeros to maintain timing
                sample = [0.0] * 63
                outlet.push_sample(sample)
            
            # Calculate FPS
            fps_counter += 1
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                current_fps = fps_counter / (current_time - last_fps_time)
                fps_counter = 0
                last_fps_time = current_time
            
            # Display visualization
            if visualize:
                # Add info overlay
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Hands: {len(results.multi_hand_landmarks) if results and results.multi_hand_landmarks else 0}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Kalman: {'ON' if apply_kalman else 'OFF'}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("LSL Hand Landmark Streaming", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    break
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n[Info] Streaming interrupted by user")
    
    finally:
        # Cleanup
        elapsed_time = time.time() - start_time
        avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        print("\n" + "="*60)
        print("STREAMING STOPPED")
        print("="*60)
        print(f"Total Frames: {frame_count}")
        print(f"Duration: {elapsed_time:.2f} seconds")
        print(f"Average FPS: {avg_fps:.2f}")
        print("="*60)
        
        tracker.cap.release()
        if visualize:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Stream hand landmarks to LSL for real-time applications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stream from default webcam with Kalman filtering
  python stream_landmarks_lsl.py
  
  # Stream from secondary camera
  python stream_landmarks_lsl.py --source 1
  
  # Track two hands simultaneously
  python stream_landmarks_lsl.py --max_hands 2
  
  # Disable Kalman smoothing for raw landmarks
  python stream_landmarks_lsl.py --no_kalman
  
  # Custom stream name (useful for multiple streams)
  python stream_landmarks_lsl.py --stream_name LeftHand
  
  # Run headless (no visualization window)
  python stream_landmarks_lsl.py --no_visualize

To log the stream, open another terminal and run:
  nml-wtf-exo-logger
        """
    )
    
    parser.add_argument("--source", type=int, default=0,
                       help="Camera source index (default: 0)")
    parser.add_argument("--stream_name", type=str, default="HandLandmarks",
                       help="LSL stream name (default: HandLandmarks)")
    parser.add_argument("--max_hands", type=int, default=1,
                       help="Maximum number of hands to track (default: 1)")
    parser.add_argument("--no_kalman", action="store_true",
                       help="Disable Kalman filtering")
    parser.add_argument("--confidence", type=float, default=0.8,
                       help="Minimum detection confidence 0.0-1.0 (default: 0.8)")
    parser.add_argument("--fps", type=int, default=30,
                       help="Target frames per second (default: 30)")
    parser.add_argument("--no_visualize", action="store_true",
                       help="Disable visualization window")
    
    args = parser.parse_args()
    
    # Run streaming
    stream_landmarks(
        source=args.source,
        stream_name=args.stream_name,
        max_hands=args.max_hands,
        apply_kalman=not args.no_kalman,
        confidence=args.confidence,
        fps=args.fps,
        visualize=not args.no_visualize
    )


if __name__ == "__main__":
    main()
