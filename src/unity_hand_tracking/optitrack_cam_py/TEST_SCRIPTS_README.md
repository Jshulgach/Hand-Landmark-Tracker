# Test Sender and Receiver Scripts

These scripts allow you to test hand tracking data broadcasting without needing cameras.

## Files

- **test_sender.py** - Broadcasts simulated hand tracking data
- **test_receiver.py** - Receives and displays hand tracking data
- **test_lsl_streaming.py** - Tests LSL stream connectivity

## Quick Start

### 1. Test with Simulated Data

**Terminal 1 - Start the sender:**
```bash
python test_sender.py
```

**Terminal 2 - Start the receiver:**
```bash
python test_receiver.py
```

You should see data flowing at ~30 Hz for both landmarks and angles.

### 2. Test with Real Hand Tracking GUI

**Terminal 1 - Start the GUI:**
```bash
python mocap_handtrack_gui.py
```
Then click "Start Tracking" in the GUI.

**Terminal 2 - Monitor the data:**
```bash
python test_receiver.py --verbose
```

### 3. Test LSL Only

Check if LSL streams are available:
```bash
python test_lsl_streaming.py
```

Run self-test to verify LSL is working:
```bash
python test_lsl_streaming.py --selftest
```

## Usage Examples

### Sender Options

```bash
# Send at 60 Hz instead of 30 Hz
python test_sender.py --rate 60

# Simulate 2 hands
python test_sender.py --hands 2

# Only broadcast via UDP (no LSL)
python test_sender.py --udp-only

# Only broadcast via LSL (no UDP)
python test_sender.py --lsl-only

# Run for 30 seconds then stop
python test_sender.py --duration 30
```

### Receiver Options

```bash
# Show detailed packet data every 5 seconds
python test_receiver.py --verbose

# Only receive UDP data
python test_receiver.py --udp-only

# Only receive LSL data
python test_receiver.py --lsl-only

# Report stats every 2 seconds instead of 1
python test_receiver.py --interval 2
```

## Data Format

### Landmarks (21 per hand)
- UDP: JSON with nested arrays `[[x, y, z], ...]`
- LSL: Flat array of 126 floats (2 hands × 21 landmarks × 3 coords)

### Joint Angles (14 per hand)
- UDP: JSON with named angles `{"thumb_cmc_mcp": 45.2, ...}`
- LSL: Flat array of 28 floats (2 hands × 14 angles)

Angle names (in order):
1. thumb_cmc_mcp
2. thumb_ip
3. index_mcp
4. index_pip
5. index_dip
6. middle_mcp
7. middle_pip
8. middle_dip
9. ring_mcp
10. ring_pip
11. ring_dip
12. pinky_mcp
13. pinky_pip
14. pinky_dip

## Network Configuration

Default ports:
- UDP Landmarks: 5005
- UDP Angles: 5010
- LSL Streams: Auto-discovery by name
  - `StereoHandTracker_Landmarks`
  - `StereoHandTracker_Angles`

## Troubleshooting

### No LSL streams found
- Make sure `pylsl` is installed: `pip install pylsl`
- Check that sender/GUI is running first
- Try the self-test: `python test_lsl_streaming.py --selftest`

### UDP not receiving
- Check firewall settings
- Verify ports are not in use: `netstat -an | grep 5005`
- Make sure sender and receiver are on same machine (or adjust IP)

### Low frame rate
- Check CPU usage
- Reduce send rate: `python test_sender.py --rate 15`
- Close other applications

## Integration with Unity

The data format matches Unity's expected input. Use the receiver as a reference for:
- Parsing UDP JSON packets
- Reading LSL streams with correct channel order
- Extracting landmark positions and joint angles
