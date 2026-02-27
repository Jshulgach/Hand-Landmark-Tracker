# Package Comparison: Data Sources for Kinematic Decoding

This document compares the different EMG data acquisition packages available in the workspace and how they integrate with the kinematic decoding pipeline.

---

## Overview

| Package | Purpose | Data Format | Hardware | Real-time | Use Case |
|---------|---------|-------------|----------|-----------|----------|
| **python-intan** | Intan RHX systems | .rhd, .rhs, .dat | Intan amplifiers | ✓ (TCP) | General neurophysiology |
| **python-open-ephys** | Open Ephys GUI | .oebin, .npz | Any Open Ephys board | ✓ (ZMQ) | Flexible research platform |
| **MindRove-EMG** | MindRove wristband | Custom | MindRove WiFi | ✓ (WiFi) | Wearable EMG |

---

## Detailed Comparison

### python-intan

**Strengths:**
- Native support for Intan hardware (RHD, RHS formats)
- Mature, well-tested codebase
- Extensive examples for gesture classification
- TCP streaming interface for real-time acquisition
- Built-in preprocessing pipeline

**Data Format:**
```python
# Loading data
from intan.io import load_rhd_file
result = load_rhd_file('recording.rhd')
emg = result['amplifier_data']  # shape (C, S)
fs = result['frequency_parameters']['amplifier_sample_rate']
t = result['t_amplifier']
```

**Integration with Kinematic Decoder:**
- Original `Joint_Kinematics_from_EMG` example uses python-intan
- SessionLoader expects `.npz` files generated from Intan data
- File structure: `{label}_emg_data.npz` with keys: `amplifier_data`, `sample_rate`, `t_amplifier`

**Example Usage:**
```bash
# Original pipeline (python-intan)
cd examples/Joint_Kinematics_from_EMG
python create_feature_dataset.py --root_dir /path/to/data --label MySession
```

---

### python-open-ephys

**Strengths:**
- Hardware-agnostic (works with any Open Ephys compatible device)
- Direct integration with Open Ephys GUI
- ZMQ streaming for low-latency real-time applications
- Flexible data formats (.oebin, .npz)
- Active development and community support

**Data Format:**
```python
# Loading data
from pyoephys.io import load_open_ephys_session
session = load_open_ephys_session('structure.oebin')
emg = session['amplifier_data']  # shape (C, S), float32 (µV)
fs = session['sample_rate']
t = session['t_amplifier']
```

**Integration with Kinematic Decoder:**
- **NEW:** `Joint_Kinematics_from_EMG_OpenEphys` example (this directory)
- Uses `OEphysSessionLoader` adapter
- Compatible preprocessing and feature extraction
- Same model architecture (EMGRegressor)

**Example Usage:**
```bash
# Open Ephys pipeline (NEW)
cd examples/Joint_Kinematics_from_EMG_OpenEphys
python oephys_create_dataset.py \
    --root_dir /path/to/data \
    --label MySession \
    --oebin_path /path/to/recording.oebin
```

**Real-Time Streaming:**
```bash
# ZMQ streaming from Open Ephys GUI
python oephys_realtime_predict.py \
    --model_path model.pth \
    --zmq_host 127.0.0.1 \
    --zmq_port 5556
```

---

### MindRove-EMG

**Strengths:**
- Wireless wearable form factor
- WiFi streaming (no cables)
- Built-in IMU sensors
- Integrated gesture classification
- Real-time muscle activity decoding

**Data Format:**
```python
# MindRove uses custom data structures
# Typically streams via WiFi to custom receivers
# Can save to NPZ format for offline analysis
```

**Integration with Kinematic Decoder:**
- Can adapt using similar approach to Open Ephys
- Would require custom SessionLoader adapter
- IMU data could be fused with EMG for enhanced decoding

**Potential Integration (Future Work):**
```bash
# Hypothetical MindRove integration
python mindrove_create_dataset.py \
    --root_dir /path/to/data \
    --label MySession \
    --include_imu  # Use both EMG and IMU
```

---

## Data Flow Comparison

### Offline Pipeline

**python-intan:**
```
Intan Device → .rhd file → load_rhd_file() → NPZ → SessionLoader → Features → Model → Angles
```

**python-open-ephys:**
```
OE Device → .oebin → load_open_ephys_session() → NPZ → OEphysSessionLoader → Features → Model → Angles
```

**MindRove:**
```
MindRove → WiFi → Custom format → (adapter needed) → Features → Model → Angles
```

### Real-Time Pipeline

**python-intan:**
```
Intan Device → TCP stream → IntanTcpClient → Buffer → Features → Model → Angles → LSL
```

**python-open-ephys:**
```
OE GUI → ZMQ stream → ZMQClient → Buffer → Features → Model → Angles → LSL
```

**MindRove:**
```
MindRove → WiFi → Custom receiver → Buffer → Features → Model → Angles → LSL
```

---

## Preprocessing Pipeline Compatibility

All three packages can use the **same preprocessing pipeline** from `handtrack.processing.EMGPreprocessor`:

```python
EMGPreprocessor(fs=5000):
  1. Notch filter (60 Hz)          # Remove powerline interference
  2. Bandpass filter (20-450 Hz)   # EMG frequency range
  3. Rectification                 # Full-wave rectify
  4. Lowpass filter (5 Hz)         # Envelope extraction
```

**Feature Extraction** (identical across all packages):
- RMS: Root Mean Square
- MAV: Mean Absolute Value
- WL: Waveform Length
- ZC: Zero Crossings
- SSC: Slope Sign Changes

---

## When to Use Each Package

### Use python-intan when:
- You have Intan hardware (RHD2000, RHS2000)
- Working with existing .rhd/.rhs files
- Need mature, stable codebase for neurophysiology
- Following established lab protocols

### Use python-open-ephys when:
- Using Open Ephys acquisition system
- Need hardware flexibility (change amplifiers easily)
- Want direct GUI integration
- Require low-latency ZMQ streaming
- Working with multi-modal data (combine different sources)

### Use MindRove-EMG when:
- Need wireless/wearable setup
- Working in free-movement scenarios
- Want combined EMG + IMU data
- Developing prosthetic/exoskeleton control
- Need portable system

---

## Migration Guide

### From python-intan to python-open-ephys

**Step 1:** Convert data format (if needed)
```bash
# If you have .rhd files, convert to NPZ first
# Then use python-open-ephys to load
```

**Step 2:** Update SessionLoader
```python
# Before (python-intan)
from handtrack.io import SessionLoader
loader = SessionLoader(root_dir, label)

# After (python-open-ephys)
from oephys_session_loader import OEphysSessionLoader
loader = OEphysSessionLoader(root_dir, label, oebin_path)
```

**Step 3:** Rest of pipeline is identical!
```python
# Same preprocessing
emg_filtered = preprocessor.preprocess(emg)

# Same feature extraction
features = preprocessor.extract_emg_features(emg_filtered, ...)

# Same model
model = EMGRegressor(input_dim, output_dim)
```

---

## Performance Comparison

| Metric | python-intan | python-open-ephys | MindRove-EMG |
|--------|--------------|-------------------|--------------|
| **Max sampling rate** | 30 kHz | 30 kHz | 1 kHz |
| **Typical EMG rate** | 5 kHz | 5 kHz | 500-1000 Hz |
| **Real-time latency** | <10 ms (TCP) | <5 ms (ZMQ) | <20 ms (WiFi) |
| **Channel count** | 32-128 | Unlimited (board dependent) | 8-16 |
| **Power consumption** | Medium (wired) | Medium (wired) | Low (battery) |
| **Portability** | Low | Low | High |

---

## Recommended Setups

### Lab-Based Neurophysiology
**Best Choice:** python-intan or python-open-ephys
- High channel count
- Reliable wired connection
- Precise timing
- Extensive recording capabilities

### Wearable/Prosthetic Control
**Best Choice:** MindRove-EMG or python-open-ephys (with wireless board)
- Freedom of movement
- Real-time streaming
- Combined EMG+IMU
- Portable form factor

### Multi-Modal Research Platform
**Best Choice:** python-open-ephys
- Integrate EMG from Open Ephys
- Add IMU from MindRove or external source
- Synchronize video tracking
- Flexible hardware configuration

---

## Future Integrations

### Proposed: Unified Data Interface
Create a common interface that abstracts away hardware differences:

```python
# Proposed unified API
from handtrack.io import UnifiedDataLoader

loader = UnifiedDataLoader(
    source='intan',  # or 'oephys', 'mindrove'
    root_dir='/path/to/data',
    label='MySession'
)

emg, fs, t = loader.load_emg()  # Same interface for all sources
```

### Multi-Source Fusion
Combine EMG from different sources:

```python
# Future capability
loader = MultiSourceLoader([
    ('emg', 'oephys', oebin_path),
    ('imu', 'mindrove', wifi_config),
    ('landmarks', 'video', video_path)
])

X_fused = loader.create_fused_features()
```

---

## Summary

All three packages can work with the kinematic decoding pipeline:

1. **python-intan**: Original implementation ✓
2. **python-open-ephys**: New implementation (this example) ✓
3. **MindRove-EMG**: Future work (adapter needed) ○

The **preprocessing**, **feature extraction**, and **model architecture** remain identical across all sources, making it easy to switch between hardware platforms while maintaining consistent analysis.

---

## Additional Resources

- **python-intan**: https://github.com/Neuro-Mechatronics-Interfaces/python-intan
- **python-open-ephys**: https://github.com/Neuro-Mechatronics-Interfaces/python-open-ephys
- **MindRove-EMG**: https://github.com/Neuro-Mechatronics-Interfaces/MindRove-EMG
- **Hand-Landmark-Tracker**: https://github.com/Neuro-Mechatronics-Interfaces/Hand-Landmark-Tracker
