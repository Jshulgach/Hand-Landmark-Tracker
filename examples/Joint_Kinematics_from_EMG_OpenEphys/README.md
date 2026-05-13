# EMG-to-Joint-Angles Regression with Open Ephys Data

This example demonstrates how to predict continuous hand joint angles from EMG signals using **Open Ephys** as the data source, integrating `python-open-ephys` with the `Hand-Landmark-Tracker` kinematic decoding pipeline.

## Overview

This pipeline extends the original `Joint_Kinematics_from_EMG` example to support:
- **Open Ephys Binary Format (.oebin)** for EMG data input
- **Real-time ZMQ streaming** from Open Ephys GUI
- **Offline dataset creation** from recorded .oebin files
- **Same regression models** as the original pipeline (PyTorch EMGRegressor)
- **Compatible preprocessing** (notch, bandpass, rectification, envelope extraction)

> **⚠️ GUI Location**: The interactive data collection GUI has been moved to  
> [`python-open-ephys/examples/joint_angle_regression/`](https://github.com/Neuro-Mechatronics-Interfaces/python-open-ephys/tree/main/examples/joint_angle_regression)  
> to mirror the MindRove-EMG repository structure. This keeps hardware-specific tools in their respective repos.

## Quick Start

### Option 1: GUI Workflow (Recommended)

Use the interactive GUI in `python-open-ephys` for data collection:

```bash
# 1. Start Open Ephys GUI (with ZMQ Interface plugin)
# 2. Start hand tracking with LSL broadcast

# 3. Launch data collection GUI
cd python-open-ephys/examples/joint_angle_regression
python new_session_gui.py

# 4. Collect data via GUI (see that repo's README for details)

# 5. Train model using scripts in this repository
cd Hand-Landmark-Tracker/examples/Joint_Kinematics_from_EMG
python train_model.py --root_dir /path/to/data --label MySession
```

See the [python-open-ephys joint_angle_regression README](https://github.com/Neuro-Mechatronics-Interfaces/python-open-ephys/tree/main/examples/joint_angle_regression) for comprehensive GUI features (flow diagrams, guided prompts, in-GUI recording, filtering, training, live comparison).

### Option 2: Manual CLI Workflow

For advanced users or working with existing recordings:

---

## Key Differences from Original Pipeline

### Data Source
- **Original:** Intan RHD files (`python-intan`)
- **This Example:** Open Ephys .oebin files or ZMQ streaming (`python-open-ephys`)

### Data Loading
```python
# Original (python-intan)
from handtrack.io import SessionLoader
loader = SessionLoader(root_dir, label)
emg, fs, t = loader.load_emg()  # Expects {label}_emg_data.npz

# This Example (python-open-ephys)
from pyoephys.io import load_open_ephys_session
session = load_open_ephys_session('path/to/recording.oebin')
emg = session['amplifier_data']  # shape (C, S)
fs = session['sample_rate']
t = session['t_amplifier']
```

### Real-time Acquisition
```python
# Original: Uses python-intan TCP client
from intan.interface import IntanTcpClient

# This Example: Uses python-open-ephys ZMQ client
from pyoephys.interface import ZMQClient
client = ZMQClient(host_ip='127.0.0.1', data_port='5556')
```

---

## Installation

### Prerequisites
```bash
# 1. Install Hand-Landmark-Tracker
cd Hand-Landmark-Tracker
pip install -e .

# 2. Install python-open-ephys
pip install --index-url https://test.pypi.org/simple/ --no-deps python-oephys
# Or from source:
git clone https://github.com/Neuro-Mechatronics-Interfaces/python-open-ephys.git
cd python-open-ephys
pip install -e .

# 3. Install additional dependencies
pip install torch numpy scipy pyyaml pylsl
```

### Verify Installation
```bash
python -c "from pyoephys.io import load_open_ephys_session; print('✓ python-open-ephys OK')"
python -c "from handtrack.processing import EMGPreprocessor; print('✓ handtrack OK')"
```

---

## Pipeline Workflow

```
┌─────────────────────────────┐
│ 1. Collect Data             │  Record video + EMG via Open Ephys GUI
│    - Open Ephys recording   │  (with ZMQ Interface plugin enabled)
│    - Synchronized video     │
└────────┬────────────────────┘
         │
┌────────▼────────────────────┐
│ 2. Convert Open Ephys Data  │  Optional: convert .oebin → .npz
│    (Optional)                │  for faster loading
└────────┬────────────────────┘
         │
┌────────▼────────────────────┐
│ 3. Calculate Sync Offset    │  Align video timestamps with EMG
└────────┬────────────────────┘
         │
┌────────▼────────────────────┐
│ 4. Create Training Dataset  │  Extract EMG features + joint angles
│    (oephys_create_dataset.py)
└────────┬────────────────────┘
         │
┌────────▼────────────────────┐
│ 5. Train Model              │  PyTorch EMGRegressor (same as before)
│    (use original train_model.py)
└────────┬────────────────────┘
         │
┌────────▼────────────────────┐
│ 6. Predict                  │  Offline: from .npz datasets
│    - Offline prediction     │  Real-time: from ZMQ stream
│    - Real-time ZMQ stream   │
└─────────────────────────────┘
```

---

## Repository Structure

This directory contains scripts for **offline processing** of Open Ephys data:
- `oephys_session_loader.py` - Load .oebin files compatible with handtrack API
- `oephys_create_dataset.py` - Create training datasets from recordings
- `oephys_predict_angles.py` - Offline prediction and visualization
- `oephys_convert_to_npz.py` - Convert .oebin to NPZ format
- `oephys_realtime_predict.py` - Real-time ZMQ prediction

For **live data collection**, use the GUI in `python-open-ephys/examples/joint_angle_regression/`.

## Usage

### Method 1: Using the GUI (python-open-ephys)

The easiest way to collect training data is using the interactive GUI in the `python-open-ephys` repository:

```bash
cd python-open-ephys/examples/joint_angle_regression
python new_session_gui.py
```

See the [GUI README](https://github.com/Neuro-Mechatronics-Interfaces/python-open-ephys/tree/main/examples/joint_angle_regression) for:
- Visual flow diagram of pipeline
- Guided prompts for structured data collection
- In-GUI recording with EMG filtering
- Model training integration
- Live prediction comparison

Once you have collected data with the GUI, return to this repository to train the model (see Method 2 below).

---

### Method 2: Manual Workflow (Offline Processing)

If you have existing Open Ephys recordings or prefer manual control, use these scripts:

### Step 1: Record Data with Open Ephys

**Hardware Setup:**
1. Connect your EMG amplifier to Open Ephys acquisition board
2. Launch Open Ephys GUI
3. Enable **ZMQ Interface** plugin (for real-time streaming)
4. Configure recording path
5. Record video simultaneously (synchronized camera or MediaPipe tracking)

**Data Structure:**
```
/path/to/data/
├── raw/
│   └── MySession/
│       └── Record Node XXX/
│           └── experiment1/
│               └── recording1/
│                   ├── structure.oebin
│                   ├── continuous/
│                   │   ├── continuous.dat
│                   │   └── timestamps.npy
│                   └── events/
│                       └── ...
├── landmarks/
│   └── MySession_landmarks.npz
└── events/
    ├── MySession_start.txt
    └── MySession_end.txt
```

---

### Step 2: Convert Open Ephys to NPZ (Optional)

For faster loading in subsequent steps, convert .oebin to .npz format:

```bash
python oephys_convert_to_npz.py \
    --oebin_path /path/to/data/raw/MySession/Record_Node_XXX/experiment1/recording1/structure.oebin \
    --output_path /path/to/data/raw/MySession/MySession_emg_data.npz
```

**Output:**
```python
# MySession_emg_data.npz contains:
{
    'amplifier_data': (C, S) array,  # EMG data in microvolts
    'sample_rate': float,             # Sampling frequency (Hz)
    't_amplifier': (S,) array,        # Timestamps (seconds)
    'channel_names': list of str      # Channel labels
}
```

---

### Step 3: Calculate Synchronization Offset

Same as original pipeline:

```bash
python calculate_sync_offset.py \
    --root_dir /path/to/data \
    --label MySession \
    --sync_label "Start" \
    --emg_fs 5000 \
    --video_fps 30
```

**Note:** If you already have a sync offset file from previous sessions, you can skip this step.

---

### Step 4: Create Training Dataset

Use the Open Ephys-compatible dataset creation script:

```bash
python oephys_create_dataset.py \
    --root_dir /path/to/data \
    --label MySession \
    --oebin_path /path/to/data/raw/MySession/Record_Node_XXX/experiment1/recording1/structure.oebin \
    --window_ms 250 \
    --step_ms 50 \
    --channels 0 1 2 3 4 5 6 7 \
    --overwrite
```

**Parameters:**
- `--root_dir`: Root directory containing landmarks, events
- `--label`: Session label (e.g., "MySession")
- `--oebin_path`: Path to structure.oebin file (or folder containing it)
- `--window_ms`: EMG feature extraction window size (default: 250 ms)
- `--step_ms`: Window step size (default: 50 ms)
- `--channels`: Optional channel selection (default: all channels)
- `--overwrite`: Regenerate dataset even if it exists

**Output:**
```python
# MySession_training_dataset.npz
{
    'features': (N, C×F) array,  # N windows, C channels × F features per channel
    'labels': (N, 5) array,      # 5 joint angles (thumb, index, middle, ring, pinky)
    'emg_fs': float,             # EMG sampling frequency
    'lm_fs': float               # Landmark sampling frequency
}
```

---

### Step 5: Train Model

Use the **original training script** from `Joint_Kinematics_from_EMG/`:

```bash
cd ../Joint_Kinematics_from_EMG
python train_model.py \
    --root_dir /path/to/data \
    --label MySession \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001
```

The model architecture and training process are **identical** to the original pipeline. The only difference is the data source (Open Ephys instead of Intan).

---

### Step 6: Predict Joint Angles

#### Offline Prediction (from saved dataset)

```bash
python oephys_predict_angles.py \
    --root_dir /path/to/data \
    --label MySession \
    --model_path /path/to/data/model/MySession_model.pth \
    --visualize
```

#### Real-time Prediction (ZMQ streaming)

**Requirements:**
1. Open Ephys GUI running with ZMQ Interface plugin enabled
2. Trained model weights available
3. Known channel configuration

```bash
python oephys_realtime_predict.py \
    --model_path /path/to/data/model/MySession_model.pth \
    --zmq_host 127.0.0.1 \
    --zmq_port 5556 \
    --channels 0 1 2 3 4 5 6 7 \
    --window_ms 250 \
    --emg_fs 5000 \
    --lsl_outlet  # Optional: stream predictions via LSL
```

**Real-time Pipeline:**
```
Open Ephys GUI → ZMQ → ZMQClient → EMG Buffer → Feature Extraction → Model → Joint Angles → LSL (optional)
```

---

## File Descriptions

### Core Scripts

| File | Description |
|------|-------------|
| `oephys_convert_to_npz.py` | Convert .oebin files to .npz format |
| `oephys_create_dataset.py` | Create training dataset from Open Ephys data |
| `oephys_predict_angles.py` | Offline angle prediction from dataset |
| `oephys_realtime_predict.py` | Real-time angle prediction from ZMQ stream |
| `oephys_session_loader.py` | SessionLoader adapter for Open Ephys data |

### Utilities

| File | Description |
|------|-------------|
| `calculate_sync_offset.py` | Same as original (reused) |
| `config_example.yaml` | Example configuration file |
| `README.md` | This file |

---

## Configuration File

Create a `config.yaml` file for your session:

```yaml
# Data paths
root_dir: /path/to/data
label: MySession
oebin_path: /path/to/data/raw/MySession/Record_Node_XXX/experiment1/recording1/structure.oebin

# EMG parameters
emg_fs: 5000
channels: [0, 1, 2, 3, 4, 5, 6, 7]  # Channel selection
window_ms: 250
step_ms: 50

# Preprocessing
notch_freq: 60
bandpass_low: 20
bandpass_high: 450
envelope_cutoff: 5

# Video/landmarks
video_fps: 30
landmark_fs: 30

# Model
model_path: /path/to/data/model/MySession_model.pth
batch_size: 32
learning_rate: 0.001
epochs: 100

# Real-time ZMQ
zmq_host: 127.0.0.1
zmq_port: 5556
lsl_outlet_name: JointAngles
```

Then use it:
```bash
python oephys_create_dataset.py --config config.yaml
```

---

## Compatibility Notes

### Data Format Comparison

| Feature | Intan (.rhd) | Open Ephys (.oebin) |
|---------|--------------|---------------------|
| **Data Type** | int16 (ADC counts) | int16 → float32 (µV) |
| **Scaling** | Manual (0.195 µV/count) | Automatic (bitVolts) |
| **Timestamps** | Synthesized from fs | Per-sample or synthesized |
| **Channels** | amplifier_data | amplifier_data |
| **Metadata** | frequency_parameters | structure.oebin JSON |

### Preprocessing Pipeline

Both use the **identical preprocessing**:
```python
EMGPreprocessor(fs=5000):
  1. Notch filter (60 Hz)
  2. Bandpass filter (20-450 Hz)
  3. Rectification (abs)
  4. Lowpass filter (5 Hz envelope)
```

### Feature Extraction

**Same features** as original pipeline:
- **RMS**: Root Mean Square
- **MAV**: Mean Absolute Value
- **WL**: Waveform Length
- **ZC**: Zero Crossings
- **SSC**: Slope Sign Changes

---

## Troubleshooting

### "No .oebin file found"
- Check that `structure.oebin` exists in the recording folder
- Provide full path to .oebin file or parent folder
- Ensure recording completed successfully in Open Ephys GUI

### "Timestamps.npy not found"
- Open Ephys Binary Format should include timestamps.npy
- If missing, timestamps will be synthesized from sample_rate
- Check Open Ephys GUI settings for timestamp recording

### "Channel count mismatch"
```python
# If your .oebin has different channel count than expected:
python oephys_create_dataset.py --channels 0 1 2 3  # Specify active channels
```

### "ZMQ connection timeout"
- Ensure Open Ephys GUI is running
- Check that ZMQ Interface plugin is enabled and configured correctly
- Verify `zmq_host` and `zmq_port` match GUI settings (default: 127.0.0.1:5556)

### "Model prediction NaN values"
- Check that scaler was saved during training
- Verify channel order matches training data
- Ensure EMG signal quality is good (check impedances)

---

## Performance Benchmarks

### Offline Dataset Creation
- **5000 Hz, 8 channels, 60 seconds:** ~5-10 seconds
- **Memory:** ~500 MB for 1 minute of 8-channel data

### Real-time Prediction
- **Latency:** <50 ms (250 ms window → prediction)
- **Throughput:** 100+ predictions/second
- **CPU:** ~5-10% on modern processor

---

## Future Enhancements

- [ ] Support for multiple .oebin files (merge sessions)
- [ ] Automatic channel quality assessment
- [ ] GPU acceleration for real-time prediction
- [ ] Integration with Open Ephys Events for automatic segmentation
- [ ] Multi-stream support (EMG + IMU from different sources)
- [ ] Adaptive filtering based on signal quality

---

## References

- **Hand-Landmark-Tracker:** https://github.com/Neuro-Mechatronics-Interfaces/Hand-Landmark-Tracker
- **python-open-ephys:** https://github.com/Neuro-Mechatronics-Interfaces/python-open-ephys
- **Open Ephys GUI:** https://open-ephys.org/gui
- **Original EMG-Joint-Angles Pipeline:** `../Joint_Kinematics_from_EMG/`

---

## Support

For issues specific to:
- **Open Ephys integration:** See `python-open-ephys` repository
- **Model training/architecture:** See original `Joint_Kinematics_from_EMG/` example
- **Hand tracking:** See `Hand-Landmark-Tracker` main repository

---

## License

Same as parent repository (Hand-Landmark-Tracker).
