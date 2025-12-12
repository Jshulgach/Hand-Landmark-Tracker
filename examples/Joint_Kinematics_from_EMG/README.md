# EMG-to-Joint-Angles Regression

Train neural networks to predict continuous hand joint angles from EMG signals using PyTorch.

## Overview

This pipeline demonstrates **regression** (continuous prediction) from EMG to joint angles, as opposed to **classification** (discrete gestures). The goal is to predict real-time finger joint angles for prosthetic control, motion analysis, or rehabilitation applications.

**Key Features:**
- Synchronize video landmarks with EMG data
- Extract EMG features (RMS, MAV, WL, ZC, SSC)
- Train PyTorch regression models
- Predict 5 finger joint angles (thumb, index, middle, ring, pinky)
- Real-time LSL streaming prediction

---

## Pipeline Workflow

```
┌─────────────────┐
│ 1. Collect Data │  Record video + EMG simultaneously
└────────┬────────┘
         │
┌────────▼────────────────┐
│ 2. Calculate Sync Offset│  Align video timestamps with EMG
└────────┬────────────────┘
         │
┌────────▼───────────────┐
│ 3. Create Dataset      │  Extract features + label joint angles
└────────┬───────────────┘
         │
┌────────▼────────┐
│ 4. Train Model  │  PyTorch regression (EMGRegressor)
└────────┬────────┘
         │
┌────────▼────────┐
│ 5. Predict      │  Evaluate on test data or real-time LSL
└─────────────────┘
```

---

## Step-by-Step Guide

### 1. Calculate Synchronization Offset

EMG and video are recorded separately and need timestamp alignment.

```bash
python calculate_sync_offset.py \
    --root_dir /path/to/data \
    --label MySession \
    --sync_label "Start" \
    --emg_fs 5000 \
    --video_fps 30
```

**Inputs:**
- `events.csv` - Event markers with timestamps (e.g., "Start" event)
- Video metadata with FPS

**Outputs:**
- Offset value (seconds) to align EMG and video

---

### 2. Create Feature Dataset

Extract EMG features and synchronize with video-derived joint angles.

```bash
python create_feature_dataset.py \
    --root_dir /path/to/data \
    --label MySession
```

**Inputs:**
- `emg_data.npz` or `.rhd` files - Raw EMG signals
- `video_name_landmarks.npz` - Hand landmarks from video (see `02_data_extraction/`)
- Sync offset from step 1

**Processing:**
1. Load EMG data (5000 Hz typical)
2. Apply filters: notch (60 Hz), bandpass (20-450 Hz), rectify, lowpass (5 Hz)
3. Window EMG (e.g., 250 ms windows, 50 ms step)
4. Extract features: RMS, MAV, WL, ZC, SSC per channel
5. Resample landmarks to match EMG windows
6. Compute joint angles from landmarks

**Outputs:**
- `MySession_feature_dataset.npz`:
  - `emg_features`: (N, num_channels × num_features) array
  - `landmark_labels`: (N, 63) or (N, 5) array (joint angles)
  - `emg_fs`, `lm_fs`: Sampling rates

---

### 3. Train Regression Model

Train a PyTorch neural network to map EMG features → joint angles.

```bash
python train_model.py \
    --root_dir /path/to/data \
    --label MySession
```

**Model Architecture (EMGRegressor):**
```
Input (EMG features) → FC(256) → ReLU → Dropout(0.3)
                     → FC(128) → ReLU → Dropout(0.3)
                     → FC(64)  → ReLU
                     → Output (5 joint angles)
```

**Training:**
- Loss: MSE (Mean Squared Error)
- Optimizer: Adam (lr=1e-3)
- Early stopping: 10 epochs patience
- Validation every 20 epochs

**Outputs:**
- `model/emg_regressor.pth` - Trained PyTorch model
- `model/scaler.pkl` - StandardScaler for feature normalization
- Training curves (loss vs epoch)

---

### 4. Predict Joint Angles

Evaluate the trained model on test data.

```bash
python example_predict_angles.py \
    --root_dir /path/to/data \
    --label MySession \
    --train  # Optional: train fresh model
```

**Metrics:**
- R² score per joint
- Mean Absolute Error (MAE) per joint
- Visualization: ground truth vs predicted angles

**Outputs:**
- Console: evaluation metrics
- Plot: 5 subplots (one per finger) showing predictions vs truth

---

### 5. Real-Time LSL Prediction

Stream EMG from an LSL source and predict joint angles in real-time.

```bash
python lsl_emg_predict_angles.py
```

**Workflow:**
1. Discover LSL EMG stream (e.g., from Intan device)
2. Load trained model and scaler
3. Buffer EMG samples in sliding windows
4. Extract features from each window
5. Predict joint angles
6. (Optional) Stream predictions to another LSL outlet

**Requirements:**
- `pylsl` installed
- LSL EMG stream available on network
- Trained model at `model/emg_regressor.pth`

---

## File Descriptions

### Main Pipeline Scripts

| Script | Purpose |
|--------|---------|
| `calculate_sync_offset.py` | Align EMG and video timestamps |
| `create_feature_dataset.py` | Build training dataset (features + labels) |
| `train_model.py` | Train PyTorch regression model |
| `example_predict_angles.py` | Evaluate model on test set |
| `lsl_emg_predict_angles.py` | Real-time streaming prediction |

### Utility Scripts

| Script | Purpose |
|--------|---------|
| `create_feature_dataset2.py` | Alternative feature extraction |
| `create_joint_angle_dataset.py` | Create angle-only dataset |
| `visualize_joint_angles.py` | Plot angle trajectories |
| `visualize_predictions.py` | Compare predictions vs ground truth |

### Batch Processing

| Script | Purpose |
|--------|---------|
| `predict_joint_angles_full_dataset.py` | Batch prediction on multiple files |
| `predict_landmarks_full_dataset.py` | Predict landmarks instead of angles |
| `example_create_dataset_merged.py` | Merge multiple session datasets |

### Testing/Development

| Script | Purpose |
|--------|---------|
| `test.py` | Visualization test |
| `test_predict_angles_from_video.py` | Test video-based prediction |
| `TEST_*.py` | Development/debugging scripts |

---

## Data Format

### Input: EMG Data
```python
# emg_data.npz
{
    'amplifier_data': (num_channels, num_samples),  # e.g., (64, 250000)
    'frequency_parameters': {'amplifier_sample_rate': 5000.0},
    't_amplifier': (num_samples,)  # timestamps in seconds
}
```

### Input: Landmarks
```python
# video_landmarks.npz (from extract_hand_landmarks.py)
{
    'landmarks': (num_frames, 21, 3),  # 21 landmarks × (x, y, z)
    'sampling_rate': 30.0,              # FPS
    'time_vector': (num_frames,)        # timestamps
}
```

### Output: Feature Dataset
```python
# MySession_feature_dataset.npz
{
    'emg_features': (N, num_features),  # e.g., (1000, 320) for 64ch × 5 features
    'landmark_labels': (N, 5),          # Joint angles in radians
    'emg_fs': 5000.0,
    'lm_fs': 20.0                       # Effective rate after windowing
}
```

### Output: Model
```python
# model/emg_regressor.pth
torch.nn.Module state_dict

# model/scaler.pkl
sklearn.preprocessing.StandardScaler
```

---

## Feature Extraction

**EMG Preprocessing:**
1. Notch filter (60 Hz, Q=30) - Remove powerline interference
2. Bandpass (20-450 Hz) - Isolate EMG frequency range
3. Rectification - Full-wave rectify
4. Lowpass (5 Hz) - Smooth envelope

**Windowing:**
- Window size: 250 ms (typical)
- Step size: 50 ms (80% overlap)
- Effective rate: 20 Hz output

**Features (per channel):**
- **RMS**: Root Mean Square (signal power)
- **MAV**: Mean Absolute Value (average amplitude)
- **WL**: Waveform Length (complexity)
- **ZC**: Zero Crossings (frequency indicator)
- **SSC**: Slope Sign Changes (texture)

**Total features:** `num_channels × 5`
- Example: 64 channels → 320 features per window

---

## Joint Angle Computation

From hand landmarks, compute flexion angles for each finger:

```
Angle = acos(dot(v1, v2) / (||v1|| × ||v2||))
```

Where vectors are defined by landmark positions:
- **Thumb**: THUMB_MCP → THUMB_IP → THUMB_TIP
- **Index**: INDEX_MCP → INDEX_PIP → INDEX_TIP
- **Middle**: MIDDLE_MCP → MIDDLE_PIP → MIDDLE_TIP
- **Ring**: RING_MCP → RING_PIP → RING_TIP
- **Pinky**: PINKY_MCP → PINKY_PIP → PINKY_TIP

Angles are in **radians**: 0 = straight, π = fully flexed

---

## Tips & Best Practices

**Data Collection:**
- Record EMG and video simultaneously
- Use clear "Start" event marker for synchronization
- Ensure stable EMG electrode placement
- Record diverse hand motions (open, close, individual fingers)

**Feature Tuning:**
- Adjust window_ms (100-500 ms) based on motion speed
- Smaller windows = more responsive, but noisier
- Larger windows = smoother, but delayed

**Model Training:**
- More training data = better generalization
- Augment with different hand positions/orientations
- Use cross-validation for robust evaluation

**Real-Time Performance:**
- Optimize feature extraction (vectorize operations)
- Use GPU inference for larger models
- Consider quantization for edge deployment

---

## Comparison: Regression vs Classification

This example performs **regression** (continuous angles). For **discrete gesture classification**, see:
- [python-intan/examples/gesture_classifier/](../../python-intan/examples/gesture_classifier/)

| Aspect | This Example (Regression) | gesture_classifier (Classification) |
|--------|-------------------------|-----------------------------------|
| Output | Continuous angles (radians) | Discrete labels (pinch, grasp, etc.) |
| Model | PyTorch neural network | scikit-learn (LDA, SVM, RF) |
| Use Case | Proportional control | Command recognition |
| Training | Requires video + EMG | EMG only |

See [`PACKAGE_COMPARISON.md`](./PACKAGE_COMPARISON.md) for detailed analysis.

---

## Requirements

```bash
pip install numpy scipy torch matplotlib tqdm
pip install pylsl  # For real-time streaming
```

From handtrack package:
```python
from handtrack.processing import extract_features, notch_filter, bandpass_filter
from handtrack.ml import EMGRegressor
```

---

## Troubleshooting

**ImportError: No module named 'handtrack'**
```bash
cd Hand-Landmark-Tracker
pip install -e .
```

**Sync offset calculation fails**
- Check that `events.csv` exists with "Start" marker
- Verify EMG and video timestamps overlap

**Model predicts zeros**
- Check feature normalization (use StandardScaler)
- Verify training loss decreased
- Ensure sufficient training data (>1000 samples)

**Poor prediction accuracy**
- Collect more diverse training data
- Increase model capacity (more layers/units)
- Tune hyperparameters (learning rate, epochs)
- Check for overfitting (compare train vs val loss)

**Real-time lag**
- Reduce window_ms
- Optimize feature computation
- Use compiled model (TorchScript)

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{hand_landmark_tracker_emg,
  author = {Shulgach, Jonathan},
  title = {EMG-to-Joint-Angles Regression Pipeline},
  year = {2025},
  url = {https://github.com/Jshulgach/Hand-Landmark-Tracker}
}
```

---

## Related Examples

- [02_data_extraction/](../02_data_extraction/) - Extract hand landmarks from video
- [03_streaming/](../03_streaming/) - LSL streaming for hand landmarks
- [python-intan gesture_classifier](../../python-intan/examples/gesture_classifier/) - Discrete gesture classification
