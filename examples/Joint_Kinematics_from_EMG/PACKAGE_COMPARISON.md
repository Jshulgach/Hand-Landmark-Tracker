# Joint Kinematics from EMG - Package Comparison & Recommendations

## Current State

### Hand-Landmark-Tracker: `Joint_Kinematics_from_EMG/`
**Purpose:** EMG → Joint Angles/Landmarks (Regression)
- Predicts **continuous joint angles** (5 fingers) from EMG signals
- Uses **PyTorch regression models** (EMGRegressor)
- **Regression task:** EMG features → 63D landmark positions or 5D joint angles
- Synchronizes video landmarks with EMG data
- Creates training datasets from paired EMG + video
- Real-time LSL prediction: `lsl_emg_predict_angles.py`

**Key Scripts:**
1. `create_feature_dataset.py` - Build EMG+landmark training data
2. `train_model.py` - Train PyTorch regression model
3. `example_predict_angles.py` - Predict angles from trained model
4. `calculate_sync_offset.py` - Sync EMG timestamps with video
5. `lsl_emg_predict_angles.py` - Real-time LSL streaming prediction

**Outputs:** Continuous angle predictions (radians) for thumb, index, middle, ring, pinky

---

### python-intan: `gesture_classifier/`
**Purpose:** EMG → Gesture Labels (Classification)
- Classifies **discrete gestures** (pinch, grasp, rest, etc.)
- Uses **scikit-learn classifiers** (LDA, SVM, Random Forest, etc.)
- **Classification task:** EMG features → gesture class (0, 1, 2, ...)
- Supports multiple data formats (RHD, CSV, NPZ)
- Production-ready with TensorFlow/Keras models
- Real-time device streaming: `3d_predict_from_device_realtime.py`

**Key Scripts:**
1. `1a_build_training_dataset_rhd.py` - Build from Intan RHD files
2. `1e_build_training_dataset_any.py` - Build from any format (CSV, NPZ, RHD)
3. `2_train_model.py` - Train classification model (LDA, RF, etc.)
4. `3d_predict_from_device_realtime.py` - Real-time classification from Intan device
5. Multiple orientation handling and grid permutation support

**Outputs:** Discrete gesture labels (e.g., "pinch", "grasp", "rest")

---

## Key Differences

| Feature | Hand-Landmark-Tracker | python-intan |
|---------|---------------------|--------------|
| **Task Type** | Regression (continuous angles) | Classification (discrete gestures) |
| **Framework** | PyTorch | scikit-learn + TensorFlow/Keras |
| **Input Data** | Video + EMG (synchronized) | EMG only (+ optional IMU) |
| **Output** | 5 joint angles (radians) | Gesture class labels |
| **Preprocessing** | Custom handtrack filters | intan.processing.EMGPreprocessor |
| **Features** | RMS, MAV, WL, ZC, SSC | Configurable via FEATURE_REGISTRY |
| **Use Case** | Prosthetic control, continuous motion | Gesture recognition, discrete commands |
| **Device Support** | Generic EMG systems | Intan RHD/RHS devices (native support) |
| **Real-time** | LSL streaming | Intan device streaming + LSL |
| **Maturity** | Research prototype | Production-ready |

---

## Recommendations

### ✅ Keep Both Examples in Separate Repos
**Rationale:**
1. **Different problem domains:**
   - Hand-Landmark-Tracker → **continuous motion tracking** (regression)
   - python-intan → **discrete gesture recognition** (classification)

2. **Different workflows:**
   - Hand-Landmark-Tracker requires **video + EMG synchronization**
   - python-intan works with **EMG-only** data from Intan hardware

3. **Different dependencies:**
   - Hand-Landmark-Tracker: MediaPipe, OpenCV, video processing
   - python-intan: Intan device drivers, TCP streaming

### 🔄 Suggested Improvements

#### For Hand-Landmark-Tracker:
```python
# Add README to Joint_Kinematics_from_EMG/
# Create clear workflow documentation
# Add argparse help to all scripts
# Consolidate TEST_*.py scripts (rename or remove)
# Add config file support (similar to python-intan)
```

#### For python-intan:
```python
# Add regression example for continuous angle prediction
# Could add handtrack integration example
# Document differences between classification vs regression
```

### 📋 Cross-Pollination Opportunities

**1. Preprocessing Alignment:**
```python
# Hand-Landmark-Tracker could adopt intan's EMGPreprocessor
from intan.processing import EMGPreprocessor

preprocessor = EMGPreprocessor(
    notch_freq=60.0,
    notch_q=30.0,
    bandpass_low=20.0,
    bandpass_high=450.0,
    rectify=True,
    lowpass_cutoff=5.0
)
```

**2. Feature Extraction:**
```python
# python-intan could use handtrack's feature functions
from handtrack.processing import extract_features

features = extract_features(
    emg_windowed,
    feature_list=['rms', 'mav', 'wl', 'zc', 'ssc']
)
```

**3. LSL Integration:**
Both packages could share LSL utilities:
```python
# Unified LSL helper library
from nml_common.lsl import resolve_emg_stream, create_landmark_outlet
```

**4. Config File Format:**
Hand-Landmark-Tracker should adopt python-intan's config.txt format:
```ini
# config.txt for Joint_Kinematics_from_EMG
root_directory=/path/to/data
label=HandDynamic
emg_fs=5000
video_fps=30
window_ms=250
step_ms=50
feature_list=rms,mav,wl,zc,ssc
model_filename=emg_regressor.pth
```

---

## Organization Recommendations

### Option 1: Keep As-Is (Recommended)
```
Hand-Landmark-Tracker/
└── examples/
    └── Joint_Kinematics_from_EMG/  ← Keep here (regression)
        └── README.md               ← Add comprehensive docs

python-intan/
└── examples/
    └── gesture_classifier/         ← Keep here (classification)
        └── README.md               ← Already good!
```

**Pros:**
- Clear separation of concerns
- No confusion between regression vs classification
- Each repo maintains focus on its primary use case

---

### Option 2: Create Shared EMG-ML Package (Future Work)
```
nml-emg-ml/                        ← New unified package
├── classification/                 ← From python-intan
│   ├── build_dataset.py
│   ├── train_classifier.py
│   └── predict_gestures.py
├── regression/                     ← From Hand-Landmark-Tracker
│   ├── build_dataset.py
│   ├── train_regressor.py
│   └── predict_angles.py
└── shared/
    ├── preprocessing.py
    ├── features.py
    └── lsl_utils.py
```

**Pros:**
- Unified EMG ML pipeline
- Shared preprocessing and features
- Easier to compare approaches

**Cons:**
- Requires significant refactoring
- Potential scope creep
- Dependency management complexity

---

## Immediate Action Items

### For Hand-Landmark-Tracker:

1. **Add README to Joint_Kinematics_from_EMG/**
   ```markdown
   # EMG-to-Joint-Angles Regression
   
   This directory contains scripts for training regression models
   that predict continuous joint angles from EMG signals.
   
   ## Workflow:
   1. Sync EMG + video: `calculate_sync_offset.py`
   2. Create dataset: `create_feature_dataset.py`
   3. Train model: `train_model.py`
   4. Predict: `example_predict_angles.py`
   5. Real-time: `lsl_emg_predict_angles.py`
   ```

2. **Clean up TEST_* scripts**
   - Rename to meaningful names or move to tests/
   - Add docstrings

3. **Add argparse help to all scripts**
   ```python
   parser.add_argument('--root_dir', required=True, help='Data directory')
   parser.add_argument('--label', required=True, help='Session label')
   ```

4. **Add config file support**
   ```python
   # Copy from python-intan/examples/gesture_classifier/
   from intan.io import load_config_file
   config = load_config_file('config.txt')
   ```

### For python-intan:

1. **Add note in gesture_classifier/README.md:**
   ```markdown
   ## Related Examples
   
   For **continuous joint angle prediction** (regression), see:
   - Hand-Landmark-Tracker/examples/Joint_Kinematics_from_EMG/
   
   This example focuses on **discrete gesture classification**.
   ```

2. **Consider adding regression example:**
   ```
   examples/
   ├── gesture_classifier/      ← Discrete classification
   └── angle_regressor/         ← NEW: Continuous regression
       ├── 1_build_dataset.py
       ├── 2_train_model.py
       └── 3_predict_angles.py
   ```

---

## Summary

**Decision: Keep both examples in their respective repos.**

- **Hand-Landmark-Tracker** → EMG-to-angles **regression** (continuous control)
- **python-intan** → EMG-to-gesture **classification** (discrete commands)

Both serve different use cases and should remain separate. Focus on:
1. Adding comprehensive documentation to both
2. Standardizing preprocessing approaches
3. Cross-referencing in READMEs
4. Sharing utilities where appropriate (LSL, preprocessing)

This maintains clarity while enabling future integration if needed.
