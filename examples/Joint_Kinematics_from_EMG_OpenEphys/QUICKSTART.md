# Quick Start Guide: Open Ephys EMG-to-Joint-Angles

## Interactive GUI Workflow (Recommended)

**Fastest way to collect data and train models:**

### Data Collection (python-open-ephys GUI)

The data collection GUI lives in the `python-open-ephys` repository for easier access:

1. **Install:**
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --no-deps python-oephys
   pip install torch numpy scipy pyyaml pylsl PyQt5
   ```

2. **Start Open Ephys GUI:**
   - Enable ZMQ Interface plugin
   - Configure 8 EMG channels at 5000 Hz
   - Start acquisition

3. **Start Hand Tracking:**
   ```bash
   cd Hand-Landmark-Tracker/stereo_hand_tracking
   python stereo_handtrack.py --lsl-broadcast
   ```

4. **Launch Session GUI:**
   ```bash
   cd python-open-ephys/examples/joint_angle_regression
   python new_session_gui.py
   # or on Windows:
   run_gui.bat
   ```

5. **Collect Data:**
   - Click "Connect" (EMG panel)
   - Click "Connect LSL" (hand tracking panel)
   - Enter Subject ID, Session ID
   - Click "Start Recording"
   - Perform hand movements for 2-5 minutes
   - Click "Stop & Save"

### Model Training (this repository)

After collecting data with the GUI, use these scripts to train the regression model:

6. **Train:**
   ```bash
   cd Hand-Landmark-Tracker/examples/Joint_Kinematics_from_EMG_OpenEphys
   
   # If you used the GUI, data is already in NPZ format
   # Otherwise, create dataset from .oebin files:
   python oephys_create_dataset.py --root_dir /path/to/data --label MySession
   
   # Train the model
   cd ../Joint_Kinematics_from_EMG
   python train_model.py --root_dir /path/to/data --label MySession
   ```

**Done!** Your trained model is ready for real-time prediction.

> **Note**: The GUI has been moved to `python-open-ephys/examples/joint_angle_regression/` 
> to mirror the MindRove-EMG structure and keep hardware-specific tools in their respective repositories.

---

## Manual CLI Workflow (Advanced)

## 5-Minute Setup

### Step 1: Install Dependencies
```bash
cd Hand-Landmark-Tracker
pip install -e .
pip install --index-url https://test.pypi.org/simple/ --no-deps python-oephys
pip install torch numpy scipy pyyaml pylsl
```

### Step 2: Prepare Your Data
Organize your data as:
```
/path/to/data/
├── raw/MySession/        # Open Ephys recording here
├── landmarks/            # Put MySession_landmarks.npz here
└── events/               # Put MySession_start.txt, MySession_end.txt here
```

### Step 3: Create Training Dataset
```bash
cd examples/Joint_Kinematics_from_EMG_OpenEphys
python oephys_create_dataset.py --root_dir /path/to/data --label MySession --verbose
```

### Step 4: Train Model
```bash
cd ../Joint_Kinematics_from_EMG
python train_model.py --root_dir /path/to/data --label MySession --epochs 100
```

### Step 5: Predict
```bash
cd ../Joint_Kinematics_from_EMG_OpenEphys
python oephys_predict_angles.py \
    --root_dir /path/to/data \
    --label MySession \
    --model_path /path/to/data/model/MySession_model.pth \
    --visualize
```

---

## Real-Time Prediction

### Prerequisites
1. Open Ephys GUI running
2. ZMQ Interface plugin enabled
3. Trained model available

### Run Real-Time Prediction
```bash
python oephys_realtime_predict.py \
    --model_path /path/to/model.pth \
    --scaler_path /path/to/scaler.pkl \
    --zmq_host 127.0.0.1 \
    --zmq_port 5556 \
    --lsl_outlet
```

---

## Common Issues

### "No .oebin file found"
**Solution:** Provide explicit path:
```bash
python oephys_create_dataset.py \
    --root_dir /path/to/data \
    --label MySession \
    --oebin_path /path/to/recording/structure.oebin
```

### "Scaler file not found"
**Solution:** Specify scaler path explicitly:
```bash
python oephys_predict_angles.py \
    --model_path /path/to/model.pth \
    --scaler_path /path/to/scaler.pkl \
    ...
```

### "ZMQ connection timeout"
**Solution:** 
1. Check Open Ephys GUI is running
2. Verify ZMQ Interface plugin is enabled
3. Confirm port number (default: 5556)

---

## File Checklist

Before running the pipeline, ensure you have:

- [ ] Open Ephys recording (`.oebin` + `continuous.dat`)
- [ ] Video landmarks (`{label}_landmarks.npz`)
- [ ] Event markers (`{label}_start.txt`, `{label}_end.txt`)
- [ ] Sync offset calculated (`{label}_sync_offset.txt`)

---

## Tips

### Speed up loading
Convert `.oebin` to `.npz` once:
```bash
python oephys_convert_to_npz.py \
    --oebin_path /path/to/structure.oebin \
    --output_path /path/to/MySession_emg_data.npz
```

### Use config file
Create `config.yaml` and use:
```bash
python oephys_create_dataset.py --config config.yaml
```

### Merge multiple sessions
```bash
python oephys_create_dataset.py \
    --root_dir /path/to/data \
    --merge_labels Session1 Session2 Session3 \
    --output_file combined_dataset.npz
```

---

## Next Steps

1. **Experiment with parameters:** Adjust `window_ms`, `step_ms`, `channels`
2. **Try different models:** Modify architecture in `train_model.py`
3. **Real-time applications:** Use LSL output for robot control, visualization, etc.
4. **Multi-modal fusion:** Combine with IMU data (see MindRove-EMG examples)

---

## Getting Help

- **Documentation:** See [README.md](README.md)
- **Original pipeline:** See `../Joint_Kinematics_from_EMG/`
- **Open Ephys issues:** https://github.com/Neuro-Mechatronics-Interfaces/python-open-ephys
- **Hand tracking issues:** https://github.com/Neuro-Mechatronics-Interfaces/Hand-Landmark-Tracker
