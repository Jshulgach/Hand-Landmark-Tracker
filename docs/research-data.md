# External research data

The maintained checkout contains source, tests, docs and curated reference
material. Local datasets, model weights and dormant scripts live separately.
The move preserved all 172 inventoried files byte-for-byte, including the full
build/validation snapshot. The working Python environment and runtime camera
calibration locations were retained.

Default workspace:

```text
~/Documents/Hand-Landmark-Tracker-Research/
  data/                         EMG datasets, trained weights, scalers and notes
    smoothing/                  RNN datasets and model
      figures/                  Saved trajectory figures
  archive/                      Local legacy scripts and notes
  validation/build-2026-11-01/  Preserved build artifacts, logs and environment records
  relocation-manifest.json     Source/destination paths, sizes and SHA-256 hashes
```

The root `data/`, `archive/` and `build/` folders have been cleared from the
checkout. Packaging tools can recreate ignored `build/`; it is disposable after
retaining any outputs you need. The tracked `research/camera-placement/` reports
remain curated historical reference material.

## Select a data directory

Updated EMG and smoothing examples accept `--data-dir`. An explicit directory
wins over `MAVIS_DATA_DIR`, which otherwise selects the shared external data root.
Without either, the default is the workspace's `data/` folder shown above.
Smoothing examples use its `smoothing/` subfolder by default; an explicit
`--data-dir` points directly to the desired smoothing dataset folder.

```powershell
python examples/Joint_Kinematics_from_EMG/create_joint_angle_dataset.py --help
python examples/Joint_Kinematics_from_EMG/create_joint_angle_dataset.py --data-dir "C:/my-study/data"
python examples/Smoothing_Motion_with_RNN/3_predict.py --data-dir "C:/my-study/smoothing"
```

To configure the shared root in a Windows Command Prompt session:

```cmd
set "MAVIS_DATA_DIR=%USERPROFILE%\Documents\Hand-Landmark-Tracker-Research\data"
```

For Python applications:

```python
from mavis_track.research import research_data_dir

emg_file = research_data_dir() / "emg_data.npz"
smoothing_model = research_data_dir(collection="smoothing") / "gru_smoother.pth"
```

Examples that process video accept `--video`. `HANDTRACK_VIDEO_PATH` remains
supported; otherwise they look for `HandDynamic.mp4` in their selected data
folder. Videos on an existing shared drive can stay there:

```powershell
python examples/Joint_Kinematics_from_EMG/visualize_predictions.py --data-dir "C:/my-study/data" --video "G:/recordings/HandDynamic.mp4"
python examples/Smoothing_Motion_with_RNN/1_feature_extraction.py --video "G:/recordings/HandDynamic.mp4"
```

Session-oriented examples that already accept `--root_dir` retain that interface;
point it at the external session folder. Updated research scripts expose their
path help without loading data or optional ML packages. Running training or
prediction still requires their optional scientific/ML dependencies, preferably
in a dedicated research environment. Relocation does not retrain models or
validate the historical research algorithms.
