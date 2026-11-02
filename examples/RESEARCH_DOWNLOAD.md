# MAVIS research examples

This optional download contains experimental research workflows for:

- joint kinematics from EMG
- Open Ephys session conversion and realtime prediction
- learned motion smoothing with recurrent neural networks

These scripts are provided as research references, not stable public APIs. They
may require optional scientific dependencies, project-specific configuration,
and local datasets that are not distributed with HandTrack.

Install the package and research dependencies:

```bash
python -m pip install "mavis-track[ml,io,stream]"
```

Review the README and example configuration inside each directory before
running a script. Replace placeholder dataset paths with paths on your machine.
The updated EMG/RNN scripts accept `--data-dir` and use
`~/Documents/Hand-Landmark-Tracker-Research/data/` by default. Set `MAVIS_DATA_DIR`
for a shared custom root. Smoothing data defaults to its `smoothing/` subfolder;
an explicit `--data-dir` selects that example's dataset folder directly.
Video examples accept `--video` or `HANDTRACK_VIDEO_PATH` and otherwise look for
`HandDynamic.mp4` in their selected data folder. Existing session-based scripts
keep their `--root_dir` arguments. Path help runs without the optional ML stack.
Use an installation of the updated source checkout for these new path options.
No participant data, recordings, trained models, OptiTrack SDK files, or other
vendor binaries are included in this download.
