# HandTrack research examples

This optional download contains experimental research workflows for:

- joint kinematics from EMG
- Open Ephys session conversion and realtime prediction
- learned motion smoothing with recurrent neural networks

These scripts are provided as research references, not stable public APIs. They
may require optional scientific dependencies, project-specific configuration,
and local datasets that are not distributed with HandTrack.

Install the package and research dependencies first:

```bash
python -m pip install "handtrack[ml,io,stream]"
```

Review the README and example configuration inside each directory before
running a script. Replace placeholder dataset paths with paths on your machine.
Scripts that need a source recording accept the `HANDTRACK_VIDEO_PATH`
environment variable and otherwise look for `data/HandDynamic.mp4`.
No participant data, recordings, trained models, OptiTrack SDK files, or other
vendor binaries are included in this download.
