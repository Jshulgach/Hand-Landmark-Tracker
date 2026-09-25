# HandTrack core examples

This download contains the practical examples intended for most users:

- basic webcam, OptiTrack, Unity, and passive-marker workflows
- landmark and feature extraction
- LSL and realtime streaming
- robotics and virtual-hand demonstrations

Install HandTrack before running an example:

```bash
python -m venv .venv
python -m pip install handtrack
```

For development against a repository checkout, use `python -m pip install -e .`
from the repository root instead.

Start with the unified GUI:

```bash
handtracker gui
```

Use the GUI dropdown for Hands, Face, Pose, or Everything. OptiTrack workflows
require a compatible SDK installed separately under OptiTrack's license. No
OptiTrack SDK files or recorded datasets are included in this download.

These examples are versioned with the corresponding HandTrack release. Download
the bundle attached to the same release as your installed package.

MANO virtual-hand examples require separately obtained model files. Set
`MANO_MODELS_DIR` to their directory before running those examples.
