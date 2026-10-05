# MAVIS core examples

This first-release bundle contains `mavis_webcam.py`, licenses, and the legacy
Unity CSV bone-rotation listeners. SDK assets and recorded data are excluded.

Install `mavis-track` from PyPI with Python 3.10–3.12 and run
`python mavis_webcam.py`. For source development, install the repository with
`python -m pip install .` instead.

For the desktop studio, install the GUI extra and run `mavis-track demo`.
For recording, replay, and export:

```bash
mavis-track record --source 0 --frames 300 --max-hands 2 --session-name demo
mavis-track replay recordings/demo
mavis-track export recordings/demo
```

The Python example demonstrates caller-owned capture and guaranteed cleanup.
See the project's integration guide for result coordinates, presence, and IDs.
The Unity listener files accept legacy CSV rotations; they do not consume the
MAVIS JSON landmark packets. A Unity JSON receiver is a separate integration.
