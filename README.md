<div align="center">
  <img src="https://raw.githubusercontent.com/Jshulgach/Hand-Landmark-Tracker/main/docs/source/_static/hand-demo.gif" width="70%" alt="Hand tracking demo">
</div>

# MAVIS — Motion Analysis and Visual Interaction Suite

Track hands from a webcam, video, or frames supplied by your own application.
MAVIS provides a camera-free Python API, a desktop demo, recording and replay,
CSV exports, and optional calibrated multi-camera and UDP/LSL workflows.

**Available on [PyPI](https://pypi.org/project/mavis-track/):** the distribution
name is `mavis-track`, the Python import is `mavis_track`, and the command is
`mavis-track`. Version 0.1.0 was published on October 4, 2026.
Existing `handtrack` imports and the `handtracker` command remain available
for compatibility.

## Try the desktop demo

Use Python **3.10–3.12, 64-bit**. The intended desktop platforms are Windows
x64, Linux x64, and macOS Intel/Apple Silicon. The software validation matrix
passes on all four with Python 3.10–3.12; camera permissions and visible desktop
behavior need device checks. See [the audit](AUDIT.md) for the evidence.

Create a virtual environment, then install the desktop extra:

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\activate` on Windows, or
`source .venv/bin/activate` on macOS/Linux, then:

```bash
python -m pip install "mavis-track[gui]"
mavis-track doctor --backend webcam
mavis-track demo
```

The demo starts with one camera and needs no calibration. Select hands, face,
pose, or everything in the window. Use `--source 1` for another camera or
`--source path/to/video.mp4` for a video.
For source development, clone this repository and install `.[gui]` instead.

## Include MAVIS in your project

The base package does not require Qt, pandas, or LSL. Add `mavis-track`
to your project's dependencies, or install it with `python -m pip install mavis-track`.
Pass frames from your existing capture loop:

```python
from mavis_track import HandTracker

with HandTracker(max_hands=2) as tracker:
    result = tracker.process(frame_bgr)  # Your uint8 image, shape (height, width, 3)
    for hand in result.hands:
        print(hand.track_id, hand.handedness, hand.landmarks.shape)
        # (21, 3) read-only coordinates owned by this result
```

MAVIS does not open a camera or window in this API. Empty frames produce an
empty `result.hands` tuple. Supply `timestamp=seconds` for a source clock
or use the default monotonic host clock. RGB input uses `color="rgb"`.
See [the integration guide](docs/integration.md) and the complete
[webcam example](examples/01_basic_tracking/mavis_webcam.py).

## Record, replay, and export

```bash
mavis-track record --source 0 --frames 300 --max-hands 2 --session-name demo
mavis-track record --mode hand-body --source 0 --frames 300 --session-name reach
mavis-track replay recordings/demo
mavis-track export recordings/demo
```

Recordings contain a numeric `landmarks.npz` and `session.json`; optional
`--save-video` adds annotated video. Export writes rectangular landmark,
angle, and hand identity CSVs. Missing hands use a validity mask and NaNs.
Replay honors recorded timestamps, including variable camera timing.

For synchronized single-person hand/body processing, import `HandBodyTracker`
from `mavis_track`. It returns body landmarks and both hands from the same frame,
with per-region missing-data, visibility, bounds, and timing flags. Combined
recordings retain body visibility and quality metadata; replay/export support
both regions. See the [combined API guide](docs/integration.md#combined-hand-and-body-tracking)
for usage and coordinate definitions.

## Coordinates and measurement limits

- `landmarks`: image-normalized x/y; z is the model's wrist-relative depth,
  scaled similarly to x. These coordinates are not calibrated world meters.
- `world_landmarks`: optional hand-centered MediaPipe model estimates in
  meters. Separate hands do not share a global origin.
- `angles`: 14 named geometric finger flexion estimates in degrees. The
  `angle_space` field identifies the coordinates used. These are not
  validated clinical joint measurements.
- `track_id`: short-lived association across adjacent frames. Missing hands
  reset association; IDs do not identify a person.

## Advanced workflows

Install `".[gui,stream]"` for desktop streaming. UDP can also be used with
the base package. Calibrated multi-camera tracking currently assumes **one
hand shared across views**; unrestricted multi-person/two-hand matching
requires further work.

```bash
mavis-track board --backend webcam
mavis-track calibrate --backend webcam
mavis-track gui --backend webcam --advanced-hands
```

Calibration uses matching image dimensions, held-out quality checks, and
validated camera matrices. Files live in a user data directory, with existing
`HANDTRACK_WEBCAM_CALIBRATION_DIR` and
`HANDTRACK_OPTITRACK_CALIBRATION_DIR` overrides supported. Source indices can
change when cameras are replugged: verify physical order or recalibrate.

OptiTrack remains an optional Windows SDK integration. The author has historical
Windows hardware tests; current release work has no OptiTrack hardware retest.
Install the official SDK/USB drivers separately. The tracked
[external setup guide](external/README.md) provides verified local runtime recovery
for Python 3.11 x64; source installations discover that runtime automatically.
Use `OPTITRACK_CAM_PY_PATH` for another compatible binding.

## Development and release

```bash
python -m pip install -e ".[dev,gui,io,stream,docs,release]"
python -m pytest -q
mkdocs build --strict
python -m build
python -m twine check dist/*
```

Publishing is gated by the full software matrix and package checks.
[Audit](AUDIT.md) · [Release checklist](RELEASE.md) ·
[Documentation](https://jshulgach.github.io/Hand-Landmark-Tracker/) ·
[Issues](https://github.com/Jshulgach/Hand-Landmark-Tracker/issues)

MAVIS source is MIT licensed. Optional Qt dependencies and proprietary SDKs
retain their own licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).




## Repository layout

| Path | Purpose |
| --- | --- |
| `src/mavis_track/` | Public frame API, numeric sessions and incremental recording |
| `src/handtrack/` | CLI, processing, GUI and compatibility interfaces |
| `src/unity_hand_tracking/` | Active advanced webcam/OptiTrack implementations and Unity compatibility |
| `tests/` | Hardware-independent regression tests |
| `examples/` | Runnable examples and retained research workflows |
| `research/camera-placement/` | Preserved historical Flex placement evaluations |
| `outputs/`, `recordings/` | Ignored new experiment outputs |
| `build/`, `site/`, tool caches | Ignored packaging outputs; cleared after use |

Recording, replay and export support bounded chunks for long sessions; see the
[streaming guide](docs/integration.md#long-recordings-and-bounded-memory-analysis).
The live studio keeps capture/inference off the GUI thread and displays the
latest preview without a backlog. Historical placement reports are indexed in
[research](research/camera-placement/README.md).


Research datasets, model weights, local legacy scripts and validation snapshots
are stored outside the checkout in `~/Documents/Hand-Landmark-Tracker-Research/`.
The examples accept `--data-dir` and `--video`; `MAVIS_DATA_DIR` can select another
shared data root. See [external research data](docs/research-data.md).
