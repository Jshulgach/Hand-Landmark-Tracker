# CLI Reference

The main entrypoint is `handtracker`.

## Generic Commands

### `handtracker gui`

Launch the primary live tracking GUI.

### `handtracker calibrate`

Run the multi-camera ChArUco calibration workflow.

### `handtracker cameras`

Open a live camera preview grid for the selected backend.

### `handtracker board`

Generate the ChArUco calibration board image used by the calibration workflows.

### `handtracker test-sender`

Send example packets through the broadcasting pipeline for downstream integration testing.

### `handtracker benchmark`

Run a backend timing benchmark and print summary statistics for FPS, capture time,
detection time, and triangulation time.

### `handtracker record`

Record a single webcam or video source into a session folder containing:

- `landmarks.npz`
- `angles.csv`
- `session.json`
- `annotated.mp4` when `--save-video` is enabled

### `handtracker replay`

Replay a recorded session folder by drawing the saved hand skeleton on a 2D canvas.
This command is intended to work directly with the `landmarks.npz` bundles created by
`handtracker record`.

### `handtracker export`

Export a recorded session folder into flat CSV files for downstream analysis, notebooks,
or external tools that do not want to read the original NPZ bundle directly.

### `handtracker doctor`

Inspect Python dependencies and backend availability.

### `handtracker inspect-calibration`

Summarize the saved calibration artifact for the selected backend.

## Backend Selection

All backend-aware commands support:

```bash
--backend auto
--backend optitrack
--backend webcam
```

`auto` prefers OptiTrack when the SDK is available, then falls back to webcam.

## Examples

```bash
handtracker gui --backend webcam
handtracker calibrate --backend optitrack
handtracker inspect-calibration --backend webcam
handtracker doctor --backend optitrack
handtracker benchmark --backend webcam --frames 120
handtracker record --source 0 --frames 300 --save-video
handtracker replay recordings/session_20260512_120000
handtracker export recordings/session_20260512_120000
```

## Recording Notes

`handtracker record` is intentionally scoped to single-source capture for now. Use it
with a webcam index such as `0` or with a video file path.

## Replay Notes

`handtracker replay` expects a session directory, not the `landmarks.npz` file itself.
It uses the saved session sampling rate by default, or you can override playback speed
with `--fps`.

## Export Notes

`handtracker export` writes CSV files under `<session>/exports` by default. The command
can export either the filtered `landmarks` array or the `raw_landmarks` array with
`--variant raw_landmarks`.