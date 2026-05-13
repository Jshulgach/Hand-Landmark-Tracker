# Troubleshooting

## `handtracker gui` Opens but No Camera Feed Appears

Run:

```bash
handtracker doctor
handtracker cameras
```

If OptiTrack is selected automatically, also run:

```bash
handtracker inspect-calibration --backend optitrack
```

## MediaPipe Import Errors

HandTrack currently depends on the classic `mediapipe.solutions` API. Use the pinned package version from `pyproject.toml`.

## OptiTrack SDK Not Found

Set `OPTITRACK_CAM_PY_PATH` or place the compiled SDK module in the expected backend `Release` directory.

## Calibration File Missing

Use:

```bash
handtracker calibrate --backend webcam
handtracker calibrate --backend optitrack
```

Then verify the output with:

```bash
handtracker inspect-calibration --backend webcam
```

## CLI Is Not Found

Make sure the environment is activated and the package was installed into that same environment:

```bash
python -m pip install -e .
```