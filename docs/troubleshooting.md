# Troubleshooting

## `mavis-track demo` Opens but No Camera Feed Appears

Run:

```bash
mavis-track doctor --backend webcam
mavis-track cameras --backend webcam
```

If OptiTrack is selected automatically, also run:

```bash
mavis-track inspect-calibration --backend optitrack
```

## MediaPipe Import Errors

MAVIS currently depends on the classic `mediapipe.solutions` API. Use the pinned package version from `pyproject.toml`.

## OptiTrack SDK Not Found

Set `OPTITRACK_CAM_PY_PATH` or place the compiled SDK module in the expected backend `Release` directory.

## Calibration File Missing

Use:

```bash
mavis-track calibrate --backend webcam
mavis-track calibrate --backend optitrack
```

Then verify the output with:

```bash
mavis-track inspect-calibration --backend webcam
```

## LSL local discovery

If local LSL discovery fails, consult the [official LSL configuration guide](https://labstreaminglayer.readthedocs.io/info/lslapicfg.html).
The release probe used a process-local configuration with machine scope and
`KnownPeers = {127.0.0.1}`. That check does not validate another computer's firewall
or an external recorder. Do not change global network settings just to run a demo.

## CLI Is Not Found

Make sure the environment is activated and the package was installed into that same environment:

```bash
python -m pip install -e .
```
