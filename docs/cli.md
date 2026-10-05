# CLI reference

The primary command is `mavis-track`. The `handtracker` alias remains
for compatibility. Use `mavis-track COMMAND --help` for all options.

| Command | Purpose |
| --- | --- |
| `demo` | One camera or video; no calibration; requires GUI extra |
| `gui --backend webcam` | Same single-camera desktop studio |
| `gui --backend webcam --advanced-hands` | Calibrated multi-camera desktop |
| `calibrate --backend webcam` | Capture and validate ChArUco calibration |
| `board --backend webcam` | Generate calibration board |
| `cameras --backend webcam` | Preview capture devices |
| `record` | Record one source with one or two hand slots |
| `replay` | Draw recorded image-normalized hands using saved timestamps |
| `export` | Rectangular CSV landmarks, angles, and hand metadata |
| `doctor --backend webcam` | Required and optional runtime dependency checks |
| `inspect-calibration --backend webcam` | Inspect saved calibration |
| `benchmark --backend webcam` | Calibrated backend timing; requires calibration |
| `test-sender --backend webcam` | Example downstream packets |

```bash
mavis-track demo --source 0 --mode hands
mavis-track demo --source clip.mp4 --mode pose
mavis-track record --source 0 --frames 300 --max-hands 2 --session-name demo
mavis-track replay recordings/demo
mavis-track replay recordings/demo --no-display
mavis-track export recordings/demo --variant raw_landmarks
```

Record stores `landmarks.npz` and `session.json`.
`--save-video` adds `annotated.mp4`. Export creates `landmarks.csv`,
`angles.csv` when available, `hands.csv` for the modern format, and
`export_manifest.json`. Existing session directories are never overwritten.

Video timestamps use reported presentation timestamps with a frame-rate fallback.
Live timestamps use host receipt time relative to recording start. Fractional
frame rates are preserved. Replay defaults to saved intervals; `--fps` overrides
them. It accepts a session folder or an NPZ path.

Backend commands support `--backend auto|webcam|optitrack`; auto prefers an
installed OptiTrack SDK. `demo` always uses one OpenCV source. Select a backend
explicitly when commissioning calibrated cameras.
