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
| `record` | Record hands, or synchronized hands/body with `--mode hand-body` |
| `replay` | Draw recorded hands/body using saved timestamps |
| `sync-report SESSION` | Audit raw camera videos and align them to a Vicon C3D using explicit anchors |
| `sync-replay SESSION --vicon TRIAL.c3d` | Separate synchronized video/Vicon review window |
| `export` | Hand/body CSV landmarks, angles, presence and quality metadata |
| `doctor --backend webcam` | Required and optional runtime dependency checks |
| `inspect-calibration --backend webcam` | Inspect saved calibration |
| `benchmark --backend webcam` | Calibrated backend timing; requires calibration |
| `test-sender --backend webcam` | Example downstream packets |

```bash
mavis-track demo --source 0 --mode hands
mavis-track demo --source clip.mp4 --mode pose
mavis-track record --source 0 --frames 300 --max-hands 2 --session-name demo
mavis-track record --mode hand-body --source 0 --frames 300 --session-name reach
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

Combined hand/body recording uses one person's Holistic result per source frame.
Use `--visibility-threshold 0.5` and `--max-gap 0.5` to configure quality checks.
It stores two hand slots plus body points, body visibility, and per-region flags
on one timeline in session schema 2. Replay and annotated video include the body;
export adds `body_landmarks.csv` and `tracking_quality.csv`. Hand-only recording
continues to use schema 1. See [combined tracking](integration.md#combined-hand-and-body-tracking)
for coordinate definitions, flag meanings, and single-person limitations.


Long recordings flush every `--chunk-frames` observations (default 4096), with a
small final chunk as needed. Short recordings retain `landmarks.npz`; long ones
use a schema-3 `session.json` and bounded NPZ files in `chunks/`. Replay/export
accept the same session directory and stream it. Ctrl+C preserves completed
observations and marks the recording incomplete. See the
[long-recording guide](integration.md#long-recordings-and-bounded-memory-analysis).

The live desktop studio performs capture, model initialization, inference and
mode switching in a worker. The preview always uses the newest available frame.
Stop and close remain responsive while native operations finish; the camera is
released by its owning worker before another capture is started.

Raw OptiHub video sessions use different files from numeric landmark sessions.
Use `sync-report` and `sync-replay` for their gap diagnostics, C3D alignment,
sampled exposure-signal analysis and optional calibrated marker overlays. See
[Vicon trial workflow](vicon-sync.md) for commands, anchors and confidence limits.

## Optional WiLoR model

After the [isolated WiLoR setup](wilor.md), source checkouts support:

```powershell
./.venv-wilor/Scripts/python.exe -m mavis_track demo --model wilor --source 0
./.venv-wilor/Scripts/python.exe -m mavis_track doctor --model wilor
```

Use `--wilor-device cuda --wilor-fast` to request CUDA half precision and
layer skipping, or `--wilor-root PATH` for a shared source/assets folder.
WiLoR supports single-source and OptiTrack multi-camera hand preview:

```powershell
./.venv-wilor/Scripts/python.exe -m mavis_track gui --backend optitrack --model wilor --wilor-device cuda --wilor-fast --wilor-batch-size 2
```

The grid opens every connected OptiTrack camera. Click Start Cameras and choose
matched mesh overlays or independent live camera feeds. WiLoR calibrated
triangulation, recording and streaming are not enabled.
