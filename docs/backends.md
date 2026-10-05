# Camera workflows

The default `mavis-track demo` processes one webcam or video with no
calibration. The Python frame API can use any caller-owned image source.

## Calibrated webcams

```bash
mavis-track board --backend webcam
mavis-track calibrate --backend webcam
mavis-track gui --backend webcam --advanced-hands
```

Use matching camera resolutions and a accurately printed ChArUco board.
The solver checks held-out reprojection errors, rejects failed candidates,
preserves the prior calibration, and publishes a validated result atomically.
Every fifth capture is reserved from all model fitting for a separate reconstructed
board check. Default gates require at least three reserved views, shape RMS at most
2 mm, scale error at most 2%, and planarity RMS at most 1 mm. These are commissioning
gates, not claims of physical tracking accuracy.
Calibration count, source indices, image sizes, and matrices are checked when loaded.
Legacy files lacking identity or size metadata produce explicit warnings.

Source indices are not physical camera identities and can change after USB
replugging. Verify physical order and recalibrate after a change. UID-based
remapping is supported when a camera manager supplies reliable UIDs; the current
OptiTrack binding integration does not establish that hardware identity here.

Tracking excludes frames older than 250 ms and cameras whose host receipt times
differ by more than 50 ms. Host receipt timestamps do not establish synchronized
hardware exposures. Fast motion still needs a hardware synchronization assessment.

Advanced multi-camera hand matching currently assumes one hand across views.
Two-hand and multi-person association across cameras remain future work.
Geometry diagnostics are local pixel-noise estimates; they do not include
calibration error or model bias.

## OptiTrack

The optional OptiTrack path uses `optitrack_cam` and a separately obtained
Windows Camera SDK. Set `OPTITRACK_CAM_PY_PATH` to a compatible binding folder.
The author reports historical successful Windows hardware tests. No current
OptiTrack hardware retest was possible during this release work.

SDK binaries and headers are excluded from wheel, source distribution, and curated
example downloads. Existing repository history contains earlier development
copies; those are not covered by MAVIS's MIT license.
