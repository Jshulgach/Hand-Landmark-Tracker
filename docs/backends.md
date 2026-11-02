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

Launch the tracking window with `mavis-track gui --backend optitrack`. The window
starts within the current screen's available desktop area, including window
borders and the taskbar. Camera and 3D previews resize with the window while
preserving image proportions; the right-hand controls scroll on smaller screens.
Close and relaunch an already running GUI after updating the source.

Camera preview and per-camera 2D hand overlays work without calibration. If the
saved calibration is absent, unreadable, or rejected for the connected cameras,
the GUI shows a calibration notice and keeps the feeds running. It disables the
3D view, reconstruction controls, and reconstructed joint angles; existing
UDP/LSL streams report unavailable hands rather than fabricated 3D observations.

To enable 3D, stop preview to release the cameras, run
`mavis-track calibrate --backend optitrack`, then start tracking again. Each start
validates the saved calibration against the connected camera count, identities
when available, and image sizes before enabling reconstruction.

The optional OptiTrack path uses `optitrack_cam` and a separately obtained
Windows Camera SDK. Set `OPTITRACK_CAM_PY_PATH` to a compatible binding folder.
The author reports historical successful Windows hardware tests. No current
OptiTrack hardware retest was possible during this release work.

SDK binaries and headers are excluded from wheel, source distribution, and curated
example downloads. Existing repository history contains earlier development
copies; those are not covered by MAVIS's MIT license.

## Display and calibration controls

The OptiTrack tracking window supports point-sized text, monitor DPI changes, a
resizable sidebar and a text zoom control. Camera tiles use an adaptive grid.
**Fit full image** preserves every pixel; **Fill area** crops the displayed edges
only. Display zoom/cropping never changes detector input or calibration geometry.

The OptiTrack calibration preview uses the same scalable camera tiles, readable
text and a text zoom control. Its compact footer lists these keyboard commands:

| Key | Action |
|---|---|
| Up / Down | Increase / decrease exposure by 1 |
| Shift + Up / Down | Increase / decrease exposure by 10 |
| 0–9 | Select the matching logical camera index |
| A | Apply exposure changes to all cameras |
| Space | Pause / resume automatic board capture |
| Q | Finish capture and attempt calibration with sufficient observations |
| Esc or close | Cancel without replacing the saved calibration |

Exposure values are SDK settings, not assumed milliseconds. The status line shows
the target cameras and requested values; acquisition pauses briefly after a
change to avoid collecting frames from before that change. A failed setting is
reported in the preview. With PyQt5 unavailable, an OpenCV fallback uses W/S and
Shift+W/S for exposure and displays those keys in its footer.

For the estimator accuracy audit and equal-priority hand/whole-body roadmap, see
[the MMPose comparison](mmpose-comparison.md).
