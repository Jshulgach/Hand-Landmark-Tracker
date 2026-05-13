# Backends

HandTrack currently ships with two application backends.

## OptiTrack

The OptiTrack backend uses the compiled SDK binding exposed through `optitrack_cam`.

Expected capabilities:

- multi-camera live preview
- calibration capture
- triangulated 3D hand reconstruction
- GUI-based streaming and visualization

If the SDK is unavailable, the CLI will not choose OptiTrack in `auto` mode.

## Webcam

The webcam backend provides a zero-SDK fallback path based on standard OpenCV capture devices.

Expected capabilities:

- live camera preview
- calibration capture
- GUI-based tracking and visualization
- testing and downstream protocol validation

## Choosing a Backend

Use auto selection unless you are debugging:

```bash
handtracker gui
```

Force a backend when needed:

```bash
handtracker gui --backend webcam
handtracker gui --backend optitrack
```