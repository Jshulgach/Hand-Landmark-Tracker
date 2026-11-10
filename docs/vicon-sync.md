# Vicon mocap and OptiHub video trials

The live window intentionally keeps only trial name, output folder, camera and
inference controls, recording, and a short health indicator. Open **Recording /
sync setup…** for infrequently changed settings. Detailed diagnostics belong in
trial files; alignment reports and synchronized review run after capture.

## Configure the hardware once

1. Confirm Nexus's **Actual Frame Rate** is 50 Hz. Configure the chosen Lock Lab
   GPO output with `DV_Normal.gpo`; connect it to **OptiHub External Sync In**.
   Confirm the actual output is active and produces 50 pulses/second. The program
   starts on `MXDVStart`, which is distinct from the `StartCapture` event used by
   `Duration.gpo`; do not assume selecting the program starts a continuous signal.
2. In MAVIS setup, apply **Vicon Nexus — external 50 Hz**. This selects external
   rising-edge input, divider 1 and zero added offset. Internal 50 Hz remains an
   explicit recovery option when there are no Vicon pulses.
3. If recording exposure evidence in Nexus, connect **OptiHub External Sync Out**
   to a suitable Lock Lab analog input and use **Enable exposure timing on
   External Sync Out** in setup. Check voltage range and actual pulse widths
   with a scope. Select a sampling rate that resolves the shortest pulse;
   camera frame rate alone does not determine a sufficient analog rate.

The SDK readback checks configuration, not electrical lock. The native `.pyd`
must be rebuilt for the new controls; import tests and simulated SDK tests do not
validate connected hardware. See [binding build instructions](optitrack-recording.md#build-the-extended-native-binding).

Official references: [Nexus output configuration](https://help.vicon.com/space/Nexus220/897715600/Vicon%2BConnectivity%2Bnodes),
[Vicon GPO events](https://vicon-help.atlassian.net/wiki/spaces/Connect/pages/1611203),
[OptiHub synchronization](https://docs.optitrack.com/virtual-reality/synchronization-hardware/external-device-sync-guide-optihub2).

## Start and stop a named trial

### Optional separate controller on the Nexus PC

The standalone controller operates the running MAVIS GUI over loopback UDP.
MAVIS starts its video writer before sending the existing Nexus start request;
on stop it requests Nexus stop before finishing video files. Nexus requests
remain unconfirmed. This controls trial boundaries, not exposure alignment.
Keep the Lock Lab / OptiHub hardware synchronization configured as above.

In PowerShell, opt in to the MAVIS listener before launching the camera GUI:

```powershell
$env:MAVIS_CAPTURE_CONTROL_PORT = '5015'
handtrack-optitrack-gui
```

In MAVIS, choose the video output folder, start cameras (inference may be off),
and enable **Send Nexus start/stop over network** in recording/sync setup.
For Nexus on this PC, use `127.0.0.1`, port `30`. In Nexus configure network
start/stop to Receive and arm capture; verify loopback reception on the installed
Nexus version before collecting trials. The Nexus session/database stays in Nexus.

In a second terminal with the same Python environment, run:

```powershell
python -m mavis_track._capture_controller
```

Enter the trial name and use **Start both recordings** / **Stop both recordings**.
After reinstalling the package, `mavis-capture-controller` is also available.
The listener accepts only local traffic and is disabled unless the environment
variable is set. The controller defaults to port 5015 (`--port` changes it).
Only run one controller. Commands are not automatically retried; if a reply is
lost, inspect MAVIS and Nexus before issuing another command. Closing the
controller leaves recording running. Camera readiness and recording-file health
must still be checked in MAVIS; a reply confirms MAVIS command handling, not
Nexus capture or hardware synchronization.

Enter the same trial name used by Nexus and select a video output parent folder.
Each capture gets a unique subdirectory such as
`Reach01_20261109T153000_123456Z_a1b2c3d4`. Each video filename includes the trial,
UTC start timestamp and camera index. The exact filenames are in `manifest.json`.
Repeated trial names never overwrite earlier captures. The folder is reused
within the current GUI session; it is not silently written to an OS preference.

Choose the appropriate start/stop route inside setup:

* **Existing electrical trigger:** leave network control unchecked. Arm Nexus
  for the external device and start video recording before sending its start
  signal. Send its stop signal, then finish video recording. The GUI does not
  drive a physical relay or assume that an unknown trigger device is connected.
* **Optional network requests:** enable **Send Nexus start/stop over network**,
  enter the specific Nexus PC IPv4 address and port (default 30), and arm Nexus
  with **Start/Stop over network = Receive**. The GUI opens video recording before
  requesting Nexus start, and requests Nexus stop before flushing videos. It
  sends the exact trial name; Nexus's existing session/database location remains
  configured in Nexus. Its protocol ignores DatabasePath in received packets.

Network control defaults off. A successful UDP send is logged as
`sent_unconfirmed`, never as confirmed capture. No receiving/acknowledgement
service is implemented. Check Nexus itself, firewall settings, and trial files.
Sending a network command does not establish frame-accurate start alignment;
GPO hardware timing and observed alignment events are still required.
[Nexus UDP protocol](https://vicon-help.atlassian.net/wiki/spaces/Nexus216/pages/11609729/UDP%2Bcapture%2Bbroadcast%2Btrigger)
and [electrical remote input behavior](https://vicon-help.atlassian.net/wiki/spaces/Connect/pages/1608732)
describe these different mechanisms. The Lock's electrical inputs are separate
start/stop gate-to-ground inputs; do not connect an arbitrary voltage output to
them or short both simultaneously.

Stop recording before changing trial fields, codec or sync mode. The health
indicator shows waiting frames, camera errors, gap/drop warnings or trigger
errors. Its tooltip and the trial logs contain detail. An error sending a Nexus
request does not discard an otherwise usable local video recording.

## Files saved without filling the live window

* `manifest.json`: original trial name, UTC/host start references, video files,
  camera settings and serials when available, codec, software/binding versions,
  initial sync configuration, counts, and calibration SHA-256.
* `frames.jsonl`: written video indices, host delivery times, SDK IDs and optional
  hardware clock ticks. SDK gaps and resets are annotated. A queue drop and an
  SDK-ID gap can describe the same missing image; do not add them as unique losses.
* `sync.jsonl`: periodic configuration and per-camera streaming/error samples.
* `events.jsonl`: lifecycle events. `control_events.jsonl` separately audits Nexus
  requests even when the video writer queue is full or has failed.
* `diagnostics.json`: per-camera gap/reset counts, host delivery intervals and
  queue drops by camera/type. Delivery jitter is not exposure jitter.
* `calibration.npz`: immutable copy of the configured calibration, when present.
  Its presence does not assert it passed validation or register Vicon coordinates.

MJPG is the default storage codec. FFV1 is available in setup for lossless
recording of decoded camera frames. Test your camera count, codec, resolution,
duration and storage device together; capture never waits for a slow encoder.
The frame queue is bounded and overload is explicitly counted.

## Produce an alignment report

Install the optional C3D reader and review GUI:

```powershell
python -m pip install -e ".[gui,sync]"
```

First inspect a recording, even before the Vicon export is ready:

```powershell
mavis-track sync-report "D:\Videos\Reach01_..."
```

Export the Nexus trial to C3D, preserving the intended first/last frame range and
the exposure analog channel if one was recorded. Then supply an anchor CSV with
observed correspondences:

```csv
camera_index,sdk_frame_id,vicon_frame,role,source
0,1002,102,fit,observed exposure or common event
0,1098,198,fit,observed exposure or common event
0,1000,100,check,independent beginning event
0,1100,200,check,independent ending event
```

The numbers above are format examples, not defaults. Each camera needs its own
observed anchor evidence. `vicon_frame` may be fractional for a sampled exposure
edge between mocap frames. Declare in `source` whether the correspondence refers
to exposure start, exposure midpoint, or a visual event, and use that convention
consistently. Do not manufacture anchors by equating the first saved video frame
to the first exported Vicon frame. An event must identify a particular image;
an unpaired list of pulse times alone does not establish image identities.

```powershell
mavis-track sync-report "D:\Videos\Reach01_..." --vicon "D:\Nexus\Reach01.c3d" --anchors "D:\anchors.csv" --exposure-channel "Exposure" --threshold 2 --tolerance-ms 5
```

Outputs go under `alignment/`:

* `summary.txt` and `alignment.json`: gaps, resets, host delivery intervals,
  offset, fitted clock-rate difference and independent check-event errors.
* `frame_mapping.csv`: video index / SDK ID / estimated Vicon time and nearest
  mocap frame, plus confidence flags. Extrapolated, ambiguous, out-of-range and
  missing images are explicitly identified.
* `exposure_edges.csv`, when requested: sampled rising edges, pulse widths and
  midpoints in the C3D frame/time domain. JSON reports also give analog sample
  resolution and phase/deviation within the Vicon frame period. Threshold
  crossings are limited by sampling, clipping, noise and unknown hardware delays.

Without anchors, images remain unmapped. A single fit anchor fixes an assumed
rate/offset but cannot estimate drift. With multiple fit anchors, hardware ticks
are preferred; otherwise increasing SDK IDs at the declared nominal camera rate
are used. ID-based fitted drift is frame-count consistency, not an independent
measurement of a hardware oscillator. Repeated/reset SDK IDs require explicit
segment handling; the current tool refuses automatic alignment of that camera.

The `checked` status requires at least two separate check events spanning at
least 80% of the recorded camera-time range, within the selected tolerance.
The default 5 ms is an analysis threshold, not a hardware accuracy claim. It is
meaningful only if your event measurements resolve it. Fit residuals are not
independent validation. Mapping remains an estimate between observed events.

## Review separately from capture

```powershell
mavis-track sync-replay "D:\Videos\Reach01_..." --vicon "D:\Nexus\Reach01.c3d"
```

The separate window provides a shared Vicon frame slider, play/pause, raw camera
tiles and a labeled XY marker view in meters. Missing/unmapped images stay blank;
the viewer never silently holds a previous image. Confidence appears on decoded
images. C3D, recording metadata and mapping hashes must match the report.
`--no-display` checks identities and decodes mapped frames without opening a window.

Optional `--projection projection.json` overlays Vicon markers on the images when
you already have measured camera intrinsics and Vicon-to-camera registration:

```json
{
  "schema": "mavis-vicon-projection-v1",
  "coordinate_units": "meters",
  "cameras": {
    "0": {
      "serial": 12345,
      "image_size": [1280, 1024],
      "camera_matrix": [[1000, 0, 640], [0, 1000, 512], [0, 0, 1]],
      "dist_coeffs": [0, 0, 0, 0, 0],
      "vicon_to_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    }
  }
}
```

These matrices are schema examples, not usable calibration. Each transform maps
Vicon world coordinates to the original, unrotated camera image's camera frame;
translation units are meters. Physical serial and resolution must match the
recording. No transform is inferred from the existing OptiTrack calibration.

## Commissioning before measurements

Record a short validation trial and a longer one with independently identifiable
events near the beginning and end. Check offset, jitter, drift, pulse resolution,
SDK gaps, and file integrity. Run both IR camera systems together and check for
strobe interference, marker loss and image-quality changes. Define exposure-start
versus exposure-midpoint alignment, acceptable error, exposure/motion blur and
gap policy for your experiment. Keep original Nexus trial files as well as C3D
exports; export cropping and gap filling must not silently change your reference.
