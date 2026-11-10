# OptiTrack video recording and OptiHub 2 sync

Launch `mavis-track gui --backend optitrack`. The controls are independent:

1. Uncheck **Enable Inference** before **Start Cameras** for a camera-only session.
   Camera startup then does not initialize MediaPipe detectors or load WiLoR
   weights. Inference can be enabled or disabled while capture continues. An
   already executing native inference call may finish, but its result is discarded
   after the toggle. Disabling inference clears overlays and temporal tracking;
   recording continues. Loaded models remain resident for reuse.
2. Enter a **Trial** name matching Nexus and choose the video output folder.
   **Start Recording** creates a unique timestamped session directory. The
   chosen parent folder stays selected for subsequent trials in the window.
   Each camera records independently at its original
   resolution, without tracking overlays or the SDK text overlay. Recording does
   not require calibration or inference.
3. **Stop Recording** flushes the writers in the background. Stopping cameras or
   closing the window finishes the recording before releasing capture. If a
   writer is still blocked after the timeout, the window stays open for retry.

**Show Camera Background** controls the background under inference overlays;
it does not disable inference. Model selection requires stopping cameras.
Playback FPS, codec, sync presets and optional Nexus network control are inside
**Recording / sync setup…**. The live window shows a compact recording health
indicator; detailed counters/errors are in its tooltip and saved logs. Review
tools open separately. See [Vicon trial workflow](vicon-sync.md).

## Session files

| File | Contents |
| --- | --- |
| `<trial>_<UTC timestamp>_camera_<index>.avi` | Full-resolution video for each camera that supplied frames |
| `frames.jsonl` | One timing/metadata row for each written video frame |
| `sync.jsonl` | Hub configuration/status samples, normally every 200 ms |
| `manifest.json` | Session settings, camera settings/identity when available, counters, completion and errors |
| `diagnostics.json` | SDK gaps/resets, delivery timing and queue-drop counters |
| `events.jsonl` | Recording lifecycle events |
| `control_events.jsonl` | Nexus network requests/results when enabled, independently of frame queue overload |
| `calibration.npz` | Snapshot of the configured calibration if available, identified by SHA-256 |

The manifest's `video_files` mapping is authoritative. Trial names are made safe
for filenames; the original name remains in the manifest. Names contain UTC
recording-start timestamps, not per-exposure times or a cross-computer clock
guarantee. The default MJPG codec is lossy. Optional FFV1 preserves decoded
frames exactly but may require more storage/CPU. Neither format restores sensor
data already compressed by the camera's MJPEG mode. Codec support depends on
OpenCV's installed video backend; unsupported writers fail visibly.

**Playback FPS** in setup sets the AVI's nominal playback speed. It does not change camera
frame rate or resample the captured images. Use `frames.jsonl` for actual timing;
variable frame intervals and missing frames are not represented by AVI playback.
Rows map `camera_index` and zero-based `video_frame_index` to
`host_received_monotonic` in Python `time.perf_counter()` seconds, measured before
rasterization. This clock is shared with `host_sampled_monotonic` in sync samples
and `host_started_monotonic` in the manifest. It measures receipt rather than
exposure, and has no established mapping to another computer's clock.

The extended binding adds `sdk_frame_id`, `sdk_timestamp`,
`hardware_timestamp_valid`, and hardware ticks/frequency when the SDK reports
valid hardware timing. These remain separate clock domains; do not subtract
hardware/SDK timestamps from host timestamps. Not every camera supplies hardware
timing. SDK frame IDs can reveal acquisition gaps. Physical serial numbers are
recorded when supported; camera indices alone do not establish physical identity
or remap calibration.

The original binding can record videos and host timing. Its metadata explicitly
sets `sdk_timing_available` to false, and the GUI reports sync configuration as
unavailable until the extended binding is installed. Build/import verification does not
validate timing against physical exposures.

Capture reads sequential SDK frames while recording instead of draining them
to the newest image. Preview and inference still use replaceable latest-frame
slots. A bounded queue of 32 items prevents unbounded recording memory use.
When encoding/storage cannot keep up, queue drops are counted in the GUI and
manifest. The count includes dropped video frames or sync samples; it is not a
hardware pulse count or a guarantee of zero SDK/USB losses. Inspect frame IDs and
bench-test the actual camera count, resolution, frame rate and storage device.
Disk space is checked every 100 written frames, stopping below 64 MiB free.

Writer failures are shown in the GUI and leave an incomplete manifest. Sidecars
are flushed incrementally and manifest counters checkpoint every 100 frames.
An interrupted AVI may require repair; these video sessions are separate from
the numeric sessions consumed by `mavis-track replay`/`export`.

## OptiHub 2 (2011) limits

The legacy Camera SDK 1.10 headers expose hub synchronization settings through
`CameraManager::GetSyncSettings`, including input source, trigger selection,
divider and global offset. The GUI reads these settings and records samples
alongside video. Mode changes require the explicit **Apply Sync Mode** action
and are disabled during recording.

The SDK's `Frame` documentation restricts synchronization telemetry, including
external lock, to eSync devices. The OptiHub interface has no documented input
level or timestamped edge stream. Therefore the GUI says **Input pulses/level:
unavailable (SDK)** and every manifest sets
`input_edge_recording_supported: false`. An external-source setting does not
prove a signal is connected, active, or locked. Samples describe configuration,
not the electrical signal; no pulse rate or waveform is inferred from camera FPS.

Recording every electrical edge requires an additional digital acquisition
device and a measured clock relationship to exposures (for example, recording
the incoming trigger and the hub exposure output together). Input-divider and
gating settings can prevent some input edges from producing camera exposures.
The hardware's synchronization role is described in the official
[OptiHub guide](https://docs.optitrack.com/v1.1/external-device-setup/custom-synchronization-optihub).

## Vicon Nexus at 50 Hz

Connect the Lock Lab's selected GPO/Sync Out to the OptiHub's **External Sync In**.
In Nexus, confirm the actual system rate is 50 Hz and assign `DV_Normal.gpo` to
the output socket you wired. This program sends one high/low cycle per Vicon
frame after its `MXDVStart` event; confirm pulses are active before expecting
camera frames. See the official [Nexus output settings](https://help.vicon.com/space/Nexus220/897715600/Vicon%2BConnectivity%2Bnodes)
and [GPO definitions](https://vicon-help.atlassian.net/wiki/spaces/Connect/pages/1611203).

Start cameras in MAVIS, open **Recording / sync setup…**, select **Vicon Nexus — external 50 Hz**, then click
**Apply Sync Mode**. The binding applies custom synchronization with external
input, rising-edge trigger, divider 1, and global offset 0 microseconds. It
preserves the current OptiSync/WiredSync topology and output configuration.
The preset expects Vicon to supply 50 Hz; it neither programs Vicon nor measures
the incoming pulse rate. It also sets the recording playback FPS to 50.

To recover without incoming pulses, select **Internal — 50 Hz (free run)** and
Apply. This sets the internal generator to 50 Hz. External mode can start camera
workers without fresh frames, leaving the GUI available while waiting for Vicon
or for switching back to internal mode. Camera failures still report errors.
Changing the combo box alone does nothing, and starting cameras does not apply
a preset automatically. Settings readback verifies SDK configuration, not
electrical lock or exposure alignment. The native setter attempts to restore
the previous settings if application/readback fails.

`expected_frequency_hz` in sync samples is the preset expectation, not a measured
frequency. Recordings still require an independent exposure-alignment check and
the GUI's own Start/Stop Recording actions.

## Install or build the extended native binding

On a configured Windows/Python 3.11 x64 tracking PC, close MAVIS and install the
[prebuilt sync-capable binding](installation.md#install-the-prebuilt-camera-binding):

```powershell
python external/setup_optitrack.py --upgrade-binding
python -m mavis_track doctor --backend optitrack --gui
```

This installs the verified extension without a compiler. The matching legacy
CameraLibrary DLL and USB drivers remain prerequisites.

To build from source instead:

Close MAVIS and use the Python 3.11 x64 build environment described in
[the local SDK guide](https://github.com/Jshulgach/Hand-Landmark-Tracker/blob/main/external/README.md#illumination-binding-upgrade), with
Visual Studio 2022 C++ tools, CMake and pybind11 installed:

```powershell
python external/build_optitrack.py --restore-sdk --install
```

The build verifies imports and exposes `CAMERA_RECORDING_API_VERSION = 1`,
`get_sync_status()`, `Frame.timing()` and `Camera.capture_info()`. The current
source also adds `OPTIHUB_SYNC_CONTROL_API_VERSION = 1` and
`configure_sync_preset()` for the two explicit presets. It retains the
base binding API version 2 for existing illumination/runtime compatibility.
Earlier prebuilt API v2 archives do not necessarily include these extensions.
SDK initialization, connected-hub operation and electrical timing still require
an on-hardware check after building.
The current source also exposes `configure_exposure_output()` for the setup
dialog's exposure timing output action. This does not generate Nexus remote
start/stop contact closures.
