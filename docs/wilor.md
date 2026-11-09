# Experimental WiLoR hand inference

MAVIS includes an optional experimental adapter to the
[official WiLoR implementation](https://github.com/rolpotamias/WiLoR).
It provides a caller-owned frame API and webcam/video preview with projected mesh
wireframes and joints. The Studio model selector switches between MediaPipe and
WiLoR, releasing the old model/camera before starting the replacement off the Qt
thread. Nothing is recorded by this preview.

## Validation status and scope

Software tests cover geometry, ownership, timestamps, association, installer
integrity, environment targeting, model switching and cleanup without requiring
licensed assets in CI.

On **2 November 2026**, real inference was also verified on Windows/Python 3.11
using the isolated environment below and the SHA-256-verified official checkpoints:

- A 1920 x 1080 frame from the repository's `HandDynamic.mp4` produced one right
  hand, 21 finite 3D joints and 778 finite mesh vertices. The projected overlay
  was inspected for gross alignment with the hand.
- That single CPU observation loaded the models in 14.7 seconds and processed
  the reference frame in 3.42 seconds using four CPU threads. This is **not** a
  controlled throughput or accuracy benchmark.
- Webcam 0 supplied three 640 x 480 frames and completed inference/cleanup.
  Those initial samples returned zero hand detections. During the subsequent
  native CPU preview, the operator confirmed that the mesh/joint overlay appeared
  for their hand. This is an operator observation, not a measured accuracy result.
- The native Qt CPU preview opened and published camera frames; its timer
  continued ticking while the model loaded. Nothing was recorded.
- The target RTX 3060 is in another computer. This development computer has
  Intel Iris Xe graphics: CUDA 11.8 binaries are installed, but CUDA availability
  is false here. **GPU inference, fast mode, GPU memory use and live GPU speed
  remain unverified.**

Pose/angle accuracy, infrared-camera performance, difficult occlusions and
long-session behavior still need evaluation. A plausible mesh is not an accuracy
measurement.

This first adapter supports **hands only**, with up to two detected hands.
Face/body/holistic modes remain MediaPipe. WiLoR also supports OptiTrack
multi-camera viewing with per-camera meshes and joints. This preview uses one
shared model, independent per-view tracking and bounded crop batches.
The OptiTrack GUI also supports experimental calibrated joint triangulation
and placement of a WiLoR mesh and skeleton in the shared 3D view. Recording,
UDP and LSL remain disabled for WiLoR. The standalone frame API still returns
independent per-view model estimates.
A Python caller can supply any image sources through the multi-view API; camera
capture and hardware synchronization remain the caller's responsibility.

A good-looking mesh in a web demo does not establish joint-angle accuracy or
performance on our infrared camera images. See the evaluation plan below.

## Source, assets and disk use

Keep one ignored source/assets directory, shared by the checkout and any local
environments. Branches contain our adapter, not a second copy of model weights.
Switching Git branches alone does not reduce environment or checkpoint disk use.

The integration was inspected against upstream revision
`fcb911312a38fa8badd30d9656a167485d61b8f9`. To prepare a small source checkout
from the MAVIS repository root:

```powershell
git clone --depth 1 --filter=blob:none --sparse https://github.com/rolpotamias/WiLoR.git external/WiLoR
git -C external/WiLoR sparse-checkout set wilor mano_data pretrained_models
git -C external/WiLoR rev-parse HEAD
```

If the displayed revision differs, treat it as a new runtime version requiring
validation. The inspected checkout already includes `model_config.yaml` and
`mano_mean_params.npz`; it does not include inference checkpoints or MANO.

Required layout:

```text
external/WiLoR/
  wilor/...
  pretrained_models/
    model_config.yaml
    detector.pt
    wilor_final.ckpt
  mano_data/
    mano_mean_params.npz
    MANO_RIGHT.pkl
```

Obtain `MANO_RIGHT.pkl` yourself by registering and downloading from the
[MANO website](https://mano.is.tue.mpg.de/) under its license; extract the right
model into the layout above. It is not included in MAVIS or downloaded by us.

The [upstream installation instructions](https://github.com/rolpotamias/WiLoR#installation)
link the detector and reconstruction checkpoints. The reconstruction checkpoint
alone is 2.39 GiB (about 2.56 GB); PyTorch/CUDA add substantial environment disk use.
These commands are optional manual downloads, not part of MAVIS installation:

```powershell
curl.exe -L --fail "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt" -o external/WiLoR/pretrained_models/detector.pt
curl.exe -L --fail "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt" -o external/WiLoR/pretrained_models/wilor_final.ckpt
```

## Limited-internet or offline tracking PC

Use the [installation audit and private transfer-kit workflow](wilor-install-audit.md)
to prepare the wheels and assets on an online PC, then install locally on the
RTX computer. The target does not need Git, a compiler or Python package indexes.
Windows/Python, NVIDIA and camera-driver prerequisites still apply.

## Isolated Windows environment

The tested profile is 64-bit Windows/Python 3.11, PyTorch **2.7.1+cu118**,
torchvision **0.22.1+cu118**, NumPy **1.26.4**, Lightning **2.5.2** and
Ultralytics **8.3.161**. The complete resolved dependency set lives in
`requirements/wilor-windows-cu118.txt`; its input pins are in the adjacent
`.in` file. This supersedes the earlier proposed Torch 2.0 profile.
[PyTorch provides this CUDA 11.8 pairing](https://pytorch.org/get-started/previous-versions/#v271).

The runtime includes MAVIS's normal dependencies so the model selector can
also use MediaPipe. Both OpenCV distribution names required by the upstream
packages are pinned to the same 4.11.0.86 release. Training datasets, Gradio,
XTCocoTools and an additional CUDA development toolkit are not installed.

`scripts/setup_wilor.py` installs **only** into `.venv-wilor`, explicitly
targeting that interpreter and subprocess environment. It preserves the working
`.venv`. It uses `uv` if available, with a tested `pip` fallback.
Chumpy's pinned Git revision needs pip available during its legacy build, so
the installer bootstraps build tooling and disables build isolation for that
package. The old detector checkpoint also needs `dill`, which is explicitly
pinned. Runtime Ultralytics auto-installs and settings sync are disabled.

On the **RTX 3060 computer**, first put your licensed `MANO_RIGHT.pkl` at
`external/WiLoR/mano_data/MANO_RIGHT.pkl`. This can be the only file in that
folder; the setup script preserves it in place.

From the repository root:

```powershell
git pull --ff-only
py -3.11 scripts/setup_wilor.py --setup --require-cuda
```

`--setup` performs source preparation, checkpoint download/reuse, isolated
dependency installation and verification in one command. It checks Python and
the MANO file before starting downloads. You do not need a separate
`pip install -e .`; the helper installs MAVIS in editable mode itself.

Source is downloaded into a temporary sibling directory at the pinned upstream
revision. Every destination is checked before merging. Existing MANO files,
checkpoints, partial downloads and other files stay in place; different existing
source files or Git revisions are preserved and reported. Failed preparation
cleans its temporary checkout and rolls back only newly copied source files.
A successful rerun reuses the prepared checkout without downloading source again.

Only the two required checkpoints are downloaded at a pinned Hugging Face
revision, with size/SHA-256 verification. Downloads can resume; an existing
different checkpoint is preserved and reported. Existing matching downloads are
reused. Missing checkpoints and CUDA packages can still require large downloads;
use the private transfer kit if the tracking PC cannot download those.

The `--require-cuda` check requires CUDA availability and a successful GPU tensor
operation; it fails explicitly on a CPU-only machine. No camera is opened.

Alternatively, copy the already prepared `external/WiLoR` folder from the
development computer, including its source and models, then use
`--install --require-cuda`. A virtual environment contains computer-specific
paths and must be recreated rather than copied.

To inspect an existing setup without installing or downloading:

```powershell
py -3.11 scripts/setup_wilor.py --check --require-cuda
./.venv-wilor/Scripts/python.exe -m mavis_track doctor --model wilor
```

The doctor reports assets, versions and GPU details without loading checkpoints
or opening cameras. Asset/package checks do not prove inference accuracy.

## Setup versus launching

The MANO pickle supplies the hand shape model. It does not include WiLoR's
inference source, detector/reconstruction checkpoints or PyTorch/CUDA packages.
The setup script prepares those once; it is not needed when launching the GUI.

`--prepare-source` uses only Git and Python's standard library. It does not run
pip or install `truststore`. If a truststore error appears, retain the complete
output to identify whether it came from Python startup, Git or a later install
step.

The pip fallback upgrades the **new `.venv-wilor` interpreter's pip first**.
Older pip releases cannot enable their optional truststore feature until that
package exists in the same environment. Installing it in the regular `.venv`
does not install it in `.venv-wilor`.

For an interrupted package setup, from the repository root:

```powershell
./.venv-wilor/Scripts/python.exe -m pip install --upgrade pip
py -3.11 scripts/setup_wilor.py --install --require-cuda
```

Use that first command only if `.venv-wilor/Scripts/python.exe` already exists.
The setup helper can create the isolated environment if it does not. It
preserves matching downloaded assets; the MANO file does not need to be
downloaded again.

## RTX 3060 target computer

NVIDIA lists the RTX 3060 as
[CUDA compute capability 8.6](https://developer.nvidia.com/cuda/gpus).
Record its actual VRAM and driver version rather than assuming a memory size:

```powershell
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

After installation, test a known positive video before using a webcam:

```powershell
./.venv-wilor/Scripts/python.exe -m mavis_track demo --model wilor --source docs/source/_static/HandDynamic.mp4 --wilor-device cuda --wilor-fast
./.venv-wilor/Scripts/python.exe -m mavis_track demo --model wilor --source 0 --wilor-device cuda --wilor-fast
```

Keep the whole hand in the image. Compare standard and fast modes separately;
CPU timings above do not predict RTX 3060 throughput. Stop/close the window to
release the camera. No recording is enabled by these commands.

## Live preview

Once the runtime and assets are available:

```powershell
./.venv-wilor/Scripts/python.exe -m mavis_track demo --model wilor --source 0 --wilor-root external/WiLoR
# Explicit CUDA and accelerated inference:
./.venv-wilor/Scripts/python.exe -m mavis_track demo --model wilor --source 0 --wilor-root external/WiLoR --wilor-device cuda --wilor-fast
# Video instead of webcam:
./.venv-wilor/Scripts/python.exe -m mavis_track demo --model wilor --source "C:/clips/hand.mp4" --wilor-root external/WiLoR
```

`--wilor-device auto` chooses CUDA if available, otherwise CPU. CPU inference
is available as a code path, but live speed is unmeasured and may be unsuitable.
Fast mode requires CUDA and enables upstream FP16 and backbone layer skipping;
we do not invoke `torch.compile`, which adds platform-dependent compilation and
startup cost. Measure its accuracy/speed tradeoff independently.

The selector can return to MediaPipe for hand, body or face preview.
Missing assets/dependencies appear as an error in the window; there is no
automatic fallback to a different estimator.

Use `--wilor-root` for a shared directory outside the checkout, or set
`MAVIS_WILOR_ROOT`. Imports and model loading do not change the process working
directory or download missing files.

## OptiTrack multi-camera GUI

After the isolated environment and the existing OptiTrack SDK/USB driver setup:

```powershell
./.venv-wilor/Scripts/python.exe -m mavis_track gui --backend optitrack --model wilor --wilor-device cuda --wilor-fast
```

Click **Start Tracking**. The grid discovers and displays all connected cameras.
The **MediaPipe / WiLoR** selector can change estimators while tracking is stopped.
Existing exposure, IR illumination, text zoom, fit/fill and scrolling controls
are reused. The WiLoR box-score threshold is adjustable. MediaPipe's tracking
threshold, skeleton smoothing and splay calibration controls remain disabled
for WiLoR; the camera inclusion and reprojection threshold controls apply to
calibrated WiLoR geometry.

One WiLoR network and detector are shared across the feeds. Hand crops from all
views are combined into bounded inference batches, rather than loading one large
network per camera. `--wilor-batch-size 2` is the default; choose 1 to lower
batch memory use. CUDA/fast-mode throughput still requires a target-PC check.

The display selector has two choices:

- **Mesh overlays (matched frames)** shows each result on the exact frame used
  for inference. The caption displays that frame's host-receipt age.
- **Live camera feeds** refreshes current video independently while inference
  runs. Old joints are never painted onto a newer raw frame.

The status panel shows each startup stage and elapsed time, including imports,
checkpoint loading, transfer to CPU/CUDA, and the first inference batch. Camera
video can appear before the model is ready. A first launch may take substantially
longer than subsequent batches; moving camera video alone does not confirm
inference. After the first result, check the CPU/CUDA indicator, completed batch
counter, inference timing, and hand counts. Zero detections are reported explicitly.
Select **Mesh overlays (matched frames)** to see results; **Live camera feeds**
shows current raw video and reports that overlays are off.

If startup stops progressing, the terminal's last `[WiLoR]` stage identifies where.
Initialization/inference exceptions include their traceback and Python executable
in the terminal while the live preview remains available. To check GPU detection:

```powershell
./.venv-wilor/Scripts/python.exe -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Feeds also remain visible during model loading, and remain available if model
initialization or inference fails. A disconnected/stale camera clears its tile.
Stop/close is asynchronous: the GUI stays responsive while the current native
inference finishes, then the model/cameras are released before another start.

The normal model-switch sequence is **Stop Tracking → select MediaPipe/WiLoR →
Start Tracking**. Camera workers release their handles before the next model
starts; the native SDK stays initialized until the application exits. This
avoids shutting down its global singleton on each model switch. Close/relaunch
an already running GUI after pulling this fix. Close the GUI before opening a
separate calibration or vendor camera application.

When cameras start, the GUI checks the same saved calibration, camera IDs and
resolutions as MediaPipe. A usable calibration enables **Show 3D View** and its
top, front, side, isometric and palm-plane projections. Enable that checkbox
after **Start Tracking** to see the WiLoR wireframe mesh and its own 21-joint
skeleton together. These views are centered/zoomed around the hands rather than
a floor-aligned capture-volume display. Missing/rejected calibration leaves the
per-camera overlays available and disables the shared 3D view.

WiLoR's full-image joint projections are undistorted and triangulated from
two or more enabled cameras. Association supports one hand per handedness;
multiple same-handed detections suppress that handedness rather than guessing
which people match. Cross-view track IDs are not treated as shared identities.
Joint consensus checks pixel reprojection, positive depth and at least a
two-degree ray angle. Views more than 50 ms behind the latest host receipt
timestamp are excluded; this is not hardware exposure synchronization.

A similarity transform fits a selected view's MANO skeleton to at least 12
accepted triangulated joints, including the wrist and palm bases. The same
rotation, scale and wrist translation place all 778 mesh vertices and all 21
model joints in the calibration's camera-0 frame. Thus mesh topology and the
WiLoR skeletal frame stay together. The selection favors the smallest joint
fit residual, with a 30 mm RMS fit gate and scale bounds of 0.25–4; these are
project sanity checks, not mocap accuracy specifications. The 3D label's tooltip
reports contributing cameras, joint reprojection error and mesh fit residual.

**The displayed mesh and skeleton remain fitted model estimates.** They are
distinct from the independently triangulated joints, retained separately in
the session packet. No monocular `wrist_camera_translation` is used to place
the shared mesh. Geometry clears when reconstruction is unavailable, ambiguous
or stale; frame age is shown explicitly. No WiLoR network stream is published.
Physical rig accuracy, difficult hand association and moving-hand synchronization
still require validation; the added tests use synthetic geometry and simulated
cameras.

Validation used **four simulated camera feeds with real WiLoR checkpoints** on
the Intel CPU host: all four views produced a hand mesh, only one runtime loaded,
raw video appeared during model startup, live rendering continued while inference
ran, and async shutdown completed. A separate mixed-resolution/missing-view check
preserved camera indices and a mirrored view's handedness; single/batched joint
outputs agreed within 3.8e-8 model meters in that reference test. These are
integration checks, not physical accuracy or GPU performance measurements.
The four-camera OptiTrack hub and RTX 3060 combination remains hardware-unverified.

For caller-owned multi-camera capture:

```python
from mavis_track import WiLoRMultiViewTracker

with WiLoRMultiViewTracker(device="cuda", fast=True, batch_size=2) as tracker:
    results = tracker.process_views([camera_0_bgr, camera_1_bgr, None, camera_3_bgr])
    # Same order as the inputs; None preserves a missing camera's position.
    # IDs belong to each result.source_id, not to a shared cross-camera identity.
```

## Python frame API

```python
from mavis_track import WiLoRTracker

# Your application supplies a uint8 BGR frame named frame.
with WiLoRTracker(root="external/WiLoR", device="auto") as tracker:
    result = tracker.process(frame, color="bgr", timestamp=1.0)
    for hand in result.hands:
        image_xy = hand.image_landmarks       # (21, 2), pixels
        joints = hand.joints                 # (21, 3), wrist-relative model meters
        vertices = hand.vertices             # (778, 3), same origin as joints
        angles = hand.angles                 # 14 geometric flexion estimates, degrees
        triangles = tracker.faces            # mesh triangle vertex indices
```

Use increasing seconds for caller timestamps. Without a timestamp, the API uses
a monotonic clock. Close the tracker when finished; it also supports a context
manager. Input frames are not modified, and result arrays are owned/read-only.

The 21-joint order is wrist, thumb, index, middle, ring, little finger, matching
the order already supplied by WiLoR's MANO wrapper. It must not be reordered a
second time. Left-hand crops and camera x translation are reflected back before
full-image projection.

3D axes are x right, y down, z away from the camera. Joints and vertices are
centered at the wrist; `wrist_camera_translation` is an **estimated** position
using WiLoR's assumed focal length, not calibrated metric position. The result
explicitly labels image pixels, wrist-relative model coordinates and estimated
camera translation. These are separate types from MediaPipe results so that
different z/scale conventions cannot be silently interchanged.

`detection_score` is the hand detector's box score, not joint visibility,
handedness probability or reconstruction accuracy. WiLoR does not supply
calibrated per-joint confidence here. IDs associate nearby same-handed detections
between adjacent frames; they reset after absence or a gap over 0.5 seconds.
They do not identify people. Angles remain model-derived geometric estimates.

## Accuracy and performance validation

1. Verify one reference RGB image against the pinned upstream demo: left/right
   labels, projected joints, mesh and camera translation.
2. Test a webcam and our infrared camera frames, including darkness, occlusion,
   motion blur, extreme wrist poses and two overlapping hands.
3. Compare MediaPipe/WiLoR on identical frames with labelled or independently
   measured reference data. Report joint/angle errors, detection misses,
   left/right swaps, temporal jitter and failure recovery separately.
4. Measure startup, steady-state latency p50/p95, frame age, throughput and peak
   GPU memory, separately for standard and fast mode.
5. Evaluate the experimental calibrated mesh path against measured reference
   data. Keep fitted meshes distinct from triangulated joints; assess camera
   distortion, source-frame age and cross-view hand association explicitly.

WiLoR improves hand inference capabilities; it does not replace the equal-priority
whole-body evaluation described in the [model roadmap](mmpose-comparison.md).

## Licensing

WiLoR's upstream repository/model license is
[CC BY-NC-ND 4.0](https://github.com/rolpotamias/WiLoR/blob/main/license.txt).
MANO and Ultralytics have separate terms linked by upstream. These dependencies
and assets are not relicensed under MAVIS's MIT license. MAVIS distributes
our adapter only, not upstream source, checkpoints or MANO assets.
