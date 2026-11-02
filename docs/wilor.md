# Experimental WiLoR hand inference

The `codex/wilor-backend` branch adds an optional adapter to the
[official WiLoR implementation](https://github.com/rolpotamias/WiLoR).
It provides a caller-owned frame API and webcam/video preview with projected mesh
wireframes and joints. The Studio model selector switches between MediaPipe and
WiLoR, releasing the old model/camera before starting the replacement off the Qt
thread. Nothing is recorded by this preview.

## Validation status and scope

Software tests cover crop projection, left-hand reflection, wrist centering,
coordinate labels, color conversion, timestamps, array ownership, association,
cleanup and command routing using synthetic predictions. **Real WiLoR inference,
Windows dependency compatibility, accuracy, latency and GPU memory use have not
yet been validated in this integration.** The licensed MANO asset is not available
on the development computer, and its GPU is Intel Iris Xe rather than NVIDIA.

This first adapter supports **hands only**, with up to two detected hands.
Face/body/holistic modes remain MediaPipe. WiLoR is not yet connected to the
advanced OptiTrack GUI, calibrated triangulation, recording, UDP or LSL. Explicit
unsupported CLI combinations fail rather than silently selecting MediaPipe.
A Python caller can supply frames from any camera, including OptiTrack, but must
own capture and any multi-camera association/synchronization.

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
alone is roughly 2.4 GB; PyTorch/CUDA add substantial environment disk use.
These commands are optional manual downloads, not part of MAVIS installation:

```powershell
curl.exe -L --fail "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt" -o external/WiLoR/pretrained_models/detector.pt
curl.exe -L --fail "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt" -o external/WiLoR/pretrained_models/wilor_final.ckpt
```

## Environment

Use a separate `.venv-wilor`; keep the working MAVIS/OptiTrack `.venv` intact.
Do not install large GPU dependencies until the target computer's GPU/driver and
available disk space are known. Choose a compatible Torch/torchvision pair using
[PyTorch's installer](https://pytorch.org/get-started/locally/) and the
[upstream tested environment](https://github.com/rolpotamias/WiLoR#installation),
which uses Python 3.10, Torch 2.0 and CUDA 11.7. That older CUDA build is not a
blanket recommendation for newer GPUs.

Follow the upstream dependency instructions inside that separate environment,
then install this branch with its GUI:

```powershell
python -m pip install -e ".[gui]"
python -m mavis_track doctor --model wilor --wilor-root external/WiLoR
```

The upstream requirements include training/demo dependencies and older libraries;
a minimal, reproducible Windows inference lockfile is pending validation on the
target GPU. In particular, newer Torch checkpoint-loading defaults and NumPy /
Chumpy / Ultralytics / OpenCV combinations need checking. Avoid upgrading the
working environment to resolve these. The WiLoR doctor reports assets, package
versions and CUDA/GPU details without loading checkpoints or opening cameras;
an `ok` dependency check is not proof that inference works.

## RTX 3060 target computer

The target computer has an NVIDIA RTX 3060 (operator reported 2 November 2026).
NVIDIA lists it as [CUDA compute capability 8.6](https://developer.nvidia.com/cuda/gpus).
This supports testing a CUDA runtime; throughput and memory use still require
measurement. The development laptop's Intel GPU is a separate machine.

On the RTX 3060 computer, collect the actual VRAM and driver version:

```powershell
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

Do not assume a particular amount of VRAM from the RTX 3060 name alone.
A candidate close to upstream's tested Torch 2.0 family is Torch 2.0.1 with
torchvision 0.15.2 and CUDA 11.8 wheels, a combination provided in
[PyTorch's previous-version instructions](https://pytorch.org/get-started/previous-versions/#v201).
This is a proposed compatibility test profile, not a validated WiLoR environment
or complete dependency lock. Do not mix its packages into the working environment.

The next manual asset step is to register on the
[MANO website](https://mano.is.tue.mpg.de/), agree to its license, download the
MANO models archive and extract `MANO_RIGHT.pkl`. Put that file at:

```text
<Hand-Landmark-Tracker>/external/WiLoR/mano_data/MANO_RIGHT.pkl
```

The inference adapter uses this right-hand MANO model for both hands, reflecting
left-hand predictions back to the input image. A separate MANO_LEFT.pkl is not
required by this adapter. Registration/license acceptance and this download must
be completed by the model user; MAVIS does not obtain the asset on their behalf.

## Live preview

Once the runtime and assets are available:

```powershell
python -m mavis_track demo --model wilor --source 0 --wilor-root external/WiLoR
# Explicit CUDA and accelerated inference:
python -m mavis_track demo --model wilor --source 0 --wilor-root external/WiLoR --wilor-device cuda --wilor-fast
# Video instead of webcam:
python -m mavis_track demo --model wilor --source "C:/clips/hand.mp4" --wilor-root external/WiLoR
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

## Evaluation before promotion to main

1. Verify one reference RGB image against the pinned upstream demo: left/right
   labels, projected joints, mesh and camera translation.
2. Test a webcam and our infrared camera frames, including darkness, occlusion,
   motion blur, extreme wrist poses and two overlapping hands.
3. Compare MediaPipe/WiLoR on identical frames with labelled or independently
   measured reference data. Report joint/angle errors, detection misses,
   left/right swaps, temporal jitter and failure recovery separately.
4. Measure startup, steady-state latency p50/p95, frame age, throughput and peak
   GPU memory, separately for standard and fast mode.
5. Only then connect WiLoR's 2D projections to calibrated multi-camera
   reconstruction. Keep monocular estimates distinct from triangulated data;
   assess camera distortion and cross-view hand association explicitly.

WiLoR improves hand inference capabilities; it does not replace the equal-priority
whole-body evaluation described in the [model roadmap](mmpose-comparison.md).

## Licensing

WiLoR's upstream repository/model license is
[CC BY-NC-ND 4.0](https://github.com/rolpotamias/WiLoR/blob/main/license.txt).
MANO and Ultralytics have separate terms linked by upstream. These dependencies
and assets are not relicensed under MAVIS's MIT license. This branch distributes
our adapter only, not upstream source, checkpoints or MANO assets.
