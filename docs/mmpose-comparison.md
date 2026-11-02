# MAVIS and MMPose: accuracy and versatility audit

Reviewed 2 November 2026. Hand and whole-body tracking have equal priority.
This is a source and architecture audit, not an accuracy benchmark. Neither a
passing software suite nor a plausible skeleton establishes physical accuracy.

## Recommendation

Keep MAVIS as the capture, calibration, measurement and operator application.
Add interchangeable estimators, using MediaPipe as the baseline, RTMPose Hand
for a second hand baseline, and RTMW for whole-body evaluation. Use the full
MMPose stack for reference experiments and training; evaluate lighter deployment
runtimes separately. WiLoR is an additional hand reconstruction experiment, not a
replacement for the whole-body lane.

MMPose is a much broader model and training toolbox. MAVIS already connects
Flex 13 cameras, calibration, geometric angles, explicit loss signals, LSL/UDP,
and bounded recording/replay. Those workflows are useful integration work, but
we have not demonstrated better landmark or angle accuracy than MMPose.
[MMPose project](https://github.com/open-mmlab/mmpose/).

## What overlaps, and what differs

| Area | Current MAVIS | MMPose ecosystem | Consequence |
|---|---|---|---|
| Frame/video/webcam inference | Public frame API and live tools | Inferencer accepts images, arrays, videos and webcams | These inputs alone are not a differentiator |
| Hands | MediaPipe 21-point hands; advanced calibrated multiview reconstruction | Multiple hand models and datasets | Benchmark comparable joints and visibility conditions |
| Whole body | Single-person Holistic: body 33 plus two hands; face not exposed by the combined API | Body, hand, face and whole-body model families | Add schemas and person ownership rather than assuming interchangeable indices |
| Multiple people | Combined API is single-person; advanced stereo matching assumes one hand | Multi-person pipelines and detectors | Need person/hand association across time and cameras |
| Physical measurement | Camera calibration, triangulation and angle calculations | Model-dependent 2D/3D predictions | Learned monocular depth and calibrated world coordinates need separate labels |
| Runtime application | GUI, capture adapters, quality flags, streams, recording/replay | Inferencer, visualization, training and deployment tools | Keep acquisition and downstream contracts independent of model libraries |

MMPose's inferencer already exposes per-instance keypoints and scores and can
save predictions and visualizations. An optional MAVIS backend should convert
those outputs into MAVIS observations rather than duplicate the inferencer.
[Inference guide](https://mmpose.readthedocs.io/en/latest/user_guides/inference.html).

Do not describe MMPose as exclusively monocular: its documented dataset history
includes multiview work such as Panoptic/VoxelPose. That does not supply a ready
replacement for our particular Flex 13 capture and commissioning workflow.
[3D body datasets](https://github.com/open-mmlab/mmpose/blob/main/docs/en/dataset_zoo/3d_body_keypoint.md).

## Findings in this checkout

1. **The estimator is tightly coupled to capture applications.** Single-camera
   mode switching and advanced multiview tracking use different result paths.
   Introduce an estimator interface below both; keep camera selection separate.
2. **Hand and body coordinates differ.** Holistic body world estimates are
   hip-centered; standalone hand world estimates are hand-centered. Combined
   hand angles currently use image-scaled coordinates. Neither is a shared
   laboratory coordinate system. Preserve origin, axes, units and provenance.
3. **Multiview confidence is partly a placeholder.** The advanced detector
   constructs per-landmark confidence values of 1.0. A landmark threshold cannot
   reject bad joints when every value is identical. Handedness probability is
   not a landmark accuracy estimate. Mark unavailable confidence as unavailable,
   then evaluate score calibration against labeled errors.
4. **Multiview hand association is restrictive.** The advanced pipeline takes
   one hand per view and rejects ambiguous multiple-hand observations. Add
   person ownership, left/right labels, epipolar consistency and temporal
   association before claiming robust two-hand or multi-person reconstruction.
5. **Input detail is discarded.** The OptiTrack hand path resizes full camera
   frames to 640 pixels wide before inference. Compare that baseline with a
   coarse body pass followed by hand crops from the original image. Measure
   accuracy and total latency, including detection and crop transforms.
6. **Frame freshness is not exposure synchronization.** Current gates use host
   receipt times: 250 ms maximum age and 50 ms inter-camera skew. Extend the SDK
   binding to expose hardware timestamps/frame identifiers and reliable camera
   UIDs where supported. Establish actual exposure synchronization separately.
7. **Reprojection settings need one source of truth.** The GUI final error gate
   can change while RANSAC still uses a default threshold captured in a helper
   signature. Thread an explicit runtime setting through both stages, with
   units and validation, before comparing triangulation algorithms.
8. **Angles need independent validation.** Fourteen geometric finger angles
   are useful measurements, but three-point angles are not a complete anatomical
   joint model. Define neutral pose, axes, thumb conventions and uncertainty;
   validate each reported degree of freedom against an appropriate reference.
9. **Quality flags are operational signals.** Presence, visibility, stale
   frames and dropped tracks help consumers detect unusable observations. They
   are not calibrated probabilities that a joint or angle is correct.
10. **Current benchmarking mostly measures speed.** Add labeled accuracy and
    failure-rate evaluation. Keep software regression tests separate from
    participant and camera measurement validation.

These findings refer to `mavis_track/tracker.py`, `mavis_track/body.py`, and the
OptiTrack `mocap_tracker.py`/GUI path in this checkout. They are proposed work,
not claims that all of these issues have already been corrected.

MediaPipe documents handedness as a left/right classification score and provides
hand-centered model world estimates. Those semantics should survive any
adapter; do not reinterpret them as calibrated joint confidence or global 3D.
[MediaPipe Hands](https://chuoling.github.io/mediapipe/solutions/hands.html).

## Equal-priority model shortlist

| Lane | First comparison | Later reference | Why |
|---|---|---|---|
| Hands | MediaPipe Hands versus RTMPose Hand5 21-point model | HRNet/RHD reference; InterNet for interacting hands | Establish common-joint 2D error before adding more reconstruction machinery |
| Whole body | MediaPipe Holistic versus RTMW 133-point whole body | RTMO 17-point body for crowded scenes; RTMW3D research comparison | Cover hands, body and ownership while separating body-only and monocular 3D tasks |

The RTMPose project lists Hand5 hand checkpoints and RTMW whole-body models.
Its hand offering is labeled alpha; treat that as an experimental candidate.
Published model throughput is not a measurement of this laptop's four-camera
pipeline. [RTMPose models](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose).

The hand model zoo also supplies alternative architectures and datasets; retain
one slower reference model if it helps distinguish detector failures from pose
estimation failures. [2D hand model zoo](https://mmpose.readthedocs.io/en/latest/model_zoo/hand_2d_keypoint.html).
InterNet addresses an interacting-hand 3D setting with different assumptions;
it should not be scored as though its outputs were calibrated laboratory
coordinates. [3D hand model zoo](https://mmpose.readthedocs.io/en/latest/model_zoo/hand_3d_keypoint.html).

RTMO is a one-stage multi-person body estimator, not a hand/whole-body
replacement. [RTMO project](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo).
RTMW3D is a monocular whole-body 3D candidate; root alignment and physical scale
must be reported explicitly. [RTMW3D project](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose3d).

## Integration design

Separate configuration into capture backend, estimator/model, runtime and device.
For example, `optitrack` describes acquisition, while `rtmpose-hand` describes
inference. Existing commands and frame APIs should retain their defaults.

A canonical observation should contain schema/joint names; camera, person and
track identifiers; timestamps with clock provenance; image dimensions; points,
validity masks and available scores; coordinate origin/axes/units; estimator and
checkpoint identity; and optional 3D uncertainty. Represent missing points with
explicit validity, never invented zeros. Keep measured, reconstructed and
model-inferred positions distinguishable. Maintain owned, read-only arrays.

Use explicit mappings for common anatomical joints. A 133-point whole-body
skeleton is not a 33-point body skeleton with a different array length. Preserve
unmapped face/foot/hand joints in their native schema rather than silently
throwing them away. Associate hands with the correct person and camera track.

Keep inference off the Qt thread. Share model resources across camera requests,
measure batching and frame age, and bound queues. Avoid loading a large model
once per camera. For multiview use, return original-image 2D coordinates with
crop transforms reversed; triangulate those coordinates with actual calibration.
Do not substitute monocular predicted depth for a measured cross-camera ray.

The full MMPose installation uses PyTorch, MMEngine and MMCV; some pipelines also
need MMDetection. CPU installation is possible, but that does not establish
usable live latency. Pin a tested stack in an isolated research environment.
[Installation guide](https://mmpose.readthedocs.io/en/latest/installation.html).

RTMLib offers hand/body/whole-body inference through lighter runtimes including
ONNX Runtime and OpenVINO. On the current Intel laptop, start with an ONNX CPU
benchmark and evaluate OpenVINO only after device compatibility checks.
[RTMLib](https://github.com/Tau-J/rtmlib).
Its current packaging requests both OpenCV distributions; resolve that overlap
before installing it into MAVIS's single-OpenCV environment.
[RTMLib dependencies](https://github.com/Tau-J/rtmlib/blob/main/pyproject.toml).

## Benchmark protocol

Give both lanes the same attention, participant/session coverage and reporting.
Use RGB webcam footage and actual Flex 13 grayscale/IR footage separately.
Replicating grayscale into three channels does not remove a training-domain gap.
Hold camera placement, exposure and task conditions fixed when comparing models.

Hands: open/closed poses, finger isolation, thumb opposition, self-occlusion,
objects, crossed/touching hands, rapid motion, distance and edge-of-frame crops.
Whole body: seated and standing reaches, walking, partial bodies, occlusion,
multiple people and hand-to-person ownership. Include relevant clothing,
lighting, body sizes and movement patterns. Split training/evaluation by person
and session, not adjacent video frames. Store consented benchmark data outside Git.

| Measurement | Required reporting |
|---|---|
| 2D landmarks | Pixel error and suitable PCK/OKS on a documented common-joint subset; visibility strata |
| 3D landmarks | Absolute, root-relative and alignment-adjusted errors separately; units and reference frame |
| Angles | Per-joint bias and absolute error, reference convention, repeatability and useful range |
| Reliability | Detection coverage, dropout duration, reacquisition, identity switches and ownership errors |
| Temporal behavior | Static jitter, movement lag and smoothing effects; prediction versus observation |
| Runtime | Warm-up, p50/p95 end-to-end latency, per-camera delivered FPS, frame age/skew and peak memory |

Use independent labeled or instrumented ground truth appropriate to each claim.
Exoskeleton encoders measure the device; establish device-to-anatomy alignment
before treating them as human joint-angle ground truth. Low reprojection error
alone cannot exclude correlated model bias. Smooth output alone can hide lag.
Report confidence intervals across people/sessions and failure cases, not only
means over successful frames. Predefine acceptable error and coverage for the
intended research measurement before choosing a winner.

## Delivery order

1. Freeze MediaPipe baselines and collect paired hand/body benchmark clips.
2. Add estimator-neutral observations, score semantics and coordinate tests.
3. Integrate RTMPose Hand and RTMW in parallel-priority evaluation lanes.
4. Fix synchronization evidence, multiview association and threshold consistency.
5. Validate joint angles and score/uncertainty calibration on independent data.
6. Optimize crops, batching and deployment only after accuracy/failure results.
7. Fine-tune on representative failure cases if licensing and dataset coverage allow.

Select a model per task using accuracy, availability and latency together. A
single replacement need not win both lanes. Preserve the working MediaPipe path
while experiments mature. MMPose's latest published release was v1.3.2 on
12 July 2024 at review time; pin actual versions rather than treating the main
branch and release as interchangeable. [Releases](https://github.com/open-mmlab/mmpose/releases).
Review algorithm/checkpoint/dataset terms individually; project-level Apache
licensing does not describe every component.
[License notes](https://github.com/open-mmlab/mmpose/blob/main/LICENSES.md).
