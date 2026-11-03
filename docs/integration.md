# Integrating MAVIS

Install the base package and call `HandTracker.process` inside your own image
loop. The API owns its model and filters; your application owns capture, display,
threads, and scheduling. Use one tracker per sequential source. A tracker is
not safe for simultaneous calls from multiple threads.

```python
from mavis_track import HandTracker

with HandTracker(max_hands=2, source_id="camera-a", smoothing=True) as tracker:
    for timestamp, frame_bgr in my_frame_source():
        result = tracker.process(frame_bgr, timestamp=timestamp)
        for hand in result.hands:
            my_application.update(hand.track_id, hand.landmarks)
```

`my_frame_source` and `my_application` belong to your project. A runnable
OpenCV example is available in the repository at
`examples/01_basic_tracking/mavis_webcam.py`.

## Frame and result contract

Input is a nonempty three-channel uint8 NumPy image. BGR is the default;
use `color="rgb"` for RGB. Input arrays are not modified.
Set `mirrored=True` when the incoming image has been horizontally flipped,
so handedness labels account for MediaPipe's mirror convention.

Timestamps are finite, nonnegative, strictly increasing seconds. With no supplied
timestamp, MAVIS uses `time.perf_counter()`. Do not mix clocks within a tracker.
`FrameResult.timestamp_clock` is `source` for supplied timestamps or
`monotonic` for the default; source clock meaning is the caller's responsibility.

Each result owns its read-only arrays. `result.hands` is an empty tuple when no
hands are detected. Each hand contains:

| Field | Meaning |
| --- | --- |
| `track_id` | Short-lived association within this tracker/source |
| `handedness` | Left, Right, or Unknown |
| `handedness_score` | Model classification probability; not landmark accuracy |
| `landmarks`, `raw_landmarks` | (21, 3) image-normalized x/y and wrist-relative model z |
| `world_landmarks` | Optional (21, 3) hand-centered model estimates in meters |
| `angles` | 14 named geometric flexion angles, degrees; NaN for degenerate geometry |
| `angle_space` | `hand_world_meters` or `image_scaled` |

Image x/y use width/height normalization. For image-based angle fallback, x and
z are scaled by width and y by height before geometric angles are computed.
World estimates are uncalibrated model predictions; separate hands cannot be
combined as a global scene.

Association matches handedness and nearby wrists in adjacent frames. It resets
after disappearance, a long timestamp gap, or a large jump. Reordering detections
preserves IDs in ordinary cases. Crossings and ambiguous same-label detections
are not a guarantee of continuous identity.

`close()` is idempotent. A context manager closes on success or exceptions.
Create a new tracker after closing.

## Loading recordings

```python
from mavis_track import load_session

session = load_session("recordings/demo")
points = session.landmarks   # (frames, hand_slots, 21, 3)
present = session.valid      # (frames, hand_slots)
timestamps = session.time_vector
image_sizes = session.image_sizes  # Per-frame width/height, None in older files
```

Hand slots are storage positions; consult `track_ids` for identity. Missing
hands have ID -1 and NaN coordinates. Fractional sampling rates are preserved.
Numeric legacy single-hand files are expanded to one slot. Object arrays are
rejected; loading never enables pickle. Archives are limited to 256 MiB
uncompressed and 256 fields per archive. Long recordings use bounded chunks;
see the streaming workflow below.

## UDP and LSL

```python
from handtrack.io.broadcast import UDPBroadcaster

sender = UDPBroadcaster(coordinate_space="image_normalized",
                        timestamp_clock="source")
try:
    sender.send_landmarks(result.frame_id, result.timestamp, [
        {"hand_index": slot, "track_id": hand.track_id,
         "landmarks": hand.landmarks}
        for slot, hand in enumerate(result.hands)
    ])
finally:
    sender.close()
```

Send every frame, including an empty list when hands disappear. UDP packets
include schema version, coordinate space, clock label, and hand count; missing
numeric coordinates become JSON null. Configure the receiver to clear absent
hands. UDP itself provides no delivery guarantee.

LSL is optional: install `mavis-track[stream]`, or `.[stream]` for source development.
Construct `LSLBroadcaster(coordinate_space="image_normalized")` for frame API
coordinates. Its legacy default is calibrated meters for stereo applications.
LSL timestamps **must use `pylsl.local_clock()`**; do not pass video-relative,
Unix, or arbitrary source timestamps directly. Empty detections produce an
all-NaN sample. Fixed hand slots in the legacy LSL stream do not convey track IDs.

## Unity compatibility assets

The packaged Left/Right Hand Listener C# files are legacy bone-rotation CSV
receivers. They do not consume MAVIS's JSON landmark packets. A new Unity project
needs a receiver matching the JSON contract above; copying those legacy scripts
alone does not establish that integration.

## Compatibility

`handtrack.tracker.HandTracker` retains its source-owning legacy interface.
Prefer `mavis_track.HandTracker` for new embedding; it avoids capture and
GUI ownership and supplies explicit presence, timing, and identity metadata.
For combined single-person hand/body processing, use `HandBodyTracker` below.
Desktop face tracking remains a separate MediaPipe interface.


## Combined hand and body tracking

`HandBodyTracker` processes both hands and a single person's 33 body landmarks
in one MediaPipe Holistic inference. Your application supplies each frame; no
camera, window, or background worker is opened by this API. Holistic runs its
internal face model too, but this API does not expose face results. It is not a
multi-person tracker. Existing `HandTracker` behavior is unchanged. A runnable webcam example is
`examples/01_basic_tracking/mavis_hand_body.py`.

```python
from mavis_track import HandBodyTracker, QualityFlag

with HandBodyTracker(source_id="camera-a", visibility_threshold=0.5,
                     max_gap=0.5) as tracker:
    for timestamp, frame_bgr in my_frame_source():
        result = tracker.process(frame_bgr, timestamp=timestamp)
        # These observations come from the same pixels and timestamp.
        if result.body is not None:
            shoulders = result.body.landmarks[[11, 12]]
        if result.quality.body & QualityFlag.LOW_VISIBILITY:
            my_application.mark_body_uncertain()
        for hand in result.hands:
            my_application.update(hand.handedness, hand.landmarks)
```

The input, timing, read-only snapshot, sequential-use, and close conventions
match `HandTracker`. Missing hands produce an empty or shorter `hands` tuple;
a missing body produces `body=None`. No last-known region is emitted as a
current observation. Hand association IDs reset after disappearance, a gap
longer than `max_gap` seconds, or a wrist jump over 0.25 in normalized image x/y.
IDs do not identify people, and a person replacement can go undetected.

Body results contain `landmarks` (33, 3), `visibility` (33,), optional
`world_landmarks` (33, 3), and `landmark_names`. `BODY_LANDMARK_NAMES` exposes
the same names without creating a tracker. Body image z is hip-relative model
depth; body world estimates are hip-centered meters, labeled
`body_world_meters` in recordings. Hand image z is wrist-relative.
Holistic does not provide hand-world landmarks or handedness classification
scores: `hand.world_landmarks` is `None`, `hand.handedness_score` is NaN, and
hand angles use `image_scaled` geometry. Neither those angles nor body model
estimates are validated clinical measurements. Hand and body depths cannot be
concatenated into calibrated scene coordinates.

Labels and body indices describe anatomical subject sides. Set `mirrored=True`
only if the input pixels were horizontally flipped. In that case the API swaps
left/right hand labels and paired body indices; x/y remain coordinates of the
supplied image. Record uses this setting automatically with `--flip-frame`.
Optional smoothing applies only to hand image landmarks. Raw hand observations
and all body outputs remain available without smoothing.

### Tracking quality

`result.quality.left_hand`, `.right_hand`, and `.body` are `QualityFlag` bit masks.
Multiple conditions can be present together. Flags describe raw observations,
not filtered coordinates or calibrated measurement confidence.

| Flag | Value | Meaning |
| --- | --- | --- |
| `OK` | 0 | No implemented check triggered; accuracy is not established |
| `MISSING` | 1 | Region absent or its image landmarks are unusable |
| `LOW_VISIBILITY` | 2 | At least one body landmark falls below the visibility threshold, or visibility is invalid |
| `OUT_OF_FRAME` | 4 | At least one raw image x/y lies outside [0, 1] |
| `NONFINITE` | 8 | Invalid landmark shape, nonfinite coordinates, or invalid body visibility |
| `TEMPORAL_GAP` | 16 | Input timestamp interval exceeds `max_gap` |
| `TRACKING_RESET` | 32 | A previously observed hand starts a new association ID |

Missing or malformed image landmarks suppress that region. Low-visibility and
out-of-frame measurements remain available with flags. Invalid body-world
coordinates are omitted while usable image coordinates remain. Hand visibility
is not supplied by Holistic, so an `OK` hand is not a high-confidence hand.
The first frame is not a temporal gap. Gap flags compare source timestamps;
they do not measure exposure synchronization, inference latency, or the age of
a caller's image relative to wall time. Use body visibility per landmark when
only a subset of joints is relevant to an experiment.

### Synchronized recording, replay, and export

```bash
mavis-track record --mode hand-body --source 0 --frames 300 --session-name reach
mavis-track record --mode hand-body --source clip.mp4 --session-name reach-video --save-video
mavis-track replay recordings/reach
mavis-track export recordings/reach
```

`--visibility-threshold` defaults to 0.5; `--max-gap` defaults to 0.5 seconds.
Combined recording always allocates two hand slots; `--max-hands` applies only
to hand-only recording. `--no-kalman` disables hand smoothing. The model's
visibility output is not a calibrated error probability.

Combined numeric archives use NPZ schema 2. Long runs split these into a schema-3
chunked session; short runs retain a single `landmarks.npz`. Each row of the existing hand data,
`body_landmarks` (frames, 33, 3), `body_world_landmarks` (frames, 33, 3),
`body_visibility` (frames, 33), and `quality_flags` (frames, 3) uses the **same**
`time_vector` and source frame. Quality columns follow `QUALITY_REGION_NAMES`:
left hand, right hand, body. `body_valid` means image landmarks are present and
finite; it does not mean that visibility is adequate. Missing values use NaNs,
not zero coordinates or a previous pose. An unavailable world estimate is an
all-NaN frame. Quality thresholds, coordinate labels, and names are stored in
the archive; `session.json` additionally describes the flags and smoothing.

`load_session` validates the body shapes, visibility, quality masks and presence
consistency, and returns read-only arrays. Schema 0/1 hand sessions remain
readable. Older MAVIS versions reject schema 2 rather than silently ignoring
the body data. Replay draws the saved body and hands using the common timeline;
low-visibility body landmarks are hidden. `--save-video` adds both overlays.

Export retains the existing hand CSVs and adds `body_landmarks.csv` (one row per
body landmark/frame, including visibility, presence and world estimates) and
`tracking_quality.csv` (one row per region/frame with numeric and named flags).
The export manifest identifies the body coordinate space and quality thresholds.
Annotation video is constant-frame-rate; use the numeric timeline for analysis.

For exoskeleton validation, time-align these observations with the experiment's
telemetry/EMG clock explicitly. Recording one video stream does not establish
that external devices share its clock. Evaluate tracking error against an
independent reference and keep missing/uncertain measurements marked when
computing validation statistics.


## Long recordings and bounded-memory analysis

Recording buffers at most `--chunk-frames` observations (4096 by default,
1-16384 allowed). Below that threshold it writes the existing single NPZ format.
At the threshold it starts writing `chunks/000000.npz`, `000001.npz`, etc., each
using the established hand schema 1 or combined schema 2. `session.json` uses
schema 3 to describe chunk order, frame counts, global sampling rate and status.
All chunk timestamps retain the same source clock; hand IDs and quality flags
continue across boundaries.

```bash
mavis-track record --mode hand-body --source clip.mp4 --session-name long-run
mavis-track record --source 0 --frames 108000 --chunk-frames 4096 --session-name hour
mavis-track replay recordings/long-run
mavis-track export recordings/long-run
```

Replay and CSV export stream one chunk at a time. For custom analysis:

```python
from mavis_track import iter_session_chunks

for chunk in iter_session_chunks("recordings/long-run"):
    analyze(chunk.time_vector, chunk.landmarks, chunk.valid,
            chunk.body_landmarks, chunk.quality_flags)
```

The iterator also accepts older single-file sessions and yields once. Missing
or corrupt chunks, changed layouts, out-of-order timestamps and paths outside
the session directory are rejected. Each archive keeps the existing 256 MiB
uncompressed-size and numeric/pickle-free limits.

`load_session` still materializes a session as read-only arrays. For chunked
sessions it limits the resulting arrays to 256 MiB by default; use the iterator
for larger runs. `load_session(path, max_bytes=...)` can explicitly raise this
limit (`None` disables it) for trusted data that fits in available memory.

Each flushed chunk is written before an atomic manifest replacement. A normal
finish sets `complete: true`; Ctrl+C or a handled capture/model error saves the
remaining valid observations with `complete: false` and a reason. A process kill
or power loss can lose the current buffer, but the last committed manifest and
chunks remain readable when intact on disk. This is an application checkpoint,
not a guarantee against filesystem/hardware failure. Annotated video is finalized
on orderly stop; a force-killed video may require codec-specific recovery.

## Optional WiLoR hand inference

Source installations also expose `WiLoRTracker`, `WiLoRHandResult` and
`WiLoRFrameResult`, with `WiLoRMultiViewTracker` for shared-model
inference over ordered camera frames. The WiLoR API accepts caller-owned frames and returns
pixel projections, wrist-relative 3D joints, mesh vertices and geometric angles.
Its result types keep monocular model coordinates separate from MediaPipe and
calibrated reconstruction. See [setup, examples and validation status](wilor.md).
