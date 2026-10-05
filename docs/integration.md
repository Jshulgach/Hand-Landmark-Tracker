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
uncompressed and 256 fields.

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
The desktop face/pose/holistic modes use an additional MediaPipe interface;
the initial supported embedding API is hand tracking.
