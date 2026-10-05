# Hand Landmark Tracker public package audit

Baseline assessed on October 4, 2026 against commit `d0981e4` on `main`. Implementation follow-up is on local branch `codex/mavis-tracking-release`.

**Current assessment:** MAVIS 0.1.0 is published on PyPI and TestPyPI. The new public API, desktop paths, sessions, cleanup, calibration validation, optional dependencies, and release gates have been implemented. Downloaded public artifacts match the reviewed files and pass fresh Windows installation checks. See the follow-up below for verified results and remaining external checks. The original findings are retained as baseline evidence.

The tracking foundation works. A built, installed package detected hands in existing demo videos on Windows and Linux, including two hands on Windows. A Windows webcam opened, supplied a frame, and closed successfully. At the baseline, recording failed, tests could not collect completely, several application imports were broken, and session formats and resource cleanup were inconsistent.

The software audit is complete. Its applicable software findings have been addressed for the initial webcam/frame-API release scope. Native Mac and Linux desktop camera validation remains a release requirement; advanced hardware accuracy and unrestricted cross-camera hand association remain outside verified first-release claims.

## Scope and OptiTrack evidence

The audit covered installation, distribution contents, public imports, CLI and documentation, MediaPipe processing, session recording and export, replay, UDP and LSL payloads, webcam lifecycle, calibration identity, triangulation uncertainty, tests, and release workflows.

The author reports that OptiTrack works on Windows based on many previous tests and commits. That is recorded as historical Windows evidence. No OptiTrack system was available for a current hardware retest. Its absence does not prevent resolving the software findings or finishing a webcam release. No OptiTrack hardware was commanded during this audit.

## Baseline verified behavior

| Check | Result | What the evidence establishes |
| --- | --- | --- |
| Build wheel and source distribution | Passed | Both distributions build; the wheel is approximately 192 KiB. |
| Distribution metadata check | Passed | Both artifacts pass `twine check`; this does not establish runtime correctness. |
| Strict documentation build | Passed | MkDocs builds; individual commands can still be wrong. |
| Windows installation | Passed | Installed package and dependencies work in an isolated CPython 3.11.9 environment. |
| Linux installation | Passed | Built wheel installs in an isolated Ubuntu 24.04 WSL2 x86-64 environment using CPython 3.12.3. |
| Dependency consistency | Passed | Windows and Linux environments pass their dependency checks. Both nevertheless install competing OpenCV distributions, discussed below. |
| Installed imports outside the checkout | Mixed | Core tracker and existing stereo webcam GUI import. New MediaPipe GUI and calibration wrapper imports fail. |
| Real tracking on saved videos | Passed | Windows detects hands in 12/12 sampled mono demo frames, with up to two hands, and 11/11 readable HandDynamic samples. Linux detects hands in 5/6 mono demo samples. |
| Stereo demo video used as a single image | No detections | Windows produced no hands in 12 samples. This does not evaluate stereo triangulation or establish accuracy against ground truth. |
| Mode switching and cleanup | Passed | Real MediaPipe hands, face, pose, and holistic solutions process blank RGB frames, switch modes, and close. Positive face/pose tracking was not evaluated. |
| Windows live webcam | Passed | Camera 0 supplies a 640 by 480 frame and processes it. No hand was present in that check. Capture and detector were released. No image was saved. |
| UDP localhost delivery | Passed with gaps | A one-hand packet arrives with 21 landmarks. Empty detections send no packet and nonfinite values are emitted as `NaN`. |
| LSL payload construction | Passed with gaps | A mocked outlet receives 126 landmark channels, missing-hand NaNs, and the supplied timestamp. This was a payload test, not an end-to-end external receiver test. |
| Complete Windows test suite | Failed at collection | Four test modules cannot import missing modules. |
| Tests after excluding those four modules | 61 passed and 1 failed | Remaining failure requires a machine-local calibration file absent from a clean checkout. It is not evidence that historical OptiTrack operation failed. |
| Recording CLI | Failed | Both a real no-hand video and a simulated detected-hand result reproduce crashes. |
| Session export and replay | Mixed | One-hand CSV and drawing work. Two-hand CSV is malformed and replay drawing crashes. |

The four collection failures are `test_calibration_routing.py`, `test_optitrack_calibration_identity.py`, `test_optitrack_performance_paths.py`, and `test_webcam_discovery.py`. They reference absent `handtrack.applications._calibration_runtime`, `handtrack.cameras.optitrack.mocap_tracker`, or `handtrack.cameras.webcam` modules. The later failure is `test_optitrack_uses_core_backend_calibration_file` in `tests/test_optitrack_camera_startup.py`, which asserts that the local calibration file exists.

## Baseline platform readiness

| Platform | Current evidence | Remaining release validation |
| --- | --- | --- |
| Windows x86-64 | CPython 3.11 installation, installed imports, positive video tracking, webcam capture, and offscreen GUI construction tested. Dependency resolution also succeeds for 3.10 and 3.12. | User-facing demo, repair of findings, repeated start/stop, camera unplug/replug, and supported Python versions tested in CI. |
| Linux x86-64 | Ubuntu 24.04 WSL2 CPython 3.12 installation, positive video tracking, imports, diagnostics, and offscreen Qt construction tested. | Native Linux desktop camera access, display behavior, camera permissions, and unplug/replug. WSL results do not establish these. |
| Mac Intel | Binary-only dependency resolution succeeds for CPython 3.11. | Native installation, tracking, camera permission prompts, GUI, and cleanup. |
| Mac Apple Silicon | Binary-only dependency resolution succeeds for CPython 3.11. | Native installation and tracking without translation, camera permissions, GUI, and cleanup. |

The metadata currently advertises Python `>=3.10` without an upper bound. A CPython 3.13 Windows resolution fails because pinned MediaPipe 0.10.14 has no matching wheel. Its published files cover CPython 3.9 through 3.12; the project's own lower bound excludes 3.9. A reasonable initial declaration is Python 3.10 through 3.12, subject to runtime checks on the selected platforms. [MediaPipe 0.10.14 files](https://pypi.org/project/mediapipe/0.10.14/)

## Baseline findings

Priorities describe the public package impact: P1 blocks a dependable advertised workflow or needs a security correction; P2 affects reliability, integration, or feature claims.

### P1 Recording fails with and without a detected hand

`src/handtrack/applications/_record.py:152` treats the list returned by `_process_frame` as an object with `.landmark`. With a detected hand, it raises `AttributeError: 'list' object has no attribute 'landmark'`. With no hand, line 167 calls `.keys()` on `None`. These are separate reproduced failures. Define one frame result representation and make recording consume it, including empty frames and two hands. Require successful record, load, replay, and export of the same session before closing this finding.

### P1 The easiest demo and several public paths are disconnected

`src/handtrack/applications/mediapipe_gui.py:24` imports `MediaPipeModeTracker` and `TrackingMode` from `handtrack.tracker`, but that package does not export them. The error occurs on both Windows and Linux. `handtrack.applications.webcam_calibration` also imports the absent `_calibration_runtime` module. The CLI's webcam GUI routes to the older stereo application, whose startup loads a calibration file even for a new installation. `docs/demo.md:109` describes a `setup --run-calibration` command that the CLI does not implement.

Choose the supported module layout, repair exports and wrappers, and route the first demo to a single webcam mode that can start immediately. Make calibration a later step for measured stereo 3D. Diagnostics should exercise the requested application path; current `doctor --backend webcam` passes even when these application imports fail.

### P1 Loading a shared session can execute Python pickle data

`src/handtrack/io/_session_loader.py:116` and `src/handtrack/applications/_export.py:93` load sessions with `allow_pickle=True`. A harmless object-array session caused the loader to execute a sentinel file write during this audit. Similar pickle-enabled paths exist in calibration inspection and loading. Public numeric files should use a numeric/string schema with pickle disabled, plus shape, type, size, and finite-value validation. Handle any necessary old trusted files through an explicit migration path. NumPy documents the execution risk when loading pickled object arrays. [NumPy load documentation](https://numpy.org/doc/stable/reference/generated/numpy.load.html)

### P1 The test suite cannot provide a release gate

The four missing-module collection failures prevent a complete run. Another test requires an ignored, machine-local calibration file rather than a portable fixture. Repair the module migration and use small generated calibration fixtures for software tests; keep physical device tests separate and explicitly selected. `.github/workflows/publish-pypi.yml` currently gates publishing on build and metadata checks, without a dependency on passing tests. Validate the installed wheel in a separate consumer directory and require those results before publishing.

### P1 Calibration does not preserve and validate camera identity

`src/unity_hand_tracking/webcam/mocap_tracker.py:173` replaces discovered source indices with sequential indices. Calibration saving likewise writes `list(range(num_cameras))`; loading consumes arrays by position and does not validate saved identities or image size. A camera-count mismatch only prints a warning and startup continues. Detection also uses camera 0's width and height for every camera at line 293.

Preserve actual source identity, camera-specific dimensions, coordinate units, and a schema version. Reject incompatible calibration before tracking. Wait for valid initial frames before choosing resolution. The robust calibration solver has useful synthetic tests, but the webcam CLI still uses its separate older solver; passing those tests does not validate the CLI's complete calibration path.

### P2 Two-hand export and replay have incompatible shapes

For a `(frames, 2, 21, 3)` session, `_export.py:20` makes a header with 8 columns while each data row has 128. `_replay.py:19` attempts to unpack an entire hand array as one coordinate triple and raises `ValueError`. Adopt one documented hand dimension and represent absent hands explicitly. Test one-hand, two-hand, empty, and interrupted sessions through the same pipeline. These failures are independent of the recording crash.

### P2 Cleanup after errors and camera loss is incomplete

`HandTracker` opens capture before constructing MediaPipe and has no public `close()` or context manager. A simulated detector initialization exception leaves capture unreleased. A simulated calibration exception in stereo GUI startup leaves its tracker uncleaned until the window closes. The GUI catches the exception but does not undo startup.

`CameraWorker.run()` retains its last valid image after read failures, with no capture timestamp or stale-frame status. A deterministic probe returned that image after two failed reads. Stop/join also prints successful cleanup without checking whether a worker is still alive. Make startup transactional, release every resource on exceptions, expose deterministic close semantics, and stop publishing stale frames after a device disconnects.

### P2 Hand identity and data meaning are not a stable contract

The core smoother assigns state by detection-list position in `_hand_tracker.py:202`; that position can change and is not a persistent identity. Missing detections are represented as zero coordinates without a validity mask. Multi-camera matching currently groups the first hand in each view, so it should remain a one-hand feature until association is implemented.

A public result should carry handedness, validity, timestamp, source identity, and an explicit coordinate space. Keep image-normalized coordinates, MediaPipe model world coordinates, and calibrated multi-camera coordinates distinct. Model world landmarks use meters but are centered on the hand; they do not establish the hand's calibrated global position. [MediaPipe hand landmark definitions](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)

### P2 Time and streaming need one documented convention

`SessionLoader` converts 29.97 Hz to 29 Hz. Recording assigns `frame_index / fps` timestamps even for live capture. Webcam workers supply independent latest images without capture times, so triangulation cannot assess frame age or inter-camera skew. Preserve fractional rates and use actual monotonic capture times for live data.

UDP emits no empty-hand packet, leaving a consumer to infer disappearance from its own timeout. It also serializes `NaN`, which strict JSON consumers reject. Define empty-frame behavior, schema version, units, hand identity, and finite-value handling. LSL declares landmark units as meters and forwards caller values and timestamps unchanged; the API needs to enforce or explicitly document that contract. `LSLBroadcaster.close()` currently inherits the base no-op, so it also needs deliberate resource release.

### P2 Joint angles need anatomical definitions

`compute_all_joint_angles` uses every consecutive index triple, including triples that cross from one finger to another, and returns integer keys 1 through 19. This differs from the named angle schema used by the GUI and streams. Core angle logging is commented out even when `save_angles=True`. Select canonical anatomical triples and names, document angle direction and coordinate-space requirements, and report invalid angles for missing or degenerate landmarks. Do not present existing results as validated anatomical measurements.

### P2 Camera placement uncertainty is not a calibrated error estimate

`src/handtrack/processing/_camera_geometry.py:236` uses the first three columns of the algebraic DLT matrix as the pixel-noise Jacobian. Multiplying every projection matrix by 10 leaves the image projections unchanged but reduces reported covariance by a factor of 100. A finite-difference pixel projection Jacobian at two meters also gives four times the implemented covariance in a synthetic stereo example. Replace this propagation with the pixel projection Jacobian and verify it against perturbation experiments before presenting the resulting placement plots as quantitative uncertainty. Existing finiteness and monotonicity tests do not establish this property.

### P2 Installation carries avoidable dependencies and release content

GUI, LSL, and data-analysis packages are mandatory in `pyproject.toml` even though their extras repeat them. Frame processing should be usable without installing the GUI or research stack. Both tested environments installed `opencv-python` and MediaPipe's `opencv-contrib-python`; both own the `cv2` namespace. Select one compatible distribution. [OpenCV package guidance](https://pypi.org/project/opencv-python/)

PyQt uses GPL v3 or a commercial license. Make the GUI dependency and its licensing visible when supporting downstream projects, and deliberately choose the GUI binding/licensing strategy. This finding identifies a dependency choice; it is not a determination that this project or a downstream project violates a license. [Riverbank licensing description](https://www.riverbankcomputing.com/software/pyqt/intro)

The built wheel and source distribution contain no `.dll`, `.pyd`, or `.lib` files. They also omit Unity C# listener assets and `THIRD_PARTY_NOTICES.md`. The Git repository still tracks SDK headers, documentation, samples, and native binaries despite the notice saying those are not distributed. The example-download workflow removes binaries but still copies SDK headers and documentation within the example tree. Reconcile the intended distribution boundary, notices, and curated example allowlist before releasing downloads. This is a confirmed inventory/notice contradiction, not a conclusion about SDK redistribution rights.

## Baseline integration assessment

For a standalone webcam window, the README's `HandTracker(source=0).run()` is a short integration, but it owns capture and the display loop. It is awkward inside an application that already owns its camera, GUI, game loop, or recorded frames.

The newer frame processor already supports the better shape of integration. This exact current import and usage worked from an installed package outside the checkout on Windows and Linux:

```python
from handtrack.tracker._mediapipe_modes import MediaPipeModeTracker

with MediaPipeModeTracker("hands") as tracker:
    result = tracker.process(rgb_frame)
    for hand in result.hands:
        points = [(p.x, p.y, p.z) for p in hand.landmark]
```

This is an internal module path and its result exposes MediaPipe objects. It should not be advertised as a stable public contract yet. Export a small documented tracker API, accept caller-owned frames, return a package-owned result, and keep drawing, capture, storage, and streaming optional. Then a downstream project can integrate in a few lines without handing control of its application to this package.

## A focused MAVIS first release

The chosen distribution name is **`mavis-track`**, under the MAVIS brand, Motion Analysis and Visual Interaction Suite. PyPI's public API returned no project record for `mavis-track` when checked on October 4, 2026. This check does not reserve the name; acceptance is confirmed when publishing. [PyPI project check](https://pypi.org/pypi/mavis-track/json)

The recommended Python import name is `mavis_track`, with `mavis-track` as the main command. These names have now been adopted. The original `handtrack` imports and `handtracker` command remain as compatibility interfaces.

The exact PyPI name `mavis` already belongs to another project; `handtrack` and `handtracker` also have existing unrelated listings. The selected name avoids those distribution-name conflicts. [Existing MAVIS listing](https://pypi.org/project/mavis/), [handtrack listing](https://pypi.org/project/handtrack/), [handtracker listing](https://pypi.org/project/handtracker/)

Recommended first-release scope is a single-webcam hand demo, a caller-owned-frame API, reliable recording/replay/export, and optional UDP/LSL support. Preserve OptiTrack and measured stereo workflows as explicitly advanced features with their own prerequisites. Face and pose modes can follow when their positive tracking behavior and user flows are validated.

The most useful order of work is:

1. Adopt the selected `mavis-track` distribution name, settle the proposed import and command names, and define a small frame-result contract.
2. Repair the collection failures, public imports, and immediate single-webcam demo.
3. Fix recording, two-hand sessions, pickle loading, timestamps, and cleanup together so every command shares one data contract.
4. Separate optional dependencies and curate release assets and notices.
5. Test built wheels on Windows, Linux, Intel Mac, and Apple Silicon Mac across the declared Python range; make publishing depend on those checks.
6. Run a fresh-user install and short physical camera checklist on each desktop OS. Record first launch, permissions, no camera, no hand, two hands, repeated start/stop, and unplug/replug.

OptiTrack access is unnecessary for steps 1 through 5 or the webcam portion of step 6. A future OptiTrack session can add current hardware evidence after these software corrections.

## Implementation follow-up — October 4, 2026

The release candidate uses `mavis-track` for distribution/CLI and `mavis_track`
for the public import. Python support is explicitly 3.10–3.12, 64-bit.
Version 0.1.0 was published to PyPI and TestPyPI on October 4, 2026. Both
indexes' downloaded wheel and source archive hashes match the reviewed local
artifacts. Fresh Windows installs passed metadata, commands, module imports,
and API/record/replay/export checks; the production GUI extra also passed
dependency and Qt initialization checks.

The pre-publication name change updates package metadata, imports, commands,
calibration storage, documentation, examples, the lockfile, and publishing links.
After the rename, all 103 tests pass on Windows and Linux. The rebuilt wheel's
metadata, installed command, module entry point, and API/record/replay/export
checks pass outside the source checkout on both systems.

A live Windows webcam check of the installed renamed package also passed on
October 4. The native Qt hands preview processed 405 frames at 640x480: 37
contained one hand and 368 contained two hands. Public landmark, world-landmark,
and angle values were finite, with no processing errors. The preview released
and reopened the camera successfully, then closed both tracking models and
the capture. Three additional worker start/stop cycles processed 122 frames,
including 49 empty-hand results; every worker, camera, and model was released.
Physical unplug/replug behavior remains a separate unverified device check.

| Workstream | Implemented behavior and evidence |
| --- | --- |
| Downstream API | Caller-owned BGR/RGB frames; owned read-only arrays; empty hand tuples; explicit clocks, image sizes, handedness, and short-lived IDs; context-managed cleanup; type annotations. |
| First demo | `mavis-track demo` requires no calibration. Real Qt offscreen preview switched hands/face/pose/holistic modes, restarted its source, and closed its capture/model. Positive face/pose accuracy remains unverified. |
| Recording and analysis | One/two-hand numeric sessions with presence masks, IDs, clocks, fractional rates, optional hand-world estimates, and angle coordinate labels; timed replay and rectangular CSV exports. Blank-video consumer round trips passed on Windows and Linux. A 60-frame real video recorded 80 present hand slots, including two hands, and produced annotated video and CSVs. |
| Archive safety | Pickle/object arrays, invalid shapes/metadata, duplicate fields, oversized archives, and misleading huge array headers are rejected. The harmless pickle sentinel test confirms no payload execution. |
| Camera lifecycle | Startup failures and model errors release resources. Immutable camera buffers expire after 250 ms; receipt-time skew over 50 ms excludes views from triangulation. Three native Windows webcam start/stop cycles passed at 640x480. These are host receipt times, not exposure synchronization. |
| Calibration | Counts, source indices or supplied UIDs, dimensions, and matrix validity are checked. Failed candidates preserve prior calibration. Every fifth capture is reserved from model fitting; reconstructed-board gates check shape, scale, and planarity. Synthetic integration passed with seven reserved views. Source indices still do not establish physical identity after replugging; the SDK binding's physical UID capability is unverified. |
| Angles and geometry | Fourteen named geometric finger flexion triples replace cross-finger segments; degenerate geometry yields NaN. Initial filtering no longer pulls measurements toward zero. Covariance matches an independent finite-difference pixel Jacobian and a noisy triangulation simulation. Physical precision and clinical angle accuracy are unverified. |
| Streaming | Hand-loss UDP packets and JSON nulls were received on loopback. LSL loopback received 126 channels with explicit normalized units and all-NaN hand-loss samples using local-clock timestamps. LSL discovery used a process-local machine/loopback configuration; external recorder/network integration is still unverified. |
| Installation/content | Fresh base installs work without Qt, pandas, or LSL; one OpenCV distribution is selected. Notices and legacy Unity C# assets are included; SDK assets and runtime data are excluded from wheel/sdist/example downloads. Core examples are explicitly curated. 302 previously tracked SDK/native files were removed from Git's index while preserving local copies; old Git history remains. Unity CSV listeners do not consume MAVIS JSON packets. |
| Release gates | Complete tests pass on Windows CPython 3.11 and Ubuntu/WSL CPython 3.12. CI now covers Windows/Linux/Intel Mac/Apple Silicon Mac with Python 3.10–3.12, including a built-wheel consumer test before desktop extras. Both publishing workflows depend on that matrix. Remote CI has not yet run for this branch. |

The full software suite currently passes **103 tests on Windows and Linux**.
Strict documentation build, scoped lint/type checks, wheel/sdist build, metadata
checks, and curated example creation pass. The built artifacts are approximately
0.22 MB and contain notices and Unity compatibility scripts with no SDK binaries,
headers, or datasets. Installed-wheel checks ran outside the source checkout.

Integration now takes a few lines: `from mavis_track import HandTracker`, open
a context manager, and call `tracker.process(frame)`. Capture, display, scheduling,
and your application's event loop remain caller-owned. The base package still has
MediaPipe's native/transitive dependencies; it is not a zero-dependency install.

### Remaining release validation

1. Run the configured remote CI matrix, particularly both Mac architectures and
   Python 3.10. Its configuration alone is not a completed platform check.
2. Complete native Mac/Linux camera permissions, unplug/replug, codec, and display
   checks. Windows camera and offscreen checks do not establish other desktops.
3. Configure trusted publishing for future automated releases and deploy the
   updated documentation after merging the release source. The first release
   and its fresh TestPyPI/PyPI install checks are complete.
4. Keep advanced limits explicit: one hand across calibrated views, no verified
   hardware exposure synchronization, no clinical-angle/absolute-precision claim,
   no current OptiTrack retest, and no new Unity JSON receiver validation.

These remaining checks do not require an OptiTrack system for the ordinary webcam
package. Test evidence is retained under `%TEMP%/handtrack-audit-20261004` and
`/tmp/mavis-release-20261004*`; source artifacts do not depend on those paths.

## Audit closure and retained evidence

The software assessment is complete with the findings above recorded. A release should close the findings applicable to its advertised scope and obtain the native desktop checks in the platform table. Current uncertainty is explicit: native Mac behavior, native Linux camera/display behavior, synchronized physical multi-camera accuracy, positive face/pose tracking, external Unity/LSL receiver integration, and a current OptiTrack retest were not established.

Windows probe results and distributions are retained under `%TEMP%/handtrack-audit-20261004`; Linux installation and probe logs are under `/tmp/handtrack-audit-20261004-linux*`. These are temporary audit evidence, not package dependencies. Implementation now includes the source, regression tests, documentation, packaging, and workflow changes described below.
