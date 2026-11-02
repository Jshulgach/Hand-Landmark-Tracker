# Hand Landmark Tracker public package audit

Baseline assessed on October 4, 2026 against commit `d0981e4` on `main`. Implementation follow-up is on branch `codex/mavis-tracking-release` and [pull request #1](https://github.com/Jshulgach/Hand-Landmark-Tracker/pull/1).

**Current assessment:** MAVIS 0.1.0 is published on PyPI and TestPyPI. The new public API, desktop paths, sessions, cleanup, calibration validation, optional dependencies, and release gates have been implemented. Downloaded public artifacts match the reviewed files and pass fresh Windows installation checks. Remote software validation passes on Windows, Linux, Intel Mac, and Apple Silicon Mac with Python 3.10–3.12. See the follow-up below for verified results and remaining external checks. The original findings are retained as baseline evidence.

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
| Release gates | The corrected remote matrix passes all 12 OS/Python combinations: Windows, Linux, Intel Mac, and Apple Silicon Mac with Python 3.10–3.12. Each passes base-wheel consumer checks, desktop-extra dependency checks, and 115 software tests. The quality job also passes lint/type checks, strict docs, build/metadata checks, and curated examples. Both publishing workflows depend on the complete matrix. |

The original software suite passed **103 tests on Windows and Linux**. Release
publication and strict dependency-check regression tests bring the current local
Windows suite to **115 passing tests**, also passing in each remote platform job.
Strict documentation build, scoped lint/type checks, wheel/sdist build, metadata
checks, and curated example creation pass. The built artifacts are approximately
0.22 MB and contain notices and Unity compatibility scripts with no SDK binaries,
headers, or datasets. Installed-wheel checks ran outside the source checkout.

Integration now takes a few lines: `from mavis_track import HandTracker`, open
a context manager, and call `tracker.process(frame)`. Capture, display, scheduling,
and your application's event loop remain caller-owned. The base package still has
MediaPipe's native/transitive dependencies; it is not a zero-dependency install.

### Remote CI follow-up

The [first remote matrix](https://github.com/Jshulgach/Hand-Landmark-Tracker/actions/runs/37256806112)
passed Windows and Linux on Python 3.10, 3.11, and 3.12, plus the quality checks.
Mac Intel installed and ran the public tracker and recording command, but its
video backend rounded the generated 29.97 Hz clip to 30 Hz. The consumer test now
checks preservation of the actual source-reported rate; fractional-rate behavior
remains covered by portable session tests.

Apple Silicon installed the universal2 MediaPipe 0.10.14 wheel, but `pip check`
rejected its incorrect internal Intel-only tag. Inspection of the official wheel
confirmed both Intel and ARM native binary slices. CI accepts only this exact
version/platform warning and requires native model initialization and processing
before continuing. Additional dependency errors and runtime failures still fail
the job. The [upstream metadata issue](https://github.com/google-ai-edge/mediapipe/issues/6030)
also affects later MediaPipe releases; no runtime dependency or published 0.1.0
artifact was changed to address these validation issues.

The [corrected remote matrix](https://github.com/Jshulgach/Hand-Landmark-Tracker/actions/runs/37257375975)
completed successfully for commit `3374ca9`: all 12 platform jobs and the quality
job passed. Native Apple Silicon tracking ran without translation. Mac validation
includes installed frame processing, video record/replay/export, and offscreen Qt
lifecycle tests. It does not establish camera permissions or visible desktop
behavior. An added positive-inference check exposed zero detections on Intel Mac
when processing the screen recording containing rendered hands and a small
webcam inset. The follow-up uses 12 frames of the existing `HandDynamic.mp4`, with
a real hand filling the camera view, and retains decoded-frame diagnostics on
failure. The [positive-inference matrix](https://github.com/Jshulgach/Hand-Landmark-Tracker/actions/runs/37258294168)
completed successfully for commit `21cb5f7`: all 12 platform/Python jobs passed
positive hand inference, installed recording/replay/export, desktop extras, and
115 tests; the quality job passed as well. Intel Mac Python 3.11 detected one hand
in all 12 frames using its native AVFoundation decoder. The screen-recording
result remains an example of detection varying with input and platform; no
universal detection or measurement-accuracy guarantee is implied.

### Remaining release validation

1. Complete native Mac/Linux camera permissions, unplug/replug, and display
   checks. Windows camera and offscreen checks do not establish other desktops.
   Generated MJPG recording/replay and the native video backend are covered by CI;
   broader user-camera/codec combinations still need device checks.
2. Configure trusted publishing for future automated releases and deploy the
   updated documentation after merging the release source. The first release
   and its fresh TestPyPI/PyPI install checks are complete.
3. Keep advanced limits explicit: one hand across calibrated views, no verified
   hardware exposure synchronization, no clinical-angle/absolute-precision claim,
   no current OptiTrack retest, and no new Unity JSON receiver validation.

These remaining checks do not require an OptiTrack system for the ordinary webcam
package. Test evidence is retained under `%TEMP%/handtrack-audit-20261004` and
`/tmp/mavis-release-20261004*`; source artifacts do not depend on those paths.

## Audit closure and retained evidence

The software assessment is complete with the findings above recorded. A release should close the findings applicable to its advertised scope and obtain the native desktop checks in the platform table. Current uncertainty is explicit: native Mac/Linux camera and visible display behavior, physical unplug/replug, synchronized physical multi-camera accuracy, positive face/pose tracking, external Unity/LSL receiver integration, and a current OptiTrack retest were not established.

Windows probe results and distributions are retained under `%TEMP%/handtrack-audit-20261004`; Linux installation and probe logs are under `/tmp/handtrack-audit-20261004-linux*`. These are temporary audit evidence, not package dependencies. Implementation now includes the source, regression tests, documentation, packaging, and workflow changes described below.


## Maintenance follow-up — October 30, 2026

Local implementation on `codex/combined-hand-body` adds long-session storage,
responsive preview ownership, and repository organization. These changes are
not a claim of a newly published package or a physical hardware accuracy test.

- Recording buffers at most 4096 frames by default, writes bounded schema-1/2
  NPZ chunks and atomic schema-3 manifest checkpoints, and preserves completed
  observations after Ctrl+C or handled model/capture errors. Replay/export use
  `iter_session_chunks`; whole-session materialization retains a default 256 MiB
  result limit with an explicit override for trusted data.
- A synthetic 100,000-frame combined run (55.56 source minutes at 30 FPS) wrote
  25 chunks containing 268.44 MiB of uncompressed numeric fields and headers.
  Incremental loading recovered all frames. The traced writer allocation peak
  was 13.88 MiB for this repeated-array synthetic fixture, not a measurement of
  total process/model RAM or participant-video tracking performance.
- Live GUI capture, model initialization/inference, switching, and native cleanup
  now run on one worker thread. The UI consumes a single latest-frame slot and
  waits asynchronously for stop/close. Tests gate native initialization/read calls
  to verify event-loop progress, ownership, failed switches and shutdown.
- Native offscreen Qt checks processed a public pose fixture video in hands, pose
  and holistic modes, reached EOF, released resources, and closed. The event-loop
  heartbeat continued. This does not replace a physical unplug/replug retest.
- The complete software suite passes 153 tests. Lint/type checks, strict docs,
  distribution metadata and installed consumer checks are validation gates for
  this maintenance change.
- Ninety historical Flex report/figure files were moved intact to
  `research/camera-placement` and indexed. Hash verification covered 119 research,
  calibration, and archived local legacy files. Participant/experiment data stays
  at its established paths. Twenty-five generated CMake/compiler artifacts were
  removed from Git tracking while retained locally; ignore patterns prevent
  reintroduction. Active `handtrack`/`unity_hand_tracking` interfaces are retained.

Historical Flex numerical results predate the camera-geometry corrections and
lack calibration hashes/source revisions. They remain historical evidence rather
than updated accuracy claims. The working `.venv`, media used by documentation/CI,
and local untracked experiments remain available. Temporary validation/build and
site output can be regenerated and are excluded from source control.


## Project environment isolation — October 30, 2026

The tracker `.venv` contained an editable Mini-Arm installation, Qt6/PySide6,
an older Sphinx documentation stack, and both OpenCV distributions. Package
versions and before/after inventories were saved in ignored
`build/environment-cleanup/` before pruning.

Mini-Arm's existing separate `.venv` was updated with its current editable
package plus the same PySide6 6.11.0 and pylsl 1.18.1 versions. Its dependency
check, CLI help, core imports and an offscreen native Qt6 widget passed without
connecting to robot hardware. MAVIS then removed 38 non-MAVIS packages, retaining
43 packages for its installed runtime, visualization, streaming, serial/YAML
utilities and tests. `opencv-python` was removed and the exact installed
`opencv-contrib-python` 4.13.0.92 wheel reinstalled to repair overlapping files.

MAVIS `.venv` decreased from 1634.53 MiB to 925.91 MiB (708.61 MiB smaller).
Most Qt6 space moved to Mini-Arm's own environment; it was not a comparable net
machine-wide disk saving. Unused Sphinx tools were removed, not relocated.

The cleaned MAVIS environment passes `pip check`, all 153 regression tests,
positive hand video inference, positive combined hand/body inference, native
Qt5 hands/pose/holistic preview checks and both flat/chunked recording/replay/CSV
export. An initial restricted AppContainer run hit temporary file-rename access
errors; the normal workspace-access rerun passed. No runtime source or research
files were changed as part of the environment isolation.


## External research workspace — November 1, 2026

The local root `data/` and `archive/`, the RNN example's data/figures, and the
complete remaining `build/` snapshot were relocated to
`~/Documents/Hand-Landmark-Tracker-Research/`. The external
`relocation-manifest.json` records original paths, destinations, sizes and
SHA-256 hashes for all 172 files (251.76 MiB). Hashes were verified immediately
after the move and after regression checks. The three root folders are absent
from the checkout; no files from those folders were discarded.

Seven affected EMG/RNN examples now use configurable `--data-dir`/`--video`
paths, with `MAVIS_DATA_DIR` and the external Documents workspace as defaults.
The shared path helper is independent of the process working directory and
loads no optional ML dependencies. Existing session-based `--root_dir` options
are retained. Training algorithms and model/data bytes were preserved; model
training/prediction workloads were not rerun in the lean MAVIS environment.

All 163 regression tests pass, including ten new external-path/CLI checks.
Native Qt hands/pose/holistic preview and positive combined hand/body inference
passed using relocated validation assets. Recording/replay/export are checked
separately through the consumer validation script. The working .venv and runtime
calibration locations were retained. Packaging may regenerate ignored build/
output later; generated validation records now live outside the checkout.

The requested OptiTrack history inspection found the SDK header version 1.10.0
and a shipped CameraLibrary2015x64S.dll whose ProductVersion is 1.10.3.5 (the
FileVersion resource is generically 1,0,0,0). Commit d0981e4 also contains
optitrack_cam.cp311-win_amd64.pyd. Release-cleanup commit ced0db4 removed these
vendor assets on October 4, 2026. Inspection did not install or activate the SDK;
version/path/hash evidence is stored in external validation/optitrack-history.json.
