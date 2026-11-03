# Changelog

## Unreleased

- Show WiLoR startup stages, elapsed time, selected CPU/CUDA device, completed
  batches, and zero-detection/raw-preview status. Distinguish model loading from
  first inference; print initialization/inference exception chains and the
  interpreter path while preserving live camera feeds.

- Fix OptiTrack stop/start and MediaPipe/WiLoR switching by keeping the SDK
  initialized for the application lifetime. Stop releases capture handles;
  final SDK shutdown happens after workers stop on application exit. Protect
  ownership during shutdown and clean up handles after partial startup failures.


- Add one-command WiLoR setup with --setup. Source preparation can merge into a
  folder containing a licensed MANO file and checkpoints, preserving existing
  assets, rejecting conflicting source files, and cleaning failed preparation.


- Add a private WiLoR offline transfer kit with verified wheels/assets, local-only
  installation and a real inference/GUI audit. Disable detector connectivity
  probes, honor WiLoR doctor GUI/backend flags, and discover assets/SDK beside
  a project-local environment for wheel installs.


- Fix WiLoR's pip bootstrap on fresh Python 3.11 environments: upgrade pip before
  using its built-in system certificate support, without requiring an
  uninstalled truststore feature. Show the target interpreter during setup.


- Add shared-model WiLoR inference and mesh/joint viewing for all connected
  OptiTrack cameras, independent per-view association, bounded cross-view crop
  batches, live-feed/matched-frame display and asynchronous startup/shutdown.
- Preserve exposure/IR controls and camera preview during slow/failed inference;
  keep per-camera model coordinates separate from calibrated reconstruction.
- Verify four-view real-model reference rendering on CPU and preserve the existing
  MediaPipe multi-camera workflows. Physical OptiTrack/RTX validation is pending.

- Add experimental official WiLoR hand inference, a separate frame API with
  explicit pixel/model coordinate spaces, and selectable live mesh/joint preview.
- Add a pinned Windows/Python 3.11 CUDA-capable runtime installer with verified
  resumable checkpoints and isolation from the working environment; disable
  runtime package auto-installs and settings sync.
- Verify real CPU hand reconstruction and an operator-confirmed live overlay.
  GPU execution, fast mode, infrared performance and accuracy remain pending.

- Publish a standalone Windows/Python 3.11 illumination binding and add a pinned
  prebuilt/offline ZIP installer with native preflight, backup and rollback.
  Keep auxiliary binding releases separate from PyPI/example publishing.

- Add SDK-bounded IR intensity, high-power availability/readback and Max available
  controls for individual/all OptiTrack cameras in tracking and calibration.
- Add calibration I/K intensity, M maximum and H high-power shortcuts, with
  settling after lighting changes and graceful handling of older bindings.
- Add illumination binding API v2, isolated local build/install support and
  compile-only Windows verification; no native artifact is uploaded.

- Document the required OptiTrack USB driver step for a new Windows computer,
  including the operator-verified driver-only installer workflow, and distinguish
  successful runtime imports from actual camera discovery.

- Track the external setup guide, activation scripts and checksum-verified
  OptiTrack runtime recovery utility; keep vendor files local. Discover the
  checkout runtime automatically and expose original native import errors.

- Use scalable point-sized controls across monitors, text zoom, an adaptive camera
  grid and display-only fit/fill choices in the OptiTrack tracking GUI.
- Add a minimal scalable OptiTrack calibration preview with keyboard exposure,
  per-camera/all-camera targeting, pause, finish and safe cancellation.
- Document an estimator audit and equal-priority hand/whole-body accuracy roadmap.

- Keep OptiTrack preview and 2D hand overlays running when calibration is missing
  or rejected; show calibration status and gate 3D views, controls, and outputs
  until a valid calibration is loaded on tracking start.

- Fit the OptiTrack GUI to the available screen area, with resizable camera/3D
  previews and a scrollable control panel for smaller and scaled displays.

- Bound recording memory with incremental numeric chunks, recoverable checkpoints,
  streaming replay/export, and a public `iter_session_chunks` analysis API.
- Move live preview capture, inference and mode switches off the Qt thread, with
  bounded latest-frame delivery and asynchronous stop/close cleanup.
- Consolidate historical Flex reports under `research/camera-placement`, preserve
  unused local scripts in the archive, and untrack native build caches/artifacts.

- Add the caller-owned-frame `HandBodyTracker` API for single-person synchronized
  hand/body observations, explicit coordinate spaces, read-only arrays, and
  per-region tracking-quality flags.
- Add `record --mode hand-body`, numeric session schema 2 with shared frame
  timestamps, body visibility, and quality metadata; extend load/replay/export.
  Existing hand-only API and schema 0/1 readers remain supported.


All notable changes to this project will be documented in this file.

## mavis-track 0.1.0 - 2026-10-04

First public MAVIS release on [PyPI](https://pypi.org/project/mavis-track/0.1.0/).

- Adopt `mavis-track`, `mavis_track`, and the MAVIS brand; retain legacy imports/commands.
- Add a camera-free frame API with owned arrays, short-lived hand association, explicit coordinates/clocks, and context-managed cleanup.
- Make the default desktop demo work without calibration, with mode switching and camera retry.
- Repair recording, missing/two-hand sessions, timed replay, and rectangular CSV exports.
- Reject pickle/object archives, invalid shapes, and oversized declared arrays before loading.
- Correct geometric angle triples, initial smoothing state, and pixel-noise covariance scaling.
- Validate calibration identity/count/resolution and preserve prior calibration when held-out quality gates fail.
- Reserve unseen views for reconstructed-board shape, scale, and planarity checks.
- Exclude stale/skewed camera frames; release devices/models on startup and processing failures.
- Send explicit hand-loss packets; label UDP/LSL units and clocks; release LSL outlets.
- Separate desktop/research/LSL extras and select one OpenCV distribution.
- Exclude SDK assets from distributions and retain local development copies outside Git tracking.
- Add Windows/Linux/Intel Mac/Apple Silicon Mac, Python 3.10–3.12 release gates and installed-wheel smoke tests.
- Document integration, compatibility limits, and outstanding native device validation.

## 0.1.0 development baseline - 2026-05-13

### Added

- unified `handtracker` CLI with backend auto-selection
- backend-neutral `board`, `test-sender`, `doctor`, and `inspect-calibration` commands
- backend performance benchmarking with `handtracker benchmark`
- session capture, replay, and CSV export commands: `record`, `replay`, and `export`
- MkDocs-based documentation site scaffold
- CI, docs deployment, TestPyPI, and PyPI workflow templates
- contributing guide, issue templates, and release checklist

### Changed

- README now documents the generic CLI-first workflow
- package metadata now includes docs and release extras plus public URLs
- package versioning and distribution metadata now target a first public PyPI/TestPyPI release

### Fixed

- README demo image path now points at a tracked repository asset
- calibration inspection now fails clearly when a backend artifact is missing
- session loading now supports root-level `landmarks.npz` bundles written by `record`
- MediaPipe dependency failures now surface clear guidance when the legacy Hands API is unavailable
- OptiTrack camera startup fails fast when the SDK reports cameras but does not return usable handles
