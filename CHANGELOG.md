# Changelog

All notable changes to this project will be documented in this file.

## Unreleased — MAVIS release candidate

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

## 0.1.0 - 2026-05-13

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
