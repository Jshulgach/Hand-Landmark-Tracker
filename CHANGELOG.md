# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

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