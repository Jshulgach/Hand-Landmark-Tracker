# Releases

## One-Time Repository Setup

Version 0.1.0 is published on PyPI and TestPyPI. Fresh Windows installs of
both downloaded releases passed the command, API, and recording/replay/export
checks. The production GUI extra also passed its dependency/Qt check.

The [remote software matrix](https://github.com/Jshulgach/Hand-Landmark-Tracker/actions/runs/37257375975)
passes on Windows, Linux, Intel Mac, and Apple Silicon Mac with Python 3.10–3.12,
including 115 tests in each platform job and installed-wheel consumer checks.
Native Mac/Linux webcam permissions and visible desktop behavior remain device
checks. See [troubleshooting](troubleshooting.md) for MediaPipe's Apple Silicon
wheel metadata warning and the required native runtime validation.

For automated releases and documentation deployment:

- create GitHub environments named `testpypi`, `pypi`, and `github-pages`
- configure PyPI trusted publishing for the `Publish TestPyPI` and `Publish PyPI` workflows
- enable GitHub Pages with GitHub Actions as the deployment source

## Local Validation

Before cutting a release, run:

```bash
python -m pytest -q
mavis-track --help
mkdocs build --strict
python -m build
python -m twine check dist/*
```

## Hardware Smoke Commands

```bash
mavis-track doctor
mavis-track record --source 0 --frames 120 --save-video
mavis-track replay recordings/<session-name>
mavis-track export recordings/<session-name>
```

## TestPyPI

Use the TestPyPI workflow first for every new release series.

Expected outcome:

- sdist and wheel build successfully
- metadata passes `twine check`
- package installs from TestPyPI into a clean environment
- top-level CLI works after installation

## PyPI

Promote to PyPI only after the TestPyPI artifact has been installed and smoke-tested.
Both upload workflows first run the full CI matrix. Already-published files are
skipped, so a GitHub release for the manually published 0.1.0 does not upload it again.

## Docs Deployment

The docs workflow builds the MkDocs site and publishes it to GitHub Pages.

## Release Notes

Document user-visible CLI changes, dependency changes, and backend compatibility notes in `CHANGELOG.md`.
