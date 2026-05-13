# Releases

## One-Time Repository Setup

Before the first public release:

- create GitHub environments named `testpypi`, `pypi`, and `github-pages`
- configure PyPI trusted publishing for the `Publish TestPyPI` and `Publish PyPI` workflows
- enable GitHub Pages with GitHub Actions as the deployment source

## Local Validation

Before cutting a release, run:

```bash
python -m pytest -q
handtracker --help
mkdocs build --strict
python -m build
python -m twine check dist/*
```

## Hardware Smoke Commands

```bash
handtracker doctor
handtracker record --source 0 --frames 120 --save-video
handtracker replay recordings/<session-name>
handtracker export recordings/<session-name>
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

## Docs Deployment

The docs workflow builds the MkDocs site and publishes it to GitHub Pages.

## Release Notes

Document user-visible CLI changes, dependency changes, and backend compatibility notes in `CHANGELOG.md`.