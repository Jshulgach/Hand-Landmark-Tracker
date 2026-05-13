# Release Checklist

## One-Time Repository Setup

1. Create GitHub environments named `testpypi` and `pypi` under Settings > Environments.
2. In both environments, configure PyPI trusted publishing for this repository and the matching workflow files:
	- `publish-testpypi.yml` for the `testpypi` environment
	- `publish-pypi.yml` for the `pypi` environment
3. Enable GitHub Pages under Settings > Pages and set the source to GitHub Actions.
4. Confirm the Pages environment is named `github-pages` so `.github/workflows/docs.yml` can deploy without edits.
5. Verify the `Documentation` URL in `pyproject.toml` matches the final Pages URL.

## Pre-Release Validation

```bash
python -m pytest -q
handtracker --help
handtracker doctor --help
handtracker inspect-calibration --backend webcam
mkdocs build --strict
python -m build
python -m twine check dist/*
```

## Hardware Smoke Commands

Use these against a real camera or a known-good video before the release:

```bash
handtracker doctor
handtracker record --source 0 --frames 120 --save-video
handtracker replay recordings/<session-name>
handtracker export recordings/<session-name>
```

## TestPyPI Flow

1. Run the `Publish TestPyPI` workflow.
2. Install from TestPyPI in a clean environment.
3. Run `handtracker --help` and at least one backend-neutral smoke test.

## PyPI Flow

1. Update `CHANGELOG.md`.
2. Tag the release or publish a GitHub release.
3. Let the `Publish PyPI` workflow upload the checked distributions.

## Post-Release

1. Confirm the GitHub Pages site reflects the latest docs.
2. Confirm the PyPI project description renders correctly.
3. Add any release-specific troubleshooting notes to the docs.