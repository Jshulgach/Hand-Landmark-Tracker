# Contributing

## Development Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .[dev,docs,release]
```

## Before Opening a Pull Request

Run the local checks:

```bash
python -m pytest -q
handtracker --help
handtracker doctor --help
handtracker inspect-calibration --backend webcam
mkdocs build --strict
python -m build
python -m twine check dist/*
```

## Pull Request Expectations

- keep changes focused on one user-visible objective
- update docs when CLI behavior changes
- preserve backward compatibility when replacing entrypoints
- include a validation note in the PR description

## Issue Reporting

Use the bug and feature templates so environment and backend details are captured consistently.