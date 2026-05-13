# Installation

## Supported Python Versions

HandTrack currently targets Python 3.10 and 3.11.

## Base Install

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Development Install

```bash
python -m pip install -e .[dev,docs,release]
```

## OptiTrack Install

```bash
python -m pip install -e .[optitrack]
```

If the OptiTrack SDK module is not importable, the generic CLI will fall back to the webcam backend.

## Verify the Environment

```bash
handtracker doctor
handtracker inspect-calibration --backend webcam
```

## Build the Package Locally

```bash
python -m build
python -m twine check dist/*
```