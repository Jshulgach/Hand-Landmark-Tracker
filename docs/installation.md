# Installation

MAVIS 0.1.0 is available on [PyPI](https://pypi.org/project/mavis-track/).
Python 3.10–3.12, 64-bit, is required; Python 3.13 and newer
are excluded because the pinned MediaPipe release does not provide those wheels.

```bash
python -m venv .venv
```

Activate on Windows with `.venv\Scripts\activate`; macOS/Linux use
`source .venv/bin/activate`.

```bash
python -m pip install "mavis-track[gui]"
mavis-track doctor --backend webcam
mavis-track demo
```

For source development, clone the repository and use
`python -m pip install ".[gui]"` from its root instead.

## Choose dependencies

| Install extra | Use |
| --- | --- |
| `mavis-track` | Frame API, recording, replay, CSV export, UDP |
| `mavis-track[gui]` | Qt desktop demos and advanced visualization |
| `mavis-track[stream]` / `mavis-track[lsl]` | LSL |
| `mavis-track[io]` | pandas/YAML research utilities |
| `mavis-track[applications]` | Matplotlib and serial application utilities |
| `mavis-track[ml]` | Research machine learning tools, including PyTorch |
| `mavis-track[optitrack]` | Python helpers; proprietary SDK obtained separately |
| `.[dev,gui,io,stream,docs,release]` | Tests, docs, desktop checks, release tooling |

Use only one OpenCV distribution. MAVIS selects `opencv-contrib-python`,
which MediaPipe also depends on; do not add `opencv-python` or headless
variants into the same environment. A base install still includes MediaPipe's
transitive dependencies and OpenCV's native runtime.

## Platform status

The automated release matrix covers Windows x64, Linux x64, macOS Intel, and
macOS Apple Silicon for Python 3.10–3.12. Configuring this matrix is not evidence
that it has completed. Current local runtime evidence is recorded in the audit.
Camera permissions, codecs, repeated device start/stop, and native display checks
must also be tested on each supported desktop.

Grant camera permission to the terminal/Python application on macOS. On Linux,
OpenCV may need desktop system libraries such as libGL; camera access requires
the correct device permissions. The base package is camera-free when embedded,
but it uses the standard OpenCV wheel rather than a headless variant.

OptiTrack uses an optional compiled Windows binding. Download the SDK separately,
install/build a binding for your interpreter, and set `OPTITRACK_CAM_PY_PATH`
to that folder. Its hardware compatibility is separate from webcam portability.

## Verify a distribution

```bash
python -m build
python -m twine check dist/*
```

Install the resulting wheel in a fresh environment and run
`scripts/consumer_smoke.py` from outside the repository. Development tests also
exercise the source tree, so passing those alone does not establish wheel completeness.
