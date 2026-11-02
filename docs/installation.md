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


## Keep project environments separate

Use this repository's `.venv` for MAVIS. Activate Mini-Arm's own `.venv` from
its repository when working on Mini-Arm. MAVIS uses PyQt5/Qt5; Mini-Arm's desktop
runtime uses PySide6/Qt6. Keeping these installations separate avoids pulling
both Qt runtimes and unrelated documentation tools into the tracker environment.

For a focused source installation, use `python -m pip install -e ".[gui,stream]"`
in the MAVIS environment. Add the other extras only for the workflows you use.
Sphinx is not needed for MAVIS tracking; its documentation build uses the `docs`
extra (MkDocs). MediaPipe still declares JAX/JAXlib, Matplotlib and sounddevice
as dependencies, so those remain installed even for camera tracking.

Research datasets and trained weights are stored outside the checkout. See the
[external-data guide](research-data.md) for `--data-dir`, `--video` and
`MAVIS_DATA_DIR` configuration.

## OptiTrack on a new Windows computer

For the historical Flex 13 FS setup, ensure the OptiTrack USB drivers are
installed, use **Python 3.11 x64**, and install the current checkout rather than
relying on the older published package. Camera SDK 1.10.3 was used successfully
with this setup. If drivers from another SDK are already installed, test camera
detection before changing them; the setup utility does not inspect driver status. The SDK installer does not supply this
project's Python binding. From a full Git clone with that environment active:

```bat
python -m pip install -e ".[gui,optitrack]"
python external/setup_optitrack.py --from-history
python -m mavis_track doctor --backend optitrack --gui
python -m mavis_track gui --backend optitrack
```

The tracked `external/` setup files recover the two matching runtime files from
existing Git history into a local, ignored folder. Checksums and the interpreter
ABI are checked before use; a different existing runtime is preserved. Editable
installations automatically discover that folder. USB drivers and vendor license
review still use the official installer. Source ZIPs and shallow clones need a
full clone or a copied matching runtime. See the
[external setup guide](https://github.com/Jshulgach/Hand-Landmark-Tracker/tree/main/external)
for recovery, copy and environment-variable commands.
