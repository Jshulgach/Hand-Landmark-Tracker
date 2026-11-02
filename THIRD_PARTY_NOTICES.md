# Third-party software

MAVIS uses third-party Python packages under their respective licenses.
See `pyproject.toml` and `uv.lock` for dependencies.

## MediaPipe

MediaPipe 0.10.14 supplies the legacy Solutions API used by this release.
Google distributes MediaPipe under Apache License 2.0. The Python package is
obtained from the package index, not vendored by MAVIS.

## Desktop Qt dependencies

The optional GUI uses PyQt5 and pyqtgraph. PyQt5 is available under GPLv3 or a
commercial license from Riverbank Computing; Qt and other dependencies retain
their own licenses. MAVIS's MIT license does not change those terms.

- [PyQt licensing](https://www.riverbankcomputing.com/software/pyqt/intro)

## OptiTrack Camera SDK

OptiTrack and Camera SDK are software and trademarks of NaturalPoint, Inc.
DBA OptiTrack. Obtain a compatible SDK directly from OptiTrack and comply
with its EULA. Free availability does not itself grant redistribution rights.

SDK headers, samples, documentation, native libraries, DLLs, and compiled bindings
are excluded from MAVIS's Python wheel, source distribution, and curated example
downloads. Earlier repository history contains development copies; they are
not covered by MAVIS's MIT license. Local SDK files are preserved for development.

- [Camera SDK documentation](https://docs.optitrack.com/developer-tools/camera-sdk)
- [Developer downloads](https://optitrack.com/support/downloads/developer-tools.html)
- [OptiTrack EULAs](https://optitrack.com/about/legal/eula)

MAVIS is independent of, and is not endorsed by, Google, Riverbank, or OptiTrack.


## Optional native binding build

The optional OptiTrack Python binding is built using pybind11 2.13.6 under its
BSD 3-Clause license. A locally generated binding ZIP includes pybind11's license
alongside the MAVIS license and these notices. The local bundle excludes all
vendor SDK DLLs, headers and import libraries. OptiTrack's separate terms still
apply to SDK use. No compiled illumination binding is shipped in the MAVIS
wheel, source distribution or curated example downloads.
