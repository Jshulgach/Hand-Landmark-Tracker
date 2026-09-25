# Third-party software

HandTrack uses third-party Python packages under their respective licenses. See
`pyproject.toml` and `uv.lock` for the dependency list and resolved versions.

## MediaPipe

HandTrack currently pins MediaPipe 0.10.14 because that is the version tested by
the project. MediaPipe is developed by Google and distributed under the Apache
License 2.0. MediaPipe is installed from the Python package index and is not
vendored in this repository.

## OptiTrack Camera SDK

OptiTrack and Camera SDK are trademarks and software of NaturalPoint, Inc. DBA
OptiTrack. The OptiTrack Camera SDK, its headers, samples, documentation,
libraries, DLLs, and compiled Python extensions are not distributed with
HandTrack and are not covered by this repository's MIT license.

Users who enable the optional OptiTrack backend must obtain a compatible Camera
SDK directly from OptiTrack and comply with the OptiTrack SDK EULA. Availability
at no charge does not grant permission to redistribute the SDK.

- Camera SDK documentation: https://docs.optitrack.com/developer-tools/camera-sdk
- OptiTrack developer downloads: https://optitrack.com/support/downloads/developer-tools.html
- OptiTrack EULAs: https://optitrack.com/about/legal/eula
- OptiTrack website terms: https://optitrack.com/about/legal/web-site-terms

HandTrack is an independent project and is not affiliated with or endorsed by
Google or OptiTrack.
