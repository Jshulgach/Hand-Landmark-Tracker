# Local camera dependencies

This directory is included in Git for MAVIS-owned setup files. Vendor SDKs,
binaries, installers and generated reports stay local and ignored. Nothing here
is added to the Python wheel or source distribution.

## OptiTrack: new Windows computer

The known historical runtime uses Camera SDK 1.10.x (DLL ProductVersion
1.10.3.5) and **64-bit CPython 3.11 on Windows x64**. MAVIS itself supports
other Python versions, but this particular prebuilt binding does not.

1. **Complete the OptiTrack USB driver installation on each new Windows PC.**
   Download and launch the official [Camera SDK installer](https://optitrack.com/support/downloads/developer-tools.html),
   select its USB driver component, and allow the driver installer to finish.
   Installing Python packages or restoring the local runtime does not install
   these Windows drivers. Other motion-capture software being installed does
   not establish that the OptiTrack drivers are present.

   **Driver-only workflow:** the operator confirmed on the new PC that completing
   the USB driver installer, then cancelling the remaining Camera SDK installation,
   resolved zero-camera detection. Finish the driver step before cancelling the
   parent installer. If the installer offers a driver-only/custom component
   option, use that; the available workflow depends on the installer version.
   Additional SDK development tools are not required by the prebuilt-runtime
   workflow below. Review the vendor license during installation.

   Confirm Windows recognizes the hub and cameras without driver warnings, then
   test camera detection in MAVIS. The historical Flex 13 FS setup previously
   worked with Camera SDK **1.10.3**; the exact installer version used on the new
   PC was not recorded, so this is not a claim that every driver version works.
2. Use a Python 3.11 x64 environment and a full Git clone of this project.
   From its root, install the current source and restore the local runtime:

```bat
call .venv\Scripts\activate.bat
python -m pip install -e ".[gui,optitrack]"
python external/setup_optitrack.py --from-history
python -m mavis_track doctor --backend optitrack --gui
python -m mavis_track gui --backend optitrack
```

Create the environment with `py -3.11 -m venv .venv` if it does not already
exist. Check `python --version` before setup. Optional LSL uses the `stream`
extra. Restart an already running MAVIS process after setup.

The recovery command reads the two runtime files already present at historical
commit `d0981e46b7f1838f7ac3af2ebc4f935ba763be6d`, verifies pinned SHA-256
checksums and writes them under `external/optitrack-sdk-1.10/runtime/`. It does
not download an SDK, restore headers, accept licenses, install drivers or open
cameras. It refuses to overwrite a different local runtime. This is a local
recovery utility, not a new redistribution of the SDK with MAVIS.

A shallow clone needs its history restored first (`git fetch --unshallow`). A
GitHub source ZIP has no history: use a full clone, or copy the runtime from the
configured computer and pass its directory instead:

```bat
python external/setup_optitrack.py --runtime-dir "D:\copied-sdk\runtime"
python external/setup_optitrack.py --check
```

The SDK's vendor terms continue to apply to these historical files. OptiTrack
asks applications to deliver runtime components using its installers and says
bundling its installer requires endorsement. See the
[distribution FAQ](https://www.optitrack.com/support/faq/general) and
[SDK EULA](https://www.optitrack.com/about/legal/eula). SDK files are not licensed
under MAVIS's MIT license; do not add the vendor installation to commits.

## Discovery and troubleshooting

An editable/source installation automatically searches this checkout's runtime
folder. `OPTITRACK_CAM_PY_PATH` remains available for a custom binding and takes
priority. Legacy package `Release/` locations also remain supported. An explicit
SDK path is needed for a separately installed wheel; these optional helpers set
it for the current terminal from any checkout location:

```bat
call external\Activate.cmd
```

```powershell
& .\external\Activate.ps1
```

`doctor` reports the original import exception. A missing binding, a Python ABI
mismatch, a missing CameraLibrary DLL and a missing native runtime dependency
are distinct setup issues. The official SDK installation does not create our
`optitrack_cam` Python binding. Likewise, a successful import does not prove USB
drivers are installed or cameras are connected. Confirm `optitrack:sdk : ok`,
then start preview. Calibration is only needed to enable 3D reconstruction.

An unset `OPTITRACK_CAM_PY_PATH` is normal with automatic discovery. In `doctor`,
check `optitrack:sdk : ok` and the `optitrack:binding` path. The environment row
is informational. If `optitrack:sdk` is missing, use `optitrack:import-error` to
identify the actual load failure. The activation helpers above set the optional
variable for the current terminal when explicit configuration is desired.

The recovered binding uses its matching local legacy CameraLibrary DLL. Merely
installing a newer system SDK does not switch that binding to the newer library.
Using a newer SDK runtime requires a compatible binding and separate validation;
do not replace the recovered DLL with an arbitrary newer version. An SDK import
check and a camera enumeration test verify different parts of the setup.

## Confirmed new-PC setup issue

On 2 November 2026, the operator reported `optitrack:sdk : ok` while the GUI
detected zero connected cameras. Completing the USB driver stage of the official
OptiTrack installer resolved discovery; the remaining SDK installation was
cancelled afterward. This confirms that a successful Python/native-library
import is not a USB driver or camera-detection check.
