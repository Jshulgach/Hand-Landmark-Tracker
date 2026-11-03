# WiLoR installation and offline audit — 3 November 2026

This audit covers WiLoR setup, real model loading/inference, the Qt multi-camera
window and OptiTrack binding imports. It cannot certify an unconnected RTX 3060,
its driver, USB cameras, infrared accuracy or long-session hardware behavior.
Those need the target-computer checks below.

## Findings and changes

| Finding | Result |
| --- | --- |
| Fresh pip could require truststore before upgrading itself | Fixed previously; reproduced with pip 23.1.2 and verified a successful upgrade in a fresh environment. |
| Online requirements include a Git dependency and two package indexes | Added a private transfer kit containing prebuilt wheels and a local, SHA-256-pinned requirements file. No Git/compiler/index access is needed during target installation. |
| Detector import can probe external internet connectivity | Disabled with YOLO_OFFLINE before import and ONLINE=False for an already imported detector module. Auto-install and settings sync remain disabled. |
| WiLoR doctor discarded requested GUI/backend checks | Doctor now checks the requested GUI and OptiTrack binding too. An import succeeding still does not prove cameras are connected. |
| Installed wheels looked for assets/SDK in source-checkout locations | Added discovery beside a project-local environment. The generated launcher also configures explicit asset/SDK paths. |
| A same-version older MAVIS installation could remain active | Offline installation explicitly replaces the MAVIS wheel, even if an older editable install reports version 0.1.0. |
| Interrupted transfers or conflicting user assets | Every kit file is size/SHA-256 checked before installation; different existing assets are preserved and reported. |

## Recommended transfer to the RTX computer

A private kit is prepared locally at `external/wilor-offline-win311`.
It includes Windows x64/CPython 3.11 wheels, WiLoR source, the two verified
checkpoints, your MANO file, a reference video and this computer's optional
OptiTrack runtime. It is about **5.5 GiB**. Keep the folder private: it contains
licensed assets and, when requested, the vendor runtime. It is ignored by Git
and is not published with the project.

1. Transfer the **whole folder** to the other PC, preserving its subfolders.
   Placing it under that PC's MAVIS `external` folder makes the commands below
   work unchanged. A USB drive avoids another network download.
2. Ensure **64-bit Windows CPython 3.11**, the **NVIDIA display driver** and
   **OptiTrack USB drivers** are already installed. These OS prerequisites are
   not installed by the kit. Installing/importing a Python binding does not
   install camera drivers. Close running MAVIS windows before installation.
3. Keep at least roughly **15 GiB free in addition to the kit** for a fresh
   environment, assets and installation workspace. The exact use depends on
   existing files and whether same-volume immutable checkpoints can be linked.
4. From the target MAVIS folder, run:

```powershell
py -3.11 external/wilor-offline-win311/tools/wilor_offline.py install external/wilor-offline-win311 --target . --require-cuda
```

The installer uses only local wheels with package indexes disabled. It verifies
all file hashes first, preserves the regular `.venv`, creates/reuses a separate
`.venv-wilor` and checks its dependencies. It then runs the GUI/backend, GPU
tensor and real reference-inference audits. Licensed assets are not downloaded
again. Existing different assets cause a preservation error rather than being
overwritten.

After it succeeds:

```powershell
.\external\launch_wilor_multicam.cmd
```

Click **Start Tracking**. Check all four live feeds, hand overlays, exposure and
IR controls. Model coordinates remain separate per camera; this does not enable
calibrated fusion. If GPU memory is tight, the launcher accepts
`--wilor-batch-size 1`. The installation audit report is saved to
`external/wilor-install-audit.json`.

Do not copy an existing virtual environment between computers. The local wheels
recreate it with the correct paths.

## Rebuilding a private kit on an online configured PC

These commands intentionally use internet access on the preparation PC.
The runtime environment and licensed MANO asset must already be available:

```powershell
.\.venv-wilor\Scripts\python.exe scripts/wilor_offline.py prepare external/wilor-offline-win311 --optitrack-runtime external/optitrack-sdk-1.10/runtime
```

Omit `--optitrack-runtime` if transferring that separately. Preparation builds
the source-only dependency wheels here, checks the source revision, verifies the
official checkpoint sizes/hashes, and writes `kit-manifest.json`. Hashes detect
transfer corruption; obtain the whole kit from your own trusted preparation PC.

To check a transferred kit without installing:

```powershell
py -3.11 external/wilor-offline-win311/tools/wilor_offline.py verify external/wilor-offline-win311
```

## Repeatable runtime audit

On a configured target, from the repository root:

```powershell
.\.venv-wilor\Scripts\python.exe scripts/audit_wilor.py --gui --optitrack --require-cuda --fast --block-network --video docs/source/_static/HandDynamic.mp4 --report external/wilor-target-audit.json
```

For a CPU-only development host, omit `--require-cuda --fast`.
The audit checks every pinned package version, `pip check`, model assets,
requested GUI/SDK imports, offscreen window construction and real model
inference/overlay rendering. Reference images simulate multiple views; they are
not independent physical camera observations.

The network guard blocks external **Python socket** operations and records
attempts. It is not an OS firewall and cannot certify arbitrary native-library
networking. Offline installation separately disables pip indexes and uses no
VCS requirements. A final target-PC test with its network disconnected gives
additional operational evidence.

A startup warning about Lightning upgrading an old checkpoint **in memory**,
MANO's ten shape coefficients, or an upstream deprecated torch.cross call is not
a missing asset. Keep the verified checkpoint files unchanged; there is no need
to run the suggested permanent checkpoint-upgrade command.

The target audit is intentionally separate from physical-camera testing: an SDK
module loading cannot prove that drivers, USB bandwidth, hub power and cameras
are working. The RTX GPU and physical hub are unavailable on this development PC.
