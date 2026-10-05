# MAVIS release checklist

Distribution: `mavis-track`; import: `mavis_track`; version: 0.1.0.
Version 0.1.0 was published to [PyPI](https://pypi.org/project/mavis-track/0.1.0/)
and [TestPyPI](https://test.pypi.org/project/mavis-track/0.1.0/) on October 4,
2026. Both indexes' files match the reviewed artifacts. Fresh downloaded-wheel
installs passed command, API, and record/replay/export checks on Windows;
the production GUI extra also passed its dependency/Qt check.

The [full remote CI matrix](https://github.com/Jshulgach/Hand-Landmark-Tracker/actions/runs/37257375975)
passes on Windows, Linux, Intel Mac, and Apple Silicon Mac with Python 3.10–3.12:
115 tests per platform job, installed-wheel consumer checks, and the quality job.
Native Mac/Linux webcam and visible desktop checks remain separate validation
work. Apple Silicon's known MediaPipe wheel metadata warning is documented in
`docs/troubleshooting.md`; native tracking still must succeed. For subsequent releases:

1. Review the implementation and the follow-up evidence in `AUDIT.md`.
2. Run the complete CI matrix: Windows, Linux, Intel Mac, Apple Silicon Mac,
   Python 3.10–3.12. Each job builds a wheel, installs the base package,
   runs the outside-checkout consumer smoke, and tests desktop extras.
3. Verify native camera permissions, start/stop/retry, no-hand/one-hand/two-hand
   demos, video codecs, and graceful device disconnection on each desktop.
4. Verify release artifacts contain notices and Unity listener scripts, with
   no proprietary SDK headers, DLLs, libraries, extensions, or runtime datasets.
   Curated examples use `scripts/package_examples.py` and an allowlist.
5. Configure PyPI/TestPyPI trusted publishing for this GitHub repository and
   the `pypi`/`testpypi` environments. Use environment approval protection for
   production publishing.
6. Publish TestPyPI through the manual workflow; test a fresh install using the
   built wheel or the TestPyPI package with dependencies from PyPI.
7. Check PyPI name availability again immediately before production publication.
8. Publish a reviewed release. Both publishing workflows depend on the complete
   CI workflow; they cannot bypass failed platform checks. An index lookup skips
   upload when all built filenames are already published. Partial uploads can
   resume without re-uploading files that already exist.
9. Update installation prose, tag notes with actual tested versions, and
   deploy documentation. Repository README corrections after publication appear
   in PyPI package metadata when included in a new package version.

OptiTrack's historical Windows success is documented separately. No OptiTrack
system is currently available for a fresh hardware retest. Do not delay the
ordinary webcam release for that equipment; keep the advanced SDK feature's
validation status explicit.

Source indices do not prove physical camera identity. Multi-camera two-hand
association, clinical angle validation, absolute precision, and synchronized
exposure claims are outside this initial release's verified capabilities.

The old Git history contains proprietary development copies. New Python and
example artifacts exclude them; do not rewrite history as part of this change.
Review any repository-history distribution obligations separately.

## Manual publication from Windows PowerShell

The following records the 0.1.0 publication procedure; that version is already
uploaded and its files must not be uploaded again. For a new release, update the
version and artifact paths below after completing the platform checks above. These
commands build once, upload to TestPyPI, verify that download, and then upload
the same files to PyPI. Stop if any command fails. Uploading is a separate
manual action; these commands do not run the GitHub validation workflow.

Create verified accounts and an API token for each site:
[TestPyPI tokens](https://test.pypi.org/manage/account/#api-tokens) and
[PyPI tokens](https://pypi.org/manage/account/#api-tokens). For a first project
upload, create an account-scoped token; subsequent uploads can use a token
scoped to `mavis-track`. Twine prompts for the matching site's token.
See the [PyPI token instructions](https://pypi.org/help/#apitoken) and
[official packaging guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

Build and check the artifacts:

```powershell
Set-Location 'C:\Users\jonat\Documents\GitHub\Hand-Landmark-Tracker'
python -m venv .venv-release
$mavisReleasePython = '.\.venv-release\Scripts\python.exe'
& $mavisReleasePython -m pip install --upgrade pip build twine
& $mavisReleasePython -m build --outdir dist\mavis-track-0.1.0
& $mavisReleasePython -m twine check dist\mavis-track-0.1.0\*
```

Publish to TestPyPI:

```powershell
& $mavisReleasePython -m twine upload --repository testpypi --username __token__ dist\mavis-track-0.1.0\*
```

Verify a fresh TestPyPI download. The package wheel comes from TestPyPI;
its dependencies come from the normal PyPI index:

```powershell
python -m venv .venv-testpypi
$mavisTestPython = '.\.venv-testpypi\Scripts\python.exe'
& $mavisTestPython -m pip download --no-deps --no-cache-dir --index-url https://test.pypi.org/simple/ --dest dist\testpypi 'mavis-track==0.1.0'
& $mavisTestPython -m pip install --index-url https://pypi.org/simple/ dist\testpypi\mavis_track-0.1.0-py3-none-any.whl
& $mavisTestPython -m pip check
Copy-Item scripts\consumer_smoke.py "$env:TEMP\mavis-testpypi-smoke.py"
& $mavisTestPython "$env:TEMP\mavis-testpypi-smoke.py"
```

Publish the same checked artifacts to real PyPI after TestPyPI and platform
validation pass, using the PyPI token at the prompt:

```powershell
& $mavisReleasePython -m twine upload --repository pypi --username __token__ dist\mavis-track-0.1.0\*
```

After publication, new users can install the desktop demo with
`python -m pip install "mavis-track[gui]==0.1.0"` and run `mavis-track demo`.
If release files change after an upload, bump the version in `pyproject.toml`
and `src/mavis_track/__init__.py` and use a new version-specific artifact folder.
