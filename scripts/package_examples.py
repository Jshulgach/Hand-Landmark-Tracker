"""Build source-only example downloads, excluding SDK trees and runtime data."""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "core": [],
    "research": ["Joint_Kinematics_from_EMG", "Joint_Kinematics_from_EMG_OpenEphys", "Smoothing_Motion_with_RNN"],
}
EXCLUDED = {"code_base", "Release", "Debug", "Camera SDK", "__pycache__", ".venv"}
ALLOWED = {".py", ".md", ".json", ".cs", ".txt"}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    for group, directories in GROUPS.items():
        path = output / f"mavis-{group}-examples.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for notice in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
                archive.write(ROOT / notice, notice)
            readme = "DOWNLOADS.md" if group == "core" else "RESEARCH_DOWNLOAD.md"
            archive.write(ROOT / "examples" / readme, "README.md")
            for directory in directories:
                for source in sorted((ROOT / "examples" / directory).rglob("*")):
                    if (source.is_file() and source.suffix in ALLOWED
                            and not set(source.relative_to(ROOT).parts) & EXCLUDED):
                        archive.write(source, source.relative_to(ROOT))
            if group == "core":
                source = ROOT / "examples" / "01_basic_tracking" / "mavis_webcam.py"
                archive.write(source, "mavis_webcam.py")
                for source in (ROOT / "src" / "unity_hand_tracking").glob("*_Hand_Listener.cs"):
                    archive.write(source, "unity/" + source.name)
        with zipfile.ZipFile(path) as archive:
            assert not any(set(Path(name).parts) & EXCLUDED for name in archive.namelist())
        print(path.name)

if __name__ == "__main__":
    main()
