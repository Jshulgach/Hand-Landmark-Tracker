"""Check whether all built release filenames are already present on an index."""
import argparse
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen


def already_published(project, version, dist, index):
    files = {path.name for path in Path(dist).iterdir()
             if path.is_file() and (path.name.endswith(".whl") or path.name.endswith(".tar.gz"))}
    if not files:
        raise ValueError("No built release files found")
    prefix = project.replace("-", "_") + "-" + version
    if any(not name.startswith(prefix + "-") and name != prefix + ".tar.gz" for name in files):
        raise ValueError("Release directory contains files for another project or version")
    host = {"pypi": "pypi.org", "testpypi": "test.pypi.org"}[index]
    url = f"https://{host}/pypi/{quote(project, safe='')}/{quote(version, safe='')}/json"
    try:
        with urlopen(url, timeout=30) as response:
            metadata = json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            return False
        raise
    return files <= {entry["filename"] for entry in metadata["urls"]}


def main():
    import tomllib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", choices=("pypi", "testpypi"), default="pypi")
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    published = already_published(project["name"], project["version"], args.dist, args.index)
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a", encoding="utf-8") as stream:
            stream.write(f"published={str(published).lower()}\n")
    print(f"{project['name']} {project['version']} on {args.index}: "
          + ("all files already published; upload skipped" if published else "upload needed"))


if __name__ == "__main__":
    main()
