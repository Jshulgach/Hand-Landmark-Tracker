"""External data-path helpers for the optional research examples.

These helpers do not load datasets or import ML/GUI dependencies.
"""
from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path


def research_data_dir(value: str | Path | None = None, *, collection: str | None = None) -> Path:
    """Resolve an explicit example directory, or the shared external data root.

    --data-dir overrides both the environment and the optional collection.
    MAVIS_DATA_DIR selects the shared root; default collections live beneath it.
    No directory is created by this function.
    """
    if value is not None:
        return Path(value).expanduser().resolve()
    base = Path(os.environ.get("MAVIS_DATA_DIR", str(
        Path.home() / "Documents" / "Hand-Landmark-Tracker-Research" / "data"))).expanduser()
    return (base / collection if collection else base).resolve()


def research_example_args(description: str, argv: Sequence[str] | None = None, *,
                          collection: str | None = None, video: bool = False) -> argparse.Namespace:
    """Parse example paths before importing optional scientific packages."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", type=Path, default=None,
        help="dataset/model directory; default MAVIS_DATA_DIR or external Documents research data")
    if video:
        parser.add_argument("--video", type=Path, default=None,
            help="source video; defaults to HANDTRACK_VIDEO_PATH or HandDynamic.mp4 in the data directory")
    args = parser.parse_args(argv)
    args.data_dir = research_data_dir(args.data_dir, collection=collection)
    if video:
        args.video = Path(args.video or os.environ.get("HANDTRACK_VIDEO_PATH")
                          or args.data_dir / "HandDynamic.mp4").expanduser().resolve()
    return args
