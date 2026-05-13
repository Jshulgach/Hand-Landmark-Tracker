"""Export recorded HandTrack session bundles into flat CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def _write_landmarks_csv(
    path: Path,
    landmarks: np.ndarray,
    time_vector: np.ndarray,
) -> None:
    header = ["frame", "timestamp"]
    for landmark_index in range(landmarks.shape[1]):
        prefix = f"lm{landmark_index:02d}"
        header.extend([f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"])

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for frame_index, (timestamp, frame_landmarks) in enumerate(
            zip(time_vector, landmarks, strict=True)
        ):
            writer.writerow(
                [frame_index, f"{float(timestamp):.6f}"]
                + [f"{float(value):.6f}" for value in frame_landmarks.reshape(-1)]
            )


def _write_angles_csv(
    path: Path,
    time_vector: np.ndarray,
    angle_names: list[str],
    angle_values: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "timestamp"] + angle_names)
        for frame_index, (timestamp, row) in enumerate(
            zip(time_vector, angle_values, strict=True)
        ):
            writer.writerow(
                [frame_index, f"{float(timestamp):.6f}"]
                + [f"{float(value):.6f}" for value in row]
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a recorded HandTrack session into flat CSV artifacts.",
    )
    parser.add_argument(
        "session",
        help="path to a session folder containing landmarks.npz",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="directory for exported files; defaults to <session>/exports",
    )
    parser.add_argument(
        "--variant",
        choices=("landmarks", "raw_landmarks"),
        default="landmarks",
        help="which landmark array to export",
    )
    parser.add_argument(
        "--skip-angles",
        action="store_true",
        help="skip exporting angle_values to CSV",
    )
    args = parser.parse_args(argv)

    session_dir = Path(args.session)
    if not session_dir.exists():
        print(f"[handtracker] Session folder not found: {session_dir}")
        return 1

    bundle_path = session_dir / "landmarks.npz"
    if not bundle_path.exists():
        print(f"[handtracker] landmarks.npz not found in: {session_dir}")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else session_dir / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(bundle_path, allow_pickle=True) as bundle:
        if args.variant not in bundle:
            print(
                f"[handtracker] Variant '{args.variant}' not found in {bundle_path.name}"
            )
            return 1

        landmarks = np.asarray(bundle[args.variant], dtype=np.float32)
        time_vector = np.asarray(
            bundle.get("time_vector", np.arange(landmarks.shape[0], dtype=np.float64)),
            dtype=np.float64,
        )
        sampling_rate = float(bundle.get("sampling_rate", 30.0))
        angle_names = [
            str(name)
            for name in bundle.get("angle_names", np.asarray([], dtype=str)).tolist()
        ]
        angle_values = np.asarray(
            bundle.get(
                "angle_values", np.empty((landmarks.shape[0], 0), dtype=np.float32)
            ),
            dtype=np.float32,
        )

    landmarks_csv_path = output_dir / f"{args.variant}.csv"
    _write_landmarks_csv(landmarks_csv_path, landmarks, time_vector)

    written_files = [landmarks_csv_path.name]

    if not args.skip_angles and angle_names and angle_values.size:
        angles_csv_path = output_dir / "angles.csv"
        _write_angles_csv(angles_csv_path, time_vector, angle_names, angle_values)
        written_files.append(angles_csv_path.name)

    manifest = {
        "session_dir": str(session_dir),
        "output_dir": str(output_dir),
        "variant": args.variant,
        "sampling_rate": sampling_rate,
        "frame_count": int(landmarks.shape[0]),
        "files": written_files,
    }
    manifest_path = output_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    written_files.append(manifest_path.name)

    print(f"[handtracker] Exported session: {session_dir}")
    print(f"[handtracker] Output directory: {output_dir}")
    for file_name in written_files:
        print(f"[handtracker] Wrote {file_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
