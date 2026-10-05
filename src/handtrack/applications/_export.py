"""Export validated one- or two-hand sessions into rectangular CSV files."""

import argparse
import csv
import json
from pathlib import Path
from mavis_track import load_session


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--variant", choices=("landmarks", "raw_landmarks"), default="landmarks")
    parser.add_argument("--skip-angles", action="store_true")
    args = parser.parse_args(argv)
    source = Path(args.session)
    try:
        session = load_session(source)
    except (ValueError, OSError) as exc:
        print(f"[mavis] Cannot export session: {exc}")
        return 1
    output = Path(args.output_dir) if args.output_dir else (source if source.is_dir() else source.parent) / "exports"
    points = getattr(session, args.variant)
    hands = points.shape[1]
    header = ["frame", "timestamp"]
    for hand in range(hands):
        prefix = "" if hands == 1 else f"hand{hand}_"
        header.extend(f"{prefix}lm{index:02d}_{axis}" for index in range(21) for axis in "xyz")
    output.mkdir(parents=True, exist_ok=True)
    filename = f"{args.variant}.csv"
    with (output / filename).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index, (stamp, row) in enumerate(zip(session.time_vector, points)):
            writer.writerow([index, f"{stamp:.6f}"] + [f"{value:.6f}" for value in row.reshape(-1)])
    files = [filename]
    if not args.skip_angles and session.angle_names:
        with (output / "angles.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            columns = [f"hand{hand}_{name}" if hands > 1 else name
                       for hand in range(hands) for name in session.angle_names]
            writer.writerow(["frame", "timestamp", *columns])
            for index, (stamp, row) in enumerate(zip(session.time_vector, session.angle_values)):
                writer.writerow([index, f"{stamp:.6f}"] + [f"{value:.6f}" for value in row.reshape(-1)])
        files.append("angles.csv")
    if not session.legacy_single_hand:
        with (output / "hands.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame", "timestamp", "slot", "valid", "track_id", "handedness"])
            for frame, stamp in enumerate(session.time_vector):
                for hand in range(hands):
                    writer.writerow([frame, f"{stamp:.6f}", hand, bool(session.valid[frame, hand]),
                                     session.track_ids[frame, hand], session.handedness[frame, hand]])
        files.append("hands.csv")
    manifest = {"variant": args.variant, "sampling_rate": session.sampling_rate,
                "frame_count": len(points), "hand_slots": hands,
                "coordinate_space": session.coordinate_space, "timestamp_clock": session.timestamp_clock,
                "files": files}
    if session.image_sizes is not None and len(session.image_sizes):
        manifest["first_image_size"] = session.image_sizes[0].tolist()
        manifest["image_size_varies"] = bool((session.image_sizes != session.image_sizes[0]).any())
    (output / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[mavis] Exported session: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
