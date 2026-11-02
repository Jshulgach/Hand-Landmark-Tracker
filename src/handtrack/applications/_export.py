"""Stream validated hand/body sessions into rectangular CSV files."""

import argparse
import csv
import json
from contextlib import ExitStack
from itertools import chain
from pathlib import Path

from mavis_track import BODY_LANDMARK_NAMES, QualityFlag, iter_session_chunks


def _writers(stack, output, session, variant, skip_angles):
    hands = session.landmarks.shape[1]
    headers = {f"{variant}.csv": ["frame", "timestamp"] + [
        f"{'' if hands == 1 else f'hand{hand}_'}lm{index:02d}_{axis}"
        for hand in range(hands) for index in range(21) for axis in "xyz"]}
    if not skip_angles and session.angle_names:
        headers["angles.csv"] = ["frame", "timestamp"] + [
            f"hand{hand}_{name}" if hands > 1 else name
            for hand in range(hands) for name in session.angle_names]
    if not session.legacy_single_hand:
        headers["hands.csv"] = ["frame", "timestamp", "slot", "valid", "track_id", "handedness"]
    if session.body_landmarks is not None:
        headers["body_landmarks.csv"] = ["frame", "timestamp", "valid", "landmark_index", "landmark_name",
                                         "x", "y", "z", "visibility", "world_x", "world_y", "world_z"]
        headers["tracking_quality.csv"] = ["frame", "timestamp", "region", "flags", "flag_names"]
    writers = {}
    for name, header in headers.items():
        handle = stack.enter_context((output / (name + ".tmp")).open("w", newline="", encoding="utf-8"))
        writer = csv.writer(handle)
        writer.writerow(header)
        writers[name] = writer
    return writers


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--variant", choices=("landmarks", "raw_landmarks"), default="landmarks")
    parser.add_argument("--skip-angles", action="store_true")
    args = parser.parse_args(argv)
    source = Path(args.session)
    output = Path(args.output_dir) if args.output_dir else (source if source.is_dir() else source.parent) / "exports"
    files = []
    try:
        iterator = iter_session_chunks(source)
        first = next(iterator)
        hands = first.landmarks.shape[1]
        output.mkdir(parents=True, exist_ok=True)
        frame = 0
        first_size, varies = None, False
        with ExitStack() as stack:
            writers = _writers(stack, output, first, args.variant, args.skip_angles)
            files = list(writers)
            for session in chain((first,), iterator):
                points = getattr(session, args.variant)
                for index, stamp in enumerate(session.time_vector):
                    prefix = [frame, f"{stamp:.6f}"]
                    writers[f"{args.variant}.csv"].writerow(prefix + [f"{v:.6f}" for v in points[index].reshape(-1)])
                    if "angles.csv" in writers:
                        writers["angles.csv"].writerow(prefix + [f"{v:.6f}" for v in session.angle_values[index].reshape(-1)])
                    if "hands.csv" in writers:
                        for hand in range(hands):
                            writers["hands.csv"].writerow(prefix + [hand, bool(session.valid[index, hand]),
                                                                  session.track_ids[index, hand], session.handedness[index, hand]])
                    if "body_landmarks.csv" in writers:
                        for landmark, name in enumerate(BODY_LANDMARK_NAMES):
                            writers["body_landmarks.csv"].writerow(prefix + [bool(session.body_valid[index]), landmark, name,
                                *session.body_landmarks[index, landmark], session.body_visibility[index, landmark],
                                *session.body_world_landmarks[index, landmark]])
                        for column, region in enumerate(session.quality_region_names):
                            flag = QualityFlag(int(session.quality_flags[index, column]))
                            writers["tracking_quality.csv"].writerow(prefix + [region, int(flag), flag.name])
                    if session.image_sizes is not None:
                        size = session.image_sizes[index].tolist()
                        if first_size is None:
                            first_size = size
                        varies = varies or size != first_size
                    frame += 1
        manifest = {"variant": args.variant, "sampling_rate": first.sampling_rate, "frame_count": frame,
                        "hand_slots": hands, "coordinate_space": first.coordinate_space,
                        "timestamp_clock": first.timestamp_clock, "files": files}
        if first.body_landmarks is not None:
            manifest.update(tracking_mode=first.tracking_mode,
                            body_world_coordinate_space=first.body_world_coordinate_space,
                            quality_region_names=list(first.quality_region_names),
                            quality_flag_definitions={"OK": 0, **{flag.name: int(flag) for flag in QualityFlag}},
                            quality_visibility_threshold=first.quality_visibility_threshold,
                            quality_max_gap=first.quality_max_gap)
        if first_size is not None:
            manifest.update(first_image_size=first_size, image_size_varies=varies)
        for name in files:
            (output / (name + ".tmp")).replace(output / name)
        temporary = output / "export_manifest.tmp.json"
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(output / "export_manifest.json")
    except (ValueError, OSError, StopIteration) as exc:
        print(f"[mavis] Cannot export session: {exc}")
        return 1
    finally:
        for name in files:
            (output / (name + ".tmp")).unlink(missing_ok=True)
    print(f"[mavis] Exported session: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
