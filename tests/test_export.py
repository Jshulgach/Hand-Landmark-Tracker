import csv
import json

import numpy as np
from handtrack.applications import _export


def test_export_writes_landmark_and_angle_csv_files(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    landmarks = np.arange(2 * 21 * 3, dtype=np.float32).reshape(2, 21, 3) / 100.0
    raw_landmarks = landmarks + 1.0
    time_vector = np.array([0.0, 0.1], dtype=np.float64)
    angle_names = np.array(["thumb_ip", "index_pip"], dtype=str)
    angle_values = np.array([[10.0, 20.0], [11.0, 21.0]], dtype=np.float32)
    np.savez(
        session_dir / "landmarks.npz",
        landmarks=landmarks,
        raw_landmarks=raw_landmarks,
        angle_names=angle_names,
        angle_values=angle_values,
        sampling_rate=10,
        time_vector=time_vector,
    )

    output_dir = session_dir / "csv"

    exit_code = _export.main(
        [
            str(session_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0

    with (output_dir / "landmarks.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0][0:5] == ["frame", "timestamp", "lm00_x", "lm00_y", "lm00_z"]
    assert rows[1][0:2] == ["0", "0.000000"]
    assert len(rows) == 3

    with (output_dir / "angles.csv").open(newline="", encoding="utf-8") as handle:
        angle_rows = list(csv.reader(handle))
    assert angle_rows[0] == ["frame", "timestamp", "thumb_ip", "index_pip"]
    assert angle_rows[2] == ["1", "0.100000", "11.000000", "21.000000"]

    manifest = json.loads(
        (output_dir / "export_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["variant"] == "landmarks"
    assert manifest["frame_count"] == 2
    assert manifest["files"] == ["landmarks.csv", "angles.csv"]
