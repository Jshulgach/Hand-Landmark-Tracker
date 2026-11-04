"""Calibration rejections must identify failed gates without publishing candidates."""
from types import SimpleNamespace

import numpy as np
import pytest

from handtrack.applications import _calibration_capture as capture
from handtrack.applications import _calibration_runtime as runtime
from handtrack.applications import _inspect_calibration as inspect


@pytest.fixture
def calibration(monkeypatch, tmp_path):
    destination = tmp_path / "latest.npz"
    destination.write_bytes(b"previous validated calibration")
    result = SimpleNamespace(heldout_errors=np.array([.3, .4]), per_view_errors=np.array([.2]),
        inlier_indices=np.array([0]), outlier_indices=np.array([], dtype=int),
        heldout_indices=np.array([1, 2]), std_intrinsics=np.zeros(18), model_flags=0,
        candidate_flags=np.array([0]), candidate_heldout_rms=np.array([.35]))
    board = SimpleNamespace(shape_rms_mm=.5, scale_error_percent=.4,
                            planarity_rms_mm=.2, num_views=7, num_points=140)
    pair = SimpleNamespace(camera_a=0, camera_b=1, heldout_errors=np.array([.4]),
        rotation=np.eye(3), translation=np.array([[-.3], [0], [0]]),
        rms=.3, inlier_indices=np.arange(20))
    for name, value in {"CALIBRATION_FILE": str(destination), "_camera_ids": [0, 1],
            "_image_sizes": None, "_diagnostics": None, "_intrinsic_results": [result, result],
            "_stereo_results": [pair], "_board_validation": board,
            "_reserved_indices": [4, 9, 14], "CHARUCO_SQUARE_LENGTH": .04,
            "CHARUCO_MARKER_LENGTH": .03}.items():
        monkeypatch.setattr(runtime, name, value, raising=False)
    def publish():
        runtime.save_calibration(2, (1280, 1024), 36, [np.diag([900., 900., 1.])]*2,
            [np.zeros(5)]*2, [np.eye(3)]*2, [np.zeros(3), np.array([-.3, 0, 0])])
    return destination, board, pair, result, publish


@pytest.mark.parametrize("field,value,reason", [
    ("shape_rms_mm", 4., "Board shape RMS: 4.000 mm (limit 2.000 mm)"),
    ("scale_error_percent", -3., "Board scale error: 3.000 % (limit 2.000 %)"),
    ("planarity_rms_mm", 2., "Board planarity RMS: 2.000 mm (limit 1.000 mm)"),
    ("num_views", 2, "Held-out board views: 2; need at least 3"),
])
def test_low_intrinsic_rms_does_not_hide_failed_board_metric(calibration, field, value, reason):
    destination, board, _, _, publish = calibration
    setattr(board, field, value)
    with pytest.raises(runtime.CalibrationQualityError, match="previous calibration preserved") as error:
        publish()
    assert reason in str(error.value)
    assert destination.read_bytes() == b"previous validated calibration"
    with np.load(str(destination)+".candidate.npz", allow_pickle=False) as data:
        assert not data["quality_passed"].item()
        assert reason in data["quality_failure_reasons"].tolist()


def test_stereo_failure_is_visible_in_old_and_new_candidate_inspection(calibration, monkeypatch, capsys):
    destination, _, pair, _, publish = calibration
    pair.heldout_errors = np.array([.4, 3.2])
    with pytest.raises(runtime.CalibrationQualityError, match="Stereo 0-1"):
        publish()
    monkeypatch.setattr(inspect, "_load_backend_config",
                        lambda backend: SimpleNamespace(CALIBRATION_FILE=str(destination)))
    assert inspect.main(["--backend", "optitrack", "--candidate"]) == 1
    output = capsys.readouterr().out
    assert "Stereo 0-1" in output and "max 3.200 px" in output
    assert "board_shape_rms_mm" in output
    assert destination.read_bytes() == b"previous validated calibration"
    # Older candidates lack saved failure reasons/thresholds but retain metrics.
    path = str(destination)+".candidate.npz"
    with np.load(path, allow_pickle=False) as data:
        old = {key: data[key] for key in data.files if key != "quality_failure_reasons"}
    np.savez(path, **old)
    assert inspect.main(["--backend", "optitrack", "--candidate"]) == 1
    assert "max 3.200 px" in capsys.readouterr().out


def test_quality_rejection_has_nonzero_exit_and_clean_shutdown(calibration, monkeypatch, capsys):
    stopped = []
    manager = SimpleNamespace(stop_all=lambda: stopped.append(True))
    module = SimpleNamespace(CameraManager=lambda: manager,
        calibrate_intrinsics=object(), calibrate_stereo_pairs=object(), save_calibration=object())
    original = vars(module).copy()
    monkeypatch.setattr(runtime, "_load_backend", lambda backend: original["CameraManager"])
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: module)
    # main resets solver state before capture, so inject the rejection at the boundary.
    monkeypatch.setattr(runtime, "save_calibration",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            runtime.CalibrationQualityError("Board shape RMS: 4 mm")))
    module._capture_main = lambda: (module.CameraManager(), module.save_calibration())
    assert runtime.main("optitrack") == 1
    assert stopped == [True]
    assert module.CameraManager is original["CameraManager"]
    output = capsys.readouterr()
    assert "Board shape RMS: 4 mm" in output.out
    assert "Traceback" not in output.out + output.err


def test_uid_calibration_passes_saved_diagnostics_to_final_validation(calibration, monkeypatch):
    destination, _, _, _, publish = calibration
    monkeypatch.setattr(runtime, "_diagnostics", [
        {"uid": (1, 11), "serial": "A"}, {"uid": (1, 22), "serial": "B"}])
    publish()
    with np.load(destination, allow_pickle=False) as data:
        assert data["quality_passed"].item()
        assert data["camera_uid_low"].tolist() == [11, 22]


def board_results(offset=0., reverse=False):
    points = np.column_stack((np.arange(6, dtype=float), np.zeros(6))) + offset
    ids = np.arange(6).reshape(-1, 1)
    if reverse:
        points, ids = points[::-1], ids[::-1]
    return [(True, (points.reshape(-1, 1, 2), ids)) for _ in range(4)]


def test_capture_gate_waits_for_still_board_and_matches_corner_ids():
    gate = capture.BoardCaptureGate()
    assert not gate.check(board_results(), 10.)[0]
    assert not gate.check(board_results(5.), 10.3)[0]  # Moving board resets anchor.
    assert not gate.check(board_results(5., reverse=True), 10.6)[0]
    assert gate.check(board_results(5., reverse=True), 10.81)[0]
    assert not gate.check([(False, (None, None))]*4, 11.)[0]
    assert not gate.check(board_results(5.), 11.1)[0]


def test_capture_gate_rejects_stale_skewed_and_duplicate_frames():
    gate = capture.BoardCaptureGate()
    result = board_results()
    assert not gate.check(result, 10., [9.5]*4)[0]
    assert not gate.check(result, 10., [9.9, 10., 10., 10.])[0]
    assert not gate.check(result, 10., [10.]*4)[0]
    assert not gate.check(result, 10.1, [10.]*4)[0]
    assert not gate.check(result, 10.3, [10.3]*4)[0]
    assert gate.check(result, 10.6, [10.6]*4)[0]
    assert not gate.check(result, 10.7, [float("nan")]*4)[0]


def test_cli_forwards_candidate_flag(monkeypatch):
    from handtrack.applications import _cli
    calls = []
    monkeypatch.setattr(inspect, "main", lambda argv: calls.append(argv) or 1)
    assert _cli.main(["inspect-calibration", "--backend", "optitrack", "--candidate"]) == 1
    assert calls == [["--backend", "optitrack", "--candidate"]]


def test_corrupt_candidate_quality_metrics_are_reported_without_traceback(tmp_path, monkeypatch, capsys):
    destination = tmp_path / "latest.npz"
    np.savez(str(destination)+".candidate.npz", num_cameras=2, quality_passed=False,
             heldout_errors_0=np.array(["not a number"]))
    monkeypatch.setattr(inspect, "_load_backend_config",
                        lambda backend: SimpleNamespace(CALIBRATION_FILE=str(destination)))
    assert inspect.main(["--backend", "optitrack", "--candidate"]) == 1
    assert "unreadable" in capsys.readouterr().out
