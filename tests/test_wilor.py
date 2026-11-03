"""Geometry, lifecycle and optional-dependency contracts without model downloads."""
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mavis_track import WiLoRTracker
from mavis_track import wilor


def prediction(right=1, offset=0):
    joints = np.zeros((21, 3), dtype=np.float32)
    joints[:, 0] = np.linspace(.01, .09, 21)
    joints[:, 1] = np.linspace(.02, .10, 21)
    joints[:, 2] = np.linspace(.003, .01, 21)
    return dict(joints=joints, vertices=np.tile(joints[0], (778, 1)),
                camera=np.array([1., .1, -.2]), center=np.array([320. + offset, 240.]),
                box_size=200., bbox=np.array([220., 140., 420., 340.]),
                score=.9, right=right)


@pytest.fixture
def fake_runtime(monkeypatch, tmp_path):
    for name in wilor.missing_assets(tmp_path):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    state = SimpleNamespace(items=[prediction()], images=[], closed=0)
    class Runtime:
        def __init__(self, root, device, fast):
            self.faces = np.array([[0, 1, 2]], dtype=np.int32)
            self.device = "cpu"
            self.cfg = SimpleNamespace(EXTRA=SimpleNamespace(FOCAL_LENGTH=5000),
                                       MODEL=SimpleNamespace(IMAGE_SIZE=256))
        def infer(self, bgr, confidence, max_hands):
            state.images.append(bgr.copy())
            return state.items
        def close(self):
            state.closed += 1
    monkeypatch.setattr(wilor, "_OfficialWiLoR", Runtime)
    return tmp_path, state


def test_import_is_lazy_and_missing_assets_do_not_import_torch(tmp_path):
    script = (
        "import sys; from mavis_track import WiLoRTracker; "
        "assert 'torch' not in sys.modules; assert 'ultralytics' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with pytest.raises(FileNotFoundError, match="MANO_RIGHT.pkl"):
        WiLoRTracker(root=tmp_path)


def test_full_image_projection_matches_analytic_perspective_and_left_reflection():
    item = prediction()
    before = item["joints"].copy()
    args = (item["joints"], item["vertices"], item["camera"],
            item["center"], item["box_size"], (640, 480), 1000.)
    right = wilor._project_hand(*args, 1)
    left = wilor._project_hand(*args, 0)
    # Centered crop, scale=1, size=200: camera depth=10 m, x=.1, y=-.2.
    point = before[0] + [.1, -.2, 10.]
    expected = point[:2] / point[2] * 1000 + [320., 240.]
    np.testing.assert_allclose(right[0][0], expected, atol=1e-6)
    np.testing.assert_allclose(left[0][:, 0], 640 - right[0][:, 0], atol=1e-6)
    np.testing.assert_allclose(left[0][:, 1], right[0][:, 1], atol=1e-6)
    np.testing.assert_allclose(right[2][0], [0, 0, 0])
    np.testing.assert_array_equal(item["joints"], before)


def test_crop_offset_and_wrist_origin_reconstruct_same_camera_points():
    item = prediction()
    image, _, joints, _, wrist = wilor._project_hand(
        item["joints"], item["vertices"], item["camera"], [360, 220],
        200., (640, 480), 1000., 1)
    np.testing.assert_allclose(wrist, item["joints"][0] + [.5, -.4, 10.], atol=1e-6)
    camera = joints + wrist
    np.testing.assert_allclose(image, camera[:, :2] / camera[:, 2:] * 1000 + [320, 240])


@pytest.mark.parametrize("change", [
    {"camera": [0, 0, 0]}, {"camera": [float("nan"), 0, 0]},
    {"box_size": -1}, {"right": 2}, {"vertices": np.zeros((10, 3))},
])
def test_invalid_geometry_is_not_exposed_as_valid_tracking(change):
    item = prediction()
    item.update(change)
    with pytest.raises(ValueError):
        wilor._project_hand(item["joints"], item["vertices"], item["camera"],
                            item["center"], item["box_size"], (640, 480), 1000., item["right"])


def test_api_color_ownership_coordinates_and_close(fake_runtime):
    root, state = fake_runtime
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:, :, 0] = 200
    original = image.copy()
    with WiLoRTracker(root=root, mirrored=True) as tracker:
        result = tracker.process(image, color="rgb", timestamp=1)
        hand = result.hands[0]
        assert hand.handedness == "Left" and hand.detection_score == .9
        assert result.image_space == "pixels"
        assert result.joint_space == "wrist_relative_camera_meters"
        assert result.translation_space == "model_estimated_camera_meters"
        assert result.timestamp_clock == "source"
        assert state.images[0][0, 0].tolist() == [0, 0, 200]
        assert hand.joints.shape == (21, 3) and hand.vertices.shape == (778, 3)
        assert len(hand.angles) == 14
        for array in (hand.image_landmarks, hand.image_vertices, hand.joints, hand.vertices,
                      hand.wrist_camera_translation, hand.bbox):
            assert not array.flags.writeable and array.dtype == np.float32
        state.items[0]["joints"][:] = 99
        assert not np.any(hand.joints == 99)
        np.testing.assert_array_equal(image, original)
    tracker.close()
    assert state.closed == 1
    with pytest.raises(RuntimeError, match="closed"):
        tracker.process(image, timestamp=2)


def test_adjacent_ids_not_reused_after_missing_hand_or_long_gap(fake_runtime):
    root, state = fake_runtime
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with WiLoRTracker(root=root) as tracker:
        first = tracker.process(frame, timestamp=1).hands[0].track_id
        assert tracker.process(frame, timestamp=1.1).hands[0].track_id == first
        state.items = []
        assert tracker.process(frame, timestamp=1.2).hands == ()
        state.items = [prediction()]
        second = tracker.process(frame, timestamp=1.3).hands[0].track_id
        assert second != first
        assert tracker.process(frame, timestamp=2).hands[0].track_id != second


def test_bad_inputs_and_timestamps_do_not_advance_state(fake_runtime):
    root, state = fake_runtime
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    with WiLoRTracker(root=root) as tracker:
        tracker.process(frame, timestamp=2)
        for stamp in (2, 1, -1, float("nan")):
            with pytest.raises(ValueError, match="timestamps"):
                tracker.process(frame, timestamp=stamp)
        for invalid in (np.zeros((3, 3), np.uint8), np.zeros((3, 3, 3), np.float32)):
            with pytest.raises(ValueError, match="frame"):
                tracker.process(invalid, timestamp=3)
        result = tracker.process(frame, timestamp=3)
        assert result.frame_id == 1 and len(state.images) == 2


def test_wilor_overlay_draws_mesh_and_joints_without_changing_result(fake_runtime):
    from handtrack.applications._wilor_preview import WiLoRModeTracker
    root, _ = fake_runtime
    tracker = WiLoRModeTracker(root=root)
    try:
        frame = np.zeros((480, 640, 3), np.uint8)
        result = tracker.process(frame)
        points = result.hands[0].image_landmarks.copy()
        tracker.draw(frame, result)
        assert np.any(frame)
        np.testing.assert_array_equal(result.hands[0].image_landmarks, points)
        with pytest.raises(ValueError, match="hands only"):
            tracker.set_mode("pose")
    finally:
        tracker.close()


def test_cli_forwards_wilor_options_without_optitrack_auto_routing(monkeypatch):
    from handtrack.applications import _cli, mediapipe_gui
    called = []
    monkeypatch.setattr(_cli, "_optitrack_sdk_available", lambda: True)
    monkeypatch.setattr(mediapipe_gui, "main", lambda argv: called.extend(argv) or 0)
    assert _cli.main(["gui", "--model", "wilor", "--wilor-root", "X:/WiLoR",
                      "--wilor-device", "cuda", "--wilor-fast"]) == 0
    assert "--model" in called and called[called.index("--model") + 1] == "wilor"
    assert called[called.index("--wilor-root") + 1] == "X:/WiLoR"
    assert "--wilor-fast" in called


@pytest.mark.parametrize("args", [
    ["demo", "--model", "wilor", "--mode", "pose"],
    ["gui", "--model", "wilor", "--backend", "webcam", "--advanced-hands"],
])
def test_unimplemented_wilor_workflows_are_rejected_instead_of_falling_back(args):
    from handtrack.applications import _cli
    with pytest.raises(SystemExit) as error:
        _cli.main(args)
    assert error.value.code == 2



def test_wilor_doctor_reports_missing_assets_without_loading_checkpoints(monkeypatch, tmp_path, capsys):
    from handtrack.applications import _cli, _wilor_doctor
    monkeypatch.setattr(_wilor_doctor.importlib.util, "find_spec", lambda _: None)
    monkeypatch.setitem(sys.modules, "torch", None)
    assert _cli.main(["doctor", "--model", "wilor", "--wilor-root", str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert "MANO_RIGHT.pkl" in output and "not tested" in output
    assert "Camera SDK" not in output



def test_detector_loader_disables_unscoped_installs_and_sync_even_if_preloaded(monkeypatch, tmp_path):
    settings = {"sync": True}
    checks = SimpleNamespace(AUTOINSTALL=True)
    utils = SimpleNamespace(ONLINE=True)
    detector = object()
    imported = []
    def load(name):
        assert os.environ["YOLO_AUTOINSTALL"] == "false"
        imported.append(name)
        if name == "ultralytics":
            return SimpleNamespace(YOLO=detector, settings=settings)
        if name == "ultralytics.utils":
            return utils
        assert name == "ultralytics.utils.checks"
        return checks
    monkeypatch.setenv("YOLO_AUTOINSTALL", "true")
    monkeypatch.delenv("YOLO_CONFIG_DIR", raising=False)
    monkeypatch.setattr(wilor.importlib, "import_module", load)
    assert wilor._detector_class(tmp_path) is detector
    assert checks.AUTOINSTALL is False and settings["sync"] is False
    assert utils.ONLINE is False and os.environ["YOLO_OFFLINE"] == "true"
    assert os.environ["YOLO_CONFIG_DIR"] == str(tmp_path / ".mavis-runtime")
    assert imported == ["ultralytics", "ultralytics.utils", "ultralytics.utils.checks"]



def test_wilor_doctor_cli_keeps_requested_backend_and_gui(monkeypatch):
    from handtrack.applications import _cli, _wilor_doctor
    calls = []
    monkeypatch.setattr(_wilor_doctor, "run", lambda root, **kwargs: calls.append((root, kwargs)) or 1)
    assert _cli.main(["doctor", "--model", "wilor", "--backend", "optitrack",
                      "--gui", "--wilor-root", "assets"]) == 1
    assert calls == [("assets", {"backend": "optitrack", "gui": True})]



def test_wheel_install_finds_assets_beside_project_environment(monkeypatch, tmp_path):
    environment = tmp_path / ".venv-wilor"
    monkeypatch.setattr(sys, "prefix", str(environment))
    monkeypatch.setattr(wilor, "__file__", str(environment / "Lib/site-packages/mavis_track/wilor.py"))
    monkeypatch.delenv("MAVIS_WILOR_ROOT", raising=False)
    expected = tmp_path / "external/WiLoR"
    source = expected / "wilor/models/wilor.py"
    source.parent.mkdir(parents=True)
    source.touch()
    assert wilor.wilor_root() == expected



def test_wilor_auto_doctor_does_not_require_unavailable_optitrack(monkeypatch):
    from handtrack.applications import _cli, _wilor_doctor
    calls = []
    monkeypatch.setattr(_cli, "_optitrack_sdk_available", lambda: False)
    monkeypatch.setattr(_wilor_doctor, "run", lambda root, **kwargs: calls.append(kwargs) or 0)
    assert _cli.main(["doctor", "--model", "wilor"]) == 0
    assert calls == [{"backend": "webcam", "gui": False}]
