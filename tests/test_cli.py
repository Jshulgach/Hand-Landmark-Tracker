from handtrack.applications import _cli


def test_auto_backend_prefers_optitrack(monkeypatch):
    monkeypatch.setattr(_cli, "_optitrack_sdk_available", lambda: True)
    assert _cli._resolve_backend("auto") == "optitrack"


def test_auto_backend_falls_back_to_webcam(monkeypatch):
    monkeypatch.setattr(_cli, "_optitrack_sdk_available", lambda: False)
    assert _cli._resolve_backend("auto") == "webcam"


def test_gui_dispatches_to_selected_backend(monkeypatch):
    called = {}

    def fake_runner(module_name):
        called["module_name"] = module_name
        return 0

    monkeypatch.setattr(_cli, "_run_module_entrypoint", fake_runner)

    exit_code = _cli.main(["gui", "--backend", "webcam"])

    assert exit_code == 0
    assert called["module_name"] == "unity_hand_tracking.webcam.mocap_handracker_gui"


def test_board_dispatches_to_selected_backend(monkeypatch):
    called = {}

    def fake_runner(module_name):
        called["module_name"] = module_name
        return 0

    monkeypatch.setattr(_cli, "_run_module_entrypoint", fake_runner)

    exit_code = _cli.main(["board", "--backend", "webcam"])

    assert exit_code == 0
    assert called["module_name"] == "unity_hand_tracking.webcam.generate_charuco_board"


def test_test_sender_dispatches_to_selected_backend(monkeypatch):
    called = {}

    def fake_runner(module_name):
        called["module_name"] = module_name
        return 0

    monkeypatch.setattr(_cli, "_run_module_entrypoint", fake_runner)

    exit_code = _cli.main(["test-sender", "--backend", "optitrack"])

    assert exit_code == 0
    assert called["module_name"] == "unity_hand_tracking.optitrack_cam_py.test_sender"


def test_doctor_uses_resolved_backend(monkeypatch):
    called = {}

    monkeypatch.setattr(_cli, "_optitrack_sdk_available", lambda: True)

    def fake_doctor(selected_backend):
        called["backend"] = selected_backend
        return 0

    monkeypatch.setattr(_cli, "_run_doctor", fake_doctor)

    exit_code = _cli.main(["doctor"])

    assert exit_code == 0
    assert called["backend"] == "optitrack"


def test_inspect_calibration_uses_resolved_backend(monkeypatch):
    called = {}

    monkeypatch.setattr(_cli, "_optitrack_sdk_available", lambda: False)

    def fake_inspect(selected_backend):
        called["backend"] = selected_backend
        return 0

    monkeypatch.setattr(_cli, "_run_inspect_calibration", fake_inspect)

    exit_code = _cli.main(["inspect-calibration"])

    assert exit_code == 0
    assert called["backend"] == "webcam"


def test_benchmark_uses_resolved_backend(monkeypatch):
    called = {}

    monkeypatch.setattr(_cli, "_optitrack_sdk_available", lambda: True)

    def fake_benchmark(selected_backend, args):
        called["backend"] = selected_backend
        called["frames"] = args.frames
        return 0

    monkeypatch.setattr(_cli, "_run_benchmark", fake_benchmark)

    exit_code = _cli.main(["benchmark"])

    assert exit_code == 0
    assert called["backend"] == "optitrack"
    assert called["frames"] == 60


def test_record_dispatches_to_record_runner(monkeypatch):
    called = {"count": 0}

    def fake_record(args):
        called["count"] += 1
        called["source"] = args.source
        return 0

    monkeypatch.setattr(_cli, "_run_record", fake_record)

    exit_code = _cli.main(["record"])

    assert exit_code == 0
    assert called["count"] == 1
    assert called["source"] == "0"


def test_replay_dispatches_to_replay_runner(monkeypatch):
    called = {}

    def fake_replay(args):
        called["session"] = args.session
        called["width"] = args.width
        return 0

    monkeypatch.setattr(_cli, "_run_replay", fake_replay)

    exit_code = _cli.main(["replay", "recordings/demo-session"])

    assert exit_code == 0
    assert called["session"] == "recordings/demo-session"
    assert called["width"] == 960


def test_export_dispatches_to_export_runner(monkeypatch):
    called = {}

    def fake_export(args):
        called["session"] = args.session
        called["variant"] = args.variant
        return 0

    monkeypatch.setattr(_cli, "_run_export", fake_export)

    exit_code = _cli.main(
        ["export", "recordings/demo-session", "--variant", "raw_landmarks"]
    )

    assert exit_code == 0
    assert called["session"] == "recordings/demo-session"
    assert called["variant"] == "raw_landmarks"
