from handtrack.applications import gui


def test_webcam_gui_is_default(monkeypatch):
    called = {}

    def fake_run(module_name, passthrough_args):
        called["module"] = module_name
        called["args"] = passthrough_args
        return 0

    monkeypatch.delenv("HANDTRACK_BACKEND", raising=False)
    monkeypatch.setattr(gui, "_run_module_entrypoint", fake_run)
    monkeypatch.setattr(gui, "resolve_backend", lambda requested: requested or "webcam")

    assert gui.main([]) == 0
    assert called == {
        "module": "handtrack.applications.mediapipe_gui",
        "args": [],
    }


def test_optitrack_dispatch_does_not_forward_gui_token(monkeypatch):
    called = {}

    def fake_run(module_name, passthrough_args):
        called["module"] = module_name
        called["args"] = passthrough_args
        return 0

    monkeypatch.setattr(gui, "_run_module_entrypoint", fake_run)

    assert gui.main(["--backend", "optitrack"]) == 0
    assert called == {
        "module": "handtrack.applications.optitrack_gui",
        "args": [],
    }
