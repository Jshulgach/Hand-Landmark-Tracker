from pathlib import Path

from handtrack.applications import _evaluate_placement


def test_calibration_path_falls_back_when_backend_config_is_absent(monkeypatch):
    module_name = "handtrack.cameras.optitrack.config"

    def missing_config(_name):
        raise ModuleNotFoundError(name=module_name)

    monkeypatch.setattr(_evaluate_placement.importlib, "import_module", missing_config)

    path = _evaluate_placement._calibration_path("optitrack")

    assert path == (
        Path(_evaluate_placement.__file__).resolve().parents[1]
        / "cameras"
        / "optitrack"
        / "calibration_data"
        / "multi_camera_calib_latest.npz"
    )
