from handtrack.cameras.webcam import multi_webcam


def test_get_candidate_indices_prefers_explicit_override(monkeypatch):
    monkeypatch.setattr(multi_webcam, "WEBCAM_INDICES", [])

    assert multi_webcam._get_candidate_indices([2, 4]) == [2, 4]


def test_get_candidate_indices_uses_configured_indices(monkeypatch):
    monkeypatch.setattr(multi_webcam, "WEBCAM_INDICES", [1, 3])

    assert multi_webcam._get_candidate_indices() == [1, 3]


def test_default_candidate_indices_windows_uses_enumerated_device_count(monkeypatch):
    monkeypatch.setattr(multi_webcam.os, "name", "nt")
    monkeypatch.setattr(multi_webcam, "_get_windows_camera_count", lambda: 2)

    assert multi_webcam._default_candidate_indices() == [0, 1]


def test_default_candidate_indices_windows_falls_back_when_unknown(monkeypatch):
    monkeypatch.setattr(multi_webcam.os, "name", "nt")
    monkeypatch.setattr(multi_webcam, "_get_windows_camera_count", lambda: 0)

    assert multi_webcam._default_candidate_indices() == list(range(6))
