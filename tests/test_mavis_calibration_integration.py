from types import SimpleNamespace

import cv2
import numpy as np

from handtrack.applications import _calibration_runtime as runtime
from mavis_track.calibration import load_camera_calibration
from mavis_track._files import load_npz


def test_synthetic_calibration_reserves_views_and_publishes_validated_board(monkeypatch, tmp_path):
    runtime._load_backend('webcam')
    for name, value in (('_camera_ids', [0, 1]), ('_image_sizes', None), ('_diagnostics', None),
                        ('_intrinsic_results', []), ('_stereo_results', []),
                        ('_reserved_indices', []), ('_board_validation', None)):
        monkeypatch.setattr(runtime, name, value)
    destination = tmp_path / 'calibration.npz'
    monkeypatch.setattr(runtime, 'CALIBRATION_FILE', str(destination))
    grid = np.zeros((6 * 5, 3), np.float32)
    grid[:, :2] = np.mgrid[0:6, 0:5].T.reshape(-1, 2) * .04
    K = np.array([[900., 0, 640], [0, 890, 512], [0, 0, 1.]])
    corners = [[], []]
    ids = [[], []]
    rng = np.random.default_rng(920)
    for view in range(36):
        rotation = np.array([.3 * np.sin(view * .31), .3 * np.cos(view * .27), .1 * np.sin(view * .51)])
        position = np.array([-.08 + .1 * np.sin(view * .33), -.06 + .08 * np.cos(view * .41), .85 + .15 * np.sin(view * .23)])
        for camera in range(2):
            projected, _ = cv2.projectPoints(grid, rotation, position + [-.35 * camera, 0, 0], K, np.zeros(5))
            projected += rng.normal(0, .08, projected.shape)
            corners[camera].append(projected.astype(np.float32))
            ids[camera].append(np.arange(len(grid), dtype=np.int32).reshape(-1, 1))
    board = SimpleNamespace(getChessboardCorners=lambda: grid)
    matrices, dists = runtime.calibrate_intrinsics(corners, ids, board, (1280, 1024), 2)
    rotations, translations = runtime.calibrate_stereo_pairs(corners, ids, board, matrices, dists, (1280, 1024), 2)
    runtime.save_calibration(2, (1280, 1024), 36, matrices, dists, rotations, translations)
    data = load_npz(destination)
    assert data['quality_passed'].item()
    assert data['reserved_view_indices'].tolist() == [4, 9, 14, 19, 24, 29, 34]
    assert data['board_validation_views'].item() == 7
    assert data['board_shape_rms_mm'].item() < 2
    assert abs(data['board_scale_error_percent'].item()) < 2
    assert data['board_planarity_rms_mm'].item() < 1
    assert not destination.with_name(destination.name + '.candidate.npz').exists()
    load_camera_calibration(destination, count=2, camera_ids=[0, 1], image_sizes=[(1280, 1024)] * 2)
    # Synthetic geometry confirms software behavior, not physical accuracy.
