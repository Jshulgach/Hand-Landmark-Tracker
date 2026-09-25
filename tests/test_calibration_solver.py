import cv2
import numpy as np

from handtrack.processing._calibration_solver import (
    StereoResult,
    calibrate_intrinsics_robust,
    point_rmse,
    robust_error_threshold,
    solve_camera_pose_graph,
    validate_heldout_board_reconstruction,
)


def _synthetic_intrinsic_observations(count=36):
    rng = np.random.default_rng(741)
    object_grid = np.zeros((8 * 6, 3), np.float32)
    object_grid[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2) * 0.04
    camera_matrix = np.array(
        [[910.0, 0.0, 638.0], [0.0, 900.0, 509.0], [0.0, 0.0, 1.0]]
    )
    dist_coeffs = np.array([[-0.12, 0.035, 0.001, -0.0008, -0.006]])
    object_points = []
    image_points = []
    for index in range(count):
        rvec = np.array(
            [
                0.35 * np.sin(index * 0.37),
                0.42 * np.cos(index * 0.29),
                0.15 * np.sin(index * 0.19),
            ]
        )
        tvec = np.array(
            [
                0.16 * np.sin(index * 0.51) - 0.12,
                0.12 * np.cos(index * 0.43) - 0.10,
                0.72 + 0.20 * (index % 5) / 4,
            ]
        )
        projected, _ = cv2.projectPoints(
            object_grid, rvec, tvec, camera_matrix, dist_coeffs
        )
        noise_scale = 5.0 if index == 10 else 0.12
        projected += rng.normal(0.0, noise_scale, projected.shape)
        object_points.append(object_grid.copy())
        image_points.append(projected.astype(np.float32))
    return object_points, image_points, camera_matrix


def test_point_rmse_is_pointwise_euclidean_rms():
    observed = np.zeros((2, 1, 2), dtype=np.float32)
    projected = np.array([[[3.0, 4.0]], [[0.0, 0.0]]], dtype=np.float32)

    assert point_rmse(observed, projected) == np.sqrt(12.5)


def test_robust_threshold_rejects_large_tail():
    threshold = robust_error_threshold([0.20, 0.21, 0.22, 0.23, 4.0])

    assert 0.23 < threshold < 1.0


def test_robust_intrinsics_reject_outlier_and_recover_focal_lengths():
    object_points, image_points, expected_matrix = _synthetic_intrinsic_observations()

    result = calibrate_intrinsics_robust(
        object_points,
        image_points,
        (1280, 1024),
        min_inliers=20,
    )

    assert 10 in result.outlier_indices
    assert result.rms < 0.4
    assert np.max(result.heldout_errors) < 0.5
    np.testing.assert_allclose(
        np.diag(result.camera_matrix)[:2],
        np.diag(expected_matrix)[:2],
        rtol=0.025,
    )
    assert result.std_intrinsics.size > 0


def _make_pair(camera_a, camera_b, rotations, translations):
    relative_rotation = rotations[camera_b] @ rotations[camera_a].T
    relative_translation = (
        translations[camera_b] - relative_rotation @ translations[camera_a]
    )
    return StereoResult(
        camera_a=camera_a,
        camera_b=camera_b,
        rotation=relative_rotation,
        translation=relative_translation,
        rms=0.25,
        per_view_errors=np.full(30, 0.25),
        heldout_errors=np.full(6, 0.25),
        inlier_indices=np.arange(30),
        outlier_indices=np.empty(0, dtype=np.int32),
        heldout_indices=np.arange(6),
    )


def test_pose_graph_recovers_all_camera_poses():
    rotations = [np.eye(3)]
    translations = [np.zeros((3, 1))]
    for rvec, translation in (
        ([0.02, -0.16, 0.01], [0.42, 0.01, 0.04]),
        ([-0.03, 0.19, -0.01], [-0.39, 0.03, 0.06]),
        ([0.14, 0.02, 0.01], [0.03, 0.35, 0.10]),
    ):
        rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
        rotations.append(rotation)
        translations.append(np.asarray(translation, dtype=np.float64).reshape(3, 1))
    pairs = [
        _make_pair(a, b, rotations, translations)
        for a in range(4)
        for b in range(a + 1, 4)
    ]

    solved_rotations, solved_translations, loop_rms = solve_camera_pose_graph(4, pairs)

    assert loop_rms < 1e-8
    for expected, actual in zip(rotations, solved_rotations):
        np.testing.assert_allclose(actual, expected, atol=1e-8)
    for expected, actual in zip(translations, solved_translations):
        np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_heldout_board_reconstruction_recovers_shape_scale_and_plane():
    camera_matrix = np.array(
        [[850.0, 0.0, 640.0], [0.0, 850.0, 512.0], [0.0, 0.0, 1.0]]
    )
    camera_matrices = [camera_matrix.copy() for _ in range(3)]
    dist_coeffs = [np.zeros((1, 5)) for _ in range(3)]
    rotations = [np.eye(3) for _ in range(3)]
    translations = [
        np.array([[0.0], [0.0], [0.0]]),
        np.array([[-0.35], [0.0], [0.0]]),
        np.array([[0.0], [-0.30], [0.0]]),
    ]
    board_points = np.zeros((5 * 4, 3), dtype=np.float32)
    board_points[:, :2] = np.mgrid[0:5, 0:4].T.reshape(-1, 2) * 0.04
    all_corners = [[] for _ in range(3)]
    all_ids = [[] for _ in range(3)]
    ids = np.arange(len(board_points), dtype=np.int32).reshape(-1, 1)
    for view in range(4):
        board_rotation, _ = cv2.Rodrigues(
            np.array([0.05 * view, -0.04 * view, 0.02 * view])
        )
        world_points = (board_rotation @ board_points.T).T + np.array(
            [-0.08, -0.06, 0.8 + 0.05 * view]
        )
        for camera in range(3):
            projected, _ = cv2.projectPoints(
                world_points,
                np.zeros(3),
                translations[camera],
                camera_matrices[camera],
                dist_coeffs[camera],
            )
            all_corners[camera].append(projected.astype(np.float32))
            all_ids[camera].append(ids.copy())

    result = validate_heldout_board_reconstruction(
        all_corners,
        all_ids,
        board_points,
        camera_matrices,
        dist_coeffs,
        rotations,
        translations,
        [0, 1, 2, 3],
    )

    assert result.num_views == 4
    assert result.num_points == 80
    assert result.shape_rms_mm < 0.01
    assert result.scale_error_percent < 0.01
    assert result.planarity_rms_mm < 0.01
