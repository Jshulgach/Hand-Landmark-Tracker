import numpy as np

from handtrack.processing._camera_geometry import build_dlt_matrix, triangulation_covariance
from handtrack.processing._joint_angles import compute_all_joint_angles
from handtrack.processing._kalman_filter import Kalman3D


def camera_setup():
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1.]])
    return [K @ np.column_stack((np.eye(3), [offset, 0, 0])) for offset in (-.2, .2)]


def project(P, point):
    value = P @ np.append(point, 1)
    return value[:2] / value[2]


def test_covariance_matches_finite_difference_and_camera_scale():
    cameras = camera_setup()
    point = np.array([.08, -.04, 2.3])
    eps = 1e-6
    J = np.column_stack([np.concatenate([
        (project(P, point + np.eye(3)[axis] * eps) - project(P, point - np.eye(3)[axis] * eps)) / (2 * eps)
        for P in cameras]) for axis in range(3)])
    A = build_dlt_matrix(point, cameras)
    np.testing.assert_allclose(A[:, :3], -J, atol=1e-7)
    expected = .2 ** 2 * np.linalg.inv(J.T @ J)
    np.testing.assert_allclose(triangulation_covariance(A, .2), expected, rtol=1e-7, atol=1e-12)
    scaled = build_dlt_matrix(point, [cameras[0] * 100, cameras[1] * .01])
    np.testing.assert_allclose(triangulation_covariance(scaled, .2), expected, rtol=1e-7, atol=1e-12)


def test_predicted_covariance_agrees_with_independent_noisy_triangulation():
    cameras = camera_setup()
    point = np.array([.08, -.04, 2.3])
    sigma = .2
    pixels = np.array([project(P, point) for P in cameras])
    observations = pixels + np.random.default_rng(42).normal(0, sigma, (6000, 2, 2))
    rows = []
    for index, P in enumerate(cameras):
        rows.extend([observations[:, index, axis, None] * P[2] - P[axis] for axis in range(2)])
    _, _, V = np.linalg.svd(np.stack(rows, axis=1))
    recovered = V[:, -1, :3] / V[:, -1, 3, None]
    empirical = np.cov(recovered.T)
    predicted = triangulation_covariance(build_dlt_matrix(point, cameras), sigma)
    np.testing.assert_allclose(np.diag(empirical), np.diag(predicted), rtol=.08)


def test_angles_are_named_finger_triples_and_degenerate_angles_are_missing():
    points = np.column_stack((np.arange(21), np.zeros((21, 2)))).astype(float)
    points[0] = [-1, 0, 0]
    points[5:8] = [[0, 0, 0], [1, 0, 0], [1, 1, 0]]
    angles = compute_all_joint_angles(points)
    assert len(angles) == 14
    assert angles['index_mcp'] == 0 and angles['index_pip'] == 90
    assert all(np.isnan(value) for value in compute_all_joint_angles(np.zeros((21, 3))).values())


def test_smoothing_first_measurement_is_not_pulled_to_origin():
    filter_ = Kalman3D()
    np.testing.assert_array_equal(filter_.update([1, 2, 3]), [1, 2, 3])
