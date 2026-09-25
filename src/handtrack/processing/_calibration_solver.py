"""Robust calibration solvers and validation helpers.

This module is deliberately independent of camera acquisition and UI code so
saved ChArUco observations can be re-solved and tested without live hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class IntrinsicResult:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rms: float
    per_view_errors: np.ndarray
    heldout_errors: np.ndarray
    inlier_indices: np.ndarray
    outlier_indices: np.ndarray
    heldout_indices: np.ndarray
    std_intrinsics: np.ndarray
    model_flags: int
    candidate_flags: np.ndarray
    candidate_heldout_rms: np.ndarray


@dataclass(frozen=True)
class StereoResult:
    camera_a: int
    camera_b: int
    rotation: np.ndarray
    translation: np.ndarray
    rms: float
    per_view_errors: np.ndarray
    heldout_errors: np.ndarray
    inlier_indices: np.ndarray
    outlier_indices: np.ndarray
    heldout_indices: np.ndarray


@dataclass(frozen=True)
class BoardValidationResult:
    shape_rms_mm: float
    scale_error_percent: float
    planarity_rms_mm: float
    num_views: int
    num_points: int


def point_rmse(observed: np.ndarray, projected: np.ndarray) -> float:
    """Return true point-wise 2-D RMS reprojection error in pixels."""
    observed_2d = np.asarray(observed, dtype=np.float64).reshape(-1, 2)
    projected_2d = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
    if observed_2d.shape != projected_2d.shape or not len(observed_2d):
        raise ValueError(
            "Observed and projected points must have equal non-empty shapes"
        )
    residual = observed_2d - projected_2d
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def robust_error_threshold(
    errors: Sequence[float],
    *,
    sigma: float = 3.5,
    floor: float = 0.35,
    ceiling: float = 1.5,
) -> float:
    """Calculate a bounded median/MAD threshold for per-view errors."""
    values = np.asarray(errors, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("At least one finite error is required")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * mad
    threshold = median + sigma * max(robust_sigma, 1e-9)
    return float(np.clip(threshold, floor, ceiling))


def deterministic_holdout_indices(count: int, fraction: float = 0.2) -> np.ndarray:
    """Choose evenly distributed held-out indices without randomness."""
    if count < 2 or fraction <= 0:
        return np.empty(0, dtype=np.int32)
    holdout_count = max(1, int(round(count * fraction)))
    holdout_count = min(holdout_count, count - 1)
    return np.unique(np.linspace(0, count - 1, holdout_count, dtype=np.int32))


def _view_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    success, rvec, tvec = cv2.solvePnP(
        np.asarray(object_points, dtype=np.float32),
        np.asarray(image_points, dtype=np.float32),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return float("inf")
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    return point_rmse(image_points, projected)


def _fit_intrinsics(
    object_points: Sequence[np.ndarray],
    image_points: Sequence[np.ndarray],
    image_size: tuple[int, int],
    flags: int,
):
    result = cv2.calibrateCameraExtended(
        list(object_points),
        list(image_points),
        image_size,
        None,
        None,
        flags=flags,
    )
    (
        rms,
        camera_matrix,
        dist_coeffs,
        rvecs,
        tvecs,
        std_intrinsics,
        _std_extrinsics,
        _opencv_per_view,
    ) = result
    per_view = []
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
        per_view.append(point_rmse(img, projected))
    return (
        float(rms),
        camera_matrix,
        dist_coeffs,
        np.asarray(std_intrinsics).reshape(-1),
        np.asarray(per_view, dtype=np.float64),
    )


def calibrate_intrinsics_robust(
    object_points: Sequence[np.ndarray],
    image_points: Sequence[np.ndarray],
    image_size: tuple[int, int],
    *,
    flags: int = 0,
    min_inliers: int = 20,
    holdout_fraction: float = 0.2,
    max_view_error: float = 1.5,
) -> IntrinsicResult:
    """Fit intrinsics, reject high-error views, refit, and score held-out views."""
    if len(object_points) != len(image_points):
        raise ValueError("Object/image observation counts do not match")
    if len(object_points) < min_inliers + 1:
        raise ValueError(
            f"Need at least {min_inliers + 1} observations; got {len(object_points)}"
        )

    heldout = deterministic_holdout_indices(len(object_points), holdout_fraction)
    heldout_set = set(int(index) for index in heldout)
    training = np.array(
        [index for index in range(len(object_points)) if index not in heldout_set],
        dtype=np.int32,
    )
    if len(training) < min_inliers:
        raise ValueError(
            "Held-out split leaves too few intrinsic training observations"
        )

    train_obj = [object_points[index] for index in training]
    train_img = [image_points[index] for index in training]
    _, _, _, _, initial_errors = _fit_intrinsics(
        train_obj, train_img, image_size, flags
    )
    threshold = min(
        robust_error_threshold(initial_errors, ceiling=max_view_error),
        max_view_error,
    )
    local_inliers = np.flatnonzero(initial_errors <= threshold)
    if len(local_inliers) < min_inliers:
        order = np.argsort(initial_errors)
        local_inliers = order[:min_inliers]
    inliers = training[local_inliers]
    outliers = np.array(
        [index for index in training if index not in set(int(i) for i in inliers)],
        dtype=np.int32,
    )

    final_obj = [object_points[index] for index in inliers]
    final_img = [image_points[index] for index in inliers]
    rms, camera_matrix, dist_coeffs, std_intrinsics, per_view = _fit_intrinsics(
        final_obj, final_img, image_size, flags
    )
    heldout_errors = np.asarray(
        [
            _view_error(
                object_points[index],
                image_points[index],
                camera_matrix,
                dist_coeffs,
            )
            for index in heldout
        ],
        dtype=np.float64,
    )
    return IntrinsicResult(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rms=rms,
        per_view_errors=per_view,
        heldout_errors=heldout_errors,
        inlier_indices=np.sort(inliers),
        outlier_indices=np.sort(outliers),
        heldout_indices=heldout,
        std_intrinsics=std_intrinsics,
        model_flags=flags,
        candidate_flags=np.asarray([flags], dtype=np.int32),
        candidate_heldout_rms=np.asarray(
            [float(np.sqrt(np.mean(heldout_errors**2)))], dtype=np.float64
        ),
    )


def calibrate_intrinsic_candidates(
    object_points: Sequence[np.ndarray],
    image_points: Sequence[np.ndarray],
    image_size: tuple[int, int],
    *,
    base_flags: int = 0,
    min_inliers: int = 20,
    holdout_fraction: float = 0.2,
    max_view_error: float = 1.5,
) -> IntrinsicResult:
    """Compare standard and rational distortion using identical held-out views."""
    flags_to_try = [base_flags]
    rational_flags = base_flags | cv2.CALIB_RATIONAL_MODEL
    if rational_flags != base_flags:
        flags_to_try.append(rational_flags)
    results = [
        calibrate_intrinsics_robust(
            object_points,
            image_points,
            image_size,
            flags=flags,
            min_inliers=min_inliers,
            holdout_fraction=holdout_fraction,
            max_view_error=max_view_error,
        )
        for flags in flags_to_try
    ]
    scores = np.asarray(
        [np.sqrt(np.mean(result.heldout_errors**2)) for result in results],
        dtype=np.float64,
    )
    best_index = int(np.argmin(scores))
    # Keep the simpler model when it is within 2% of the best validation score.
    if scores[0] <= scores[best_index] * 1.02:
        best_index = 0
    return replace(
        results[best_index],
        candidate_flags=np.asarray(flags_to_try, dtype=np.int32),
        candidate_heldout_rms=scores,
    )


def stereo_view_error(
    object_points: np.ndarray,
    image_points_a: np.ndarray,
    image_points_b: np.ndarray,
    camera_matrix_a: np.ndarray,
    dist_coeffs_a: np.ndarray,
    camera_matrix_b: np.ndarray,
    dist_coeffs_b: np.ndarray,
    rotation_ab: np.ndarray,
    translation_ab: np.ndarray,
) -> float:
    """Score a stereo view by projecting one board pose into both cameras."""
    success, rvec_a, tvec_a = cv2.solvePnP(
        object_points,
        image_points_a,
        camera_matrix_a,
        dist_coeffs_a,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return float("inf")
    rotation_board_a, _ = cv2.Rodrigues(rvec_a)
    rotation_board_b = rotation_ab @ rotation_board_a
    translation_board_b = rotation_ab @ tvec_a + translation_ab
    rvec_b, _ = cv2.Rodrigues(rotation_board_b)
    projected_a, _ = cv2.projectPoints(
        object_points, rvec_a, tvec_a, camera_matrix_a, dist_coeffs_a
    )
    projected_b, _ = cv2.projectPoints(
        object_points, rvec_b, translation_board_b, camera_matrix_b, dist_coeffs_b
    )
    return float(
        np.sqrt(
            0.5
            * (
                point_rmse(image_points_a, projected_a) ** 2
                + point_rmse(image_points_b, projected_b) ** 2
            )
        )
    )


def calibrate_stereo_robust(
    camera_a: int,
    camera_b: int,
    object_points: Sequence[np.ndarray],
    image_points_a: Sequence[np.ndarray],
    image_points_b: Sequence[np.ndarray],
    camera_matrix_a: np.ndarray,
    dist_coeffs_a: np.ndarray,
    camera_matrix_b: np.ndarray,
    dist_coeffs_b: np.ndarray,
    image_size: tuple[int, int],
    *,
    min_inliers: int = 12,
    holdout_fraction: float = 0.2,
    max_view_error: float = 1.5,
) -> StereoResult:
    """Calibrate a fixed-intrinsic stereo pair with robust per-view rejection."""
    count = len(object_points)
    if count != len(image_points_a) or count != len(image_points_b):
        raise ValueError("Stereo observation counts do not match")
    if count < min_inliers + 1:
        raise ValueError(
            f"Need at least {min_inliers + 1} stereo observations; got {count}"
        )

    def fit(indices: Sequence[int]):
        rms, _, _, _, _, rotation, translation, _, _ = cv2.stereoCalibrate(
            [object_points[i] for i in indices],
            [image_points_a[i] for i in indices],
            [image_points_b[i] for i in indices],
            camera_matrix_a,
            dist_coeffs_a,
            camera_matrix_b,
            dist_coeffs_b,
            image_size,
            flags=cv2.CALIB_FIX_INTRINSIC,
        )
        return float(rms), rotation, translation

    all_indices = np.arange(count, dtype=np.int32)
    heldout = deterministic_holdout_indices(count, holdout_fraction)
    heldout_set = set(int(value) for value in heldout)
    training = np.asarray(
        [value for value in all_indices if int(value) not in heldout_set],
        dtype=np.int32,
    )
    if len(training) < min_inliers:
        raise ValueError("Held-out split leaves too few stereo training observations")
    _, rotation, translation = fit(training)
    initial_errors = np.asarray(
        [
            stereo_view_error(
                object_points[i],
                image_points_a[i],
                image_points_b[i],
                camera_matrix_a,
                dist_coeffs_a,
                camera_matrix_b,
                dist_coeffs_b,
                rotation,
                translation,
            )
            for i in training
        ],
    )
    threshold = min(
        robust_error_threshold(initial_errors, ceiling=max_view_error),
        max_view_error,
    )
    local_inliers = np.flatnonzero(initial_errors <= threshold)
    if len(local_inliers) < min_inliers:
        local_inliers = np.argsort(initial_errors)[:min_inliers]
    inliers = training[local_inliers]
    outliers = np.array(
        [i for i in training if i not in set(int(v) for v in inliers)],
        dtype=np.int32,
    )
    rms, rotation, translation = fit(inliers)
    final_errors = np.asarray(
        [
            stereo_view_error(
                object_points[i],
                image_points_a[i],
                image_points_b[i],
                camera_matrix_a,
                dist_coeffs_a,
                camera_matrix_b,
                dist_coeffs_b,
                rotation,
                translation,
            )
            for i in inliers
        ],
        dtype=np.float64,
    )
    heldout_errors = np.asarray(
        [
            stereo_view_error(
                object_points[i],
                image_points_a[i],
                image_points_b[i],
                camera_matrix_a,
                dist_coeffs_a,
                camera_matrix_b,
                dist_coeffs_b,
                rotation,
                translation,
            )
            for i in heldout
        ],
        dtype=np.float64,
    )
    return StereoResult(
        camera_a=camera_a,
        camera_b=camera_b,
        rotation=rotation,
        translation=translation,
        rms=rms,
        per_view_errors=final_errors,
        heldout_errors=heldout_errors,
        inlier_indices=np.sort(inliers),
        outlier_indices=np.sort(outliers),
        heldout_indices=heldout,
    )


def _relative_transform(
    rotation_a: np.ndarray,
    translation_a: np.ndarray,
    rotation_b: np.ndarray,
    translation_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation_ab = rotation_b @ rotation_a.T
    translation_ab = translation_b - rotation_ab @ translation_a
    return rotation_ab, translation_ab


def solve_camera_pose_graph(
    num_cameras: int,
    pair_results: Sequence[StereoResult],
) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    """Solve globally consistent world-to-camera poses from pair calibrations."""
    adjacency: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {
        index: [] for index in range(num_cameras)
    }
    for pair in pair_results:
        adjacency[pair.camera_a].append(
            (pair.camera_b, pair.rotation, pair.translation)
        )
        inverse_rotation = pair.rotation.T
        inverse_translation = -inverse_rotation @ pair.translation
        adjacency[pair.camera_b].append(
            (pair.camera_a, inverse_rotation, inverse_translation)
        )

    rotations: list[np.ndarray | None] = [None] * num_cameras
    translations: list[np.ndarray | None] = [None] * num_cameras
    rotations[0] = np.eye(3)
    translations[0] = np.zeros((3, 1))
    queue = [0]
    while queue:
        camera = queue.pop(0)
        for neighbor, rotation_cn, translation_cn in adjacency[camera]:
            if rotations[neighbor] is not None:
                continue
            rotations[neighbor] = rotation_cn @ rotations[camera]
            translations[neighbor] = rotation_cn @ translations[camera] + translation_cn
            queue.append(neighbor)
    if any(rotation is None for rotation in rotations):
        raise ValueError("Stereo pair graph is not connected to camera 0")

    resolved_rotations = [np.asarray(value) for value in rotations]
    resolved_translations = [np.asarray(value) for value in translations]
    baseline_scale = float(
        np.median([np.linalg.norm(pair.translation) for pair in pair_results])
    )
    baseline_scale = max(baseline_scale, 1e-6)

    initial = []
    for camera in range(1, num_cameras):
        rvec, _ = cv2.Rodrigues(resolved_rotations[camera])
        initial.extend(rvec.reshape(3))
        initial.extend(resolved_translations[camera].reshape(3))

    def unpack(parameters):
        rots = [np.eye(3)]
        trans = [np.zeros((3, 1))]
        offset = 0
        for _camera in range(1, num_cameras):
            rotation, _ = cv2.Rodrigues(parameters[offset : offset + 3])
            translation = parameters[offset + 3 : offset + 6].reshape(3, 1)
            rots.append(rotation)
            trans.append(translation)
            offset += 6
        return rots, trans

    def residuals(parameters):
        rots, trans = unpack(parameters)
        residual = []
        for pair in pair_results:
            predicted_rotation, predicted_translation = _relative_transform(
                rots[pair.camera_a],
                trans[pair.camera_a],
                rots[pair.camera_b],
                trans[pair.camera_b],
            )
            rotation_delta, _ = cv2.Rodrigues(pair.rotation.T @ predicted_rotation)
            weight = np.sqrt(max(len(pair.inlier_indices), 1)) / max(pair.rms, 0.1)
            residual.extend(rotation_delta.reshape(3) * weight)
            residual.extend(
                ((predicted_translation - pair.translation) / baseline_scale).reshape(3)
                * weight
            )
        return np.asarray(residual)

    solution = least_squares(
        residuals,
        np.asarray(initial),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=500,
    )
    final_rotations, final_translations = unpack(solution.x)
    loop_rms = float(np.sqrt(np.mean(residuals(solution.x) ** 2)))
    return final_rotations, final_translations, loop_rms


def _triangulate_dlt(
    image_points: Sequence[np.ndarray],
    projection_matrices: Sequence[np.ndarray],
) -> np.ndarray | None:
    rows = []
    for point, projection in zip(image_points, projection_matrices):
        x, y = np.asarray(point).reshape(2)
        rows.append(x * projection[2] - projection[0])
        rows.append(y * projection[2] - projection[1])
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    homogeneous = vh[-1]
    if abs(homogeneous[3]) < 1e-12:
        return None
    return homogeneous[:3] / homogeneous[3]


def validate_heldout_board_reconstruction(
    all_corners: Sequence[Sequence[np.ndarray]],
    all_ids: Sequence[Sequence[np.ndarray]],
    board_points: np.ndarray,
    camera_matrices: Sequence[np.ndarray],
    dist_coeffs: Sequence[np.ndarray],
    rotations: Sequence[np.ndarray],
    translations: Sequence[np.ndarray],
    heldout_indices: Sequence[int],
) -> BoardValidationResult:
    """Triangulate unseen board views and measure rigid shape/scale/planarity."""
    projection_matrices = [
        camera_matrix @ np.hstack([rotation, translation.reshape(3, 1)])
        for camera_matrix, rotation, translation in zip(
            camera_matrices, rotations, translations
        )
    ]
    shape_errors = []
    scale_errors = []
    planarity_errors = []
    total_points = 0
    valid_views = 0

    for view in heldout_indices:
        observations: dict[int, list[tuple[int, np.ndarray]]] = {}
        for camera in range(len(camera_matrices)):
            if view >= len(all_corners[camera]):
                continue
            ids = all_ids[camera][view].reshape(-1)
            corners = all_corners[camera][view].reshape(-1, 1, 2)
            undistorted = cv2.undistortPoints(
                corners,
                camera_matrices[camera],
                dist_coeffs[camera],
                P=camera_matrices[camera],
            ).reshape(-1, 2)
            for corner_id, point in zip(ids, undistorted):
                observations.setdefault(int(corner_id), []).append((camera, point))

        reconstructed = []
        expected = []
        for corner_id, views in observations.items():
            if len(views) < 2:
                continue
            point = _triangulate_dlt(
                [item[1] for item in views],
                [projection_matrices[item[0]] for item in views],
            )
            if point is not None:
                reconstructed.append(point)
                expected.append(board_points[corner_id])
        if len(reconstructed) < 6:
            continue

        reconstructed_array = np.asarray(reconstructed, dtype=np.float64)
        expected_array = np.asarray(expected, dtype=np.float64)
        reconstructed_centered = reconstructed_array - np.mean(
            reconstructed_array, axis=0
        )
        expected_centered = expected_array - np.mean(expected_array, axis=0)
        covariance = expected_centered.T @ reconstructed_centered
        u, _singular_values, vh = np.linalg.svd(covariance)
        alignment = vh.T @ u.T
        if np.linalg.det(alignment) < 0:
            vh[-1] *= -1
            alignment = vh.T @ u.T
        aligned_expected = expected_centered @ alignment.T
        shape_errors.extend(
            np.linalg.norm(aligned_expected - reconstructed_centered, axis=1)
        )
        _u, singular_values, _vh = np.linalg.svd(reconstructed_centered)
        plane_normal = _vh[-1]
        planarity_errors.extend(np.abs(reconstructed_centered @ plane_normal))

        for first in range(len(reconstructed_array)):
            for second in range(first + 1, len(reconstructed_array)):
                expected_distance = np.linalg.norm(
                    expected_array[first] - expected_array[second]
                )
                if expected_distance <= 0:
                    continue
                reconstructed_distance = np.linalg.norm(
                    reconstructed_array[first] - reconstructed_array[second]
                )
                scale_errors.append(
                    abs(reconstructed_distance / expected_distance - 1.0) * 100.0
                )
        valid_views += 1
        total_points += len(reconstructed_array)

    if not valid_views:
        raise ValueError("No held-out board view had six triangulatable corners")
    return BoardValidationResult(
        shape_rms_mm=float(np.sqrt(np.mean(np.square(shape_errors))) * 1000.0),
        scale_error_percent=float(np.median(scale_errors)),
        planarity_rms_mm=float(np.sqrt(np.mean(np.square(planarity_errors))) * 1000.0),
        num_views=valid_views,
        num_points=total_points,
    )
