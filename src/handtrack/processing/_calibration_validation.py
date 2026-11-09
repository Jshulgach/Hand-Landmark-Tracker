"""Independent endpoint triangulation and descriptive physical validation metrics."""
import itertools

import cv2
import numpy as np

from ._calibration_solver import _triangulate_dlt


def reconstruct_endpoints(detections, calibration, corner_ids, *, max_reprojection_px=1.5,
                          min_ray_angle_deg=2.):
    """Reconstruct two corners without a board-pose or known-length constraint."""
    matrices, distortions, rotations, translations = calibration[:4]
    cameras = []
    pixels = []
    for camera, (found, data) in enumerate(detections):
        if not found:
            continue
        corners, ids = data
        if corners is None or ids is None or len(corners) != ids.size:
            continue
        lookup = {int(key): point for key, point in zip(ids.ravel(), corners.reshape(-1, 2))}
        if all(key in lookup for key in corner_ids):
            cameras.append(camera)
            pixels.append(np.array([lookup[key] for key in corner_ids], dtype=np.float64))
    if len(cameras) < 2:
        raise ValueError('Both selected corners must be visible in at least two cameras')
    projections = [matrices[c] @ np.hstack((rotations[c], translations[c].reshape(3, 1)))
                   for c in cameras]
    undistorted = [cv2.undistortPoints(p.reshape(-1, 1, 2), matrices[c], distortions[c],
                                     P=matrices[c]).reshape(2, 2)
                   for c, p in zip(cameras, pixels)]
    points = []
    angles = []
    for endpoint in range(2):
        rays = [rotations[c].T @ np.linalg.solve(matrices[c], np.r_[p[endpoint], 1.])
                for c, p in zip(cameras, undistorted)]
        rays = [ray / np.linalg.norm(ray) for ray in rays]
        angle = max(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), 0., 1.)))
                    for a, b in itertools.combinations(rays, 2))
        if angle < min_ray_angle_deg:
            raise ValueError('Camera rays are too nearly parallel for validation')
        point = _triangulate_dlt([p[endpoint] for p in undistorted], projections)
        if point is None or not np.isfinite(point).all():
            raise ValueError('Invalid triangulation')
        if any((rotations[c] @ point + translations[c].reshape(3))[2] <= 0 for c in cameras):
            raise ValueError('Triangulated point is behind a camera')
        points.append(point)
        angles.append(float(angle))
    points = np.array(points)
    squared_errors = []
    for c, observed in zip(cameras, pixels):
        rvec, _ = cv2.Rodrigues(rotations[c])
        projected, _ = cv2.projectPoints(points, rvec, translations[c], matrices[c], distortions[c])
        squared_errors.append(np.sum((projected.reshape(2, 2) - observed) ** 2, axis=1))
    endpoint_rms = np.sqrt(np.mean(squared_errors, axis=0))
    if not np.isfinite(endpoint_rms).all() or np.max(endpoint_rms) > max_reprojection_px:
        raise ValueError('Endpoint reprojection error exceeds the sample filter')
    return points, tuple(cameras), float(np.max(endpoint_rms)), min(angles)


def summarize_trial(points_m, expected_mm):
    points = np.asarray(points_m, dtype=float) * 1000.
    if points.ndim != 3 or points.shape[1:] != (2, 3) or len(points) < 2:
        raise ValueError('At least two endpoint samples are required')
    if not np.isfinite(points).all() or not np.isfinite(expected_mm) or expected_mm <= 0:
        raise ValueError('Expected distance and endpoint samples must be finite and valid')
    distances = np.linalg.norm(points[:, 1] - points[:, 0], axis=1)
    errors = distances - expected_mm
    means = points.mean(axis=0)
    jitter = np.sqrt(np.mean(np.sum((points - means) ** 2, axis=2), axis=0))
    return dict(samples=len(points), mean_distance_mm=float(distances.mean()),
                mean_signed_error_mm=float(errors.mean()),
                mean_absolute_error_mm=float(np.abs(errors).mean()),
                distance_rmse_mm=float(np.sqrt(np.mean(errors ** 2))),
                max_absolute_error_mm=float(np.max(np.abs(errors))),
                distance_std_mm=float(np.std(distances, ddof=1)),
                endpoint_jitter_rms_mm=jitter.tolist(), endpoint_means_mm=means.tolist())


def summarize_locations(trials):
    report = {}
    for name in dict.fromkeys(trial['location'] for trial in trials):
        selected = [trial for trial in trials if trial['location'] == name]
        means = np.array([trial['endpoint_means_mm'] for trial in selected])
        # Return-to-position metrics require a physical fixture establishing
        # the same endpoint positions and orientation on each trial.
        midpoint_shifts = np.linalg.norm(means.mean(axis=1) - means[0].mean(axis=0), axis=1)
        endpoint_shifts = np.sqrt(np.mean(np.sum((means - means[0]) ** 2, axis=2), axis=1))
        distances = [trial['mean_distance_mm'] for trial in selected]
        report[name] = dict(trials=len(selected),
            mean_trial_distance_mm=float(np.mean(distances)),
            trial_distance_range_mm=float(np.ptp(distances)),
            max_return_midpoint_shift_mm=float(max(midpoint_shifts)) if len(selected)>1 else None,
            max_return_endpoint_rms_shift_mm=float(max(endpoint_shifts)) if len(selected)>1 else None)
    return report
