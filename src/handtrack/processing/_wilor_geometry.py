"""Calibrated WiLoR joint triangulation and similarity placement of a MANO mesh."""
from dataclasses import dataclass
import itertools

import cv2
import numpy as np

from ._calibration_solver import _triangulate_dlt


@dataclass(frozen=True)
class CalibratedWiLoRHand:
    handedness: str
    joints: np.ndarray  # mesh's skeleton, placed in camera-0 world meters
    vertices: np.ndarray  # same fitted similarity transform as joints
    triangulated_joints: np.ndarray  # independent measurements; NaN where rejected
    cameras: tuple
    source_camera: int
    joint_reprojection_rms_px: float
    mesh_fit_rms_mm: float
    mesh_scale: float


def _owned(array):
    result = np.array(array, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


def _similarity(source, target):
    centered = source - source.mean(axis=0)
    destination = target - target.mean(axis=0)
    u, singular, vt = np.linalg.svd(centered.T @ destination)
    if singular[1] < 1e-12:
        raise ValueError('Hand points are collinear')
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    scale = np.sum(singular * np.diag(correction)) / np.sum(centered ** 2)
    if not np.isfinite(scale) or not .25 <= scale <= 4.:
        raise ValueError('Mesh scale is inconsistent with triangulation')
    translation = target.mean(axis=0) - scale * source.mean(axis=0) @ rotation.T
    return scale, rotation, translation


def reconstruct_wilor(results, received_at, calibration, *, enabled=None,
                       max_reprojection_px=8., max_receipt_skew=.05):
    """Associate unique same-handed views; never use monocular camera translation.

    Per-view track IDs are unrelated. Multiple same-handed detections make that
    handedness ambiguous and suppress it. Accepted joints need pair consensus,
    positive depth, nonparallel rays and pixel reprojection agreement.
    """
    matrices, distortions, rotations, translations = calibration[:4]
    valid_views = [c for c, result in enumerate(results)
                   if result is not None and received_at[c] is not None
                   and (enabled is None or c in enabled)]
    if len(valid_views) < 2:
        return ()
    latest = max(received_at[c] for c in valid_views)
    valid_views = [c for c in valid_views if latest-received_at[c] <= max_receipt_skew]
    projections = [k @ np.hstack((r, t.reshape(3, 1))) for k, r, t in
                   zip(matrices, rotations, translations)]
    reconstructed = []
    for label in ('Left', 'Right'):
        hands = {}
        ambiguous = False
        for camera in valid_views:
            candidates = [h for h in results[camera].hands if h.handedness == label]
            if len(candidates) > 1:
                ambiguous = True
                break
            if candidates:
                hand = candidates[0]
                if (hand.image_landmarks.shape == (21, 2)
                        and np.isfinite(hand.image_landmarks).all()):
                    hands[camera] = hand
        if ambiguous or len(hands) < 2:
            continue
        pixels = {c: cv2.undistortPoints(
            h.image_landmarks.astype(np.float64).reshape(-1, 1, 2), matrices[c], distortions[c],
            P=matrices[c]).reshape(21, 2) for c, h in hands.items()}
        measured = np.full((21, 3), np.nan)
        errors = []
        contributing = set()

        def reprojection(point, camera, joint):
            if (rotations[camera] @ point + translations[camera].reshape(3))[2] <= 0:
                return float('inf')
            rvec, _ = cv2.Rodrigues(rotations[camera])
            projected, _ = cv2.projectPoints(point.reshape(1, 3), rvec, translations[camera],
                                              matrices[camera], distortions[camera])
            return float(np.linalg.norm(projected.reshape(2) - hands[camera].image_landmarks[joint]))

        for joint in range(21):
            best = None
            for a, b in itertools.combinations(hands, 2):
                rays = [rotations[c].T @ np.linalg.solve(matrices[c], np.r_[pixels[c][joint], 1.])
                        for c in (a, b)]
                cosine = abs(np.dot(*rays) / (np.linalg.norm(rays[0])*np.linalg.norm(rays[1])))
                if cosine > np.cos(np.deg2rad(2.)):
                    continue
                point = _triangulate_dlt([pixels[c][joint] for c in (a, b)],
                                         [projections[c] for c in (a, b)])
                if point is None or not np.isfinite(point).all():
                    continue
                residuals = {c: reprojection(point, c, joint) for c in hands}
                inliers = [c for c, error in residuals.items() if error <= max_reprojection_px]
                if len(inliers) < 2:
                    continue
                rank = (len(inliers), -np.mean([residuals[c] for c in inliers]))
                if best is None or rank > best[0]:
                    best = rank, inliers
            if best is None:
                continue
            inliers = best[1]
            point = _triangulate_dlt([pixels[c][joint] for c in inliers], [projections[c] for c in inliers])
            if point is None or not np.isfinite(point).all():
                continue
            residuals = [reprojection(point, c, joint) for c in inliers]
            if max(residuals) > max_reprojection_px:
                continue
            measured[joint] = point
            errors.extend(residuals)
            contributing.update(inliers)
        accepted = np.isfinite(measured).all(axis=1)
        if accepted.sum() < 12 or not accepted[[0, 5, 9, 17]].all():
            continue
        fits = []
        for camera, hand in hands.items():
            if (camera not in contributing or hand.joints.shape != (21, 3)
                    or hand.vertices.shape != (778, 3)
                    or not np.isfinite(hand.joints).all() or not np.isfinite(hand.vertices).all()):
                continue
            try:
                scale, rotation, translation = _similarity(hand.joints[accepted], measured[accepted])
            except (ValueError, np.linalg.LinAlgError):
                continue
            # Place the mesh wrist at the triangulated wrist, not WiLoR's
            # focal-length-dependent monocular camera translation.
            translation = measured[0] - scale * hand.joints[0] @ rotation.T
            joints = scale * hand.joints @ rotation.T + translation
            fit_rms = float(np.sqrt(np.mean(np.sum((joints[accepted]-measured[accepted])**2, axis=1))))
            if fit_rms > .03:
                continue
            vertices = scale * hand.vertices @ rotation.T + translation
            fits.append((fit_rms, camera, scale, joints, vertices))
        if not fits:
            continue
        fit, camera, scale, joints, vertices = min(fits, key=lambda item: item[0])
        reconstructed.append(CalibratedWiLoRHand(
            label, _owned(joints), _owned(vertices), _owned(measured), tuple(sorted(contributing)),
            camera, float(np.sqrt(np.mean(np.square(errors)))), fit*1000., float(scale)))
    return tuple(reconstructed)
