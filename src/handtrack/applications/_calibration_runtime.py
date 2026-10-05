"""Shared robust ChArUco calibration with atomic publication of validated results."""

import importlib
import itertools
from datetime import datetime
import os
from pathlib import Path
import numpy as np

from handtrack.processing._calibration_solver import (
    calibrate_intrinsic_candidates, calibrate_stereo_robust, solve_camera_pose_graph,
    validate_heldout_board_reconstruction)

_CONFIG_NAMES = ("ARUCO_DICT", "CALIBRATION_DIR", "CALIBRATION_FILE",
                 "CHARUCO_MARKER_LENGTH", "CHARUCO_SQUARE_LENGTH",
                 "CHARUCO_SQUARES_X", "CHARUCO_SQUARES_Y", "NUM_CALIBRATION_IMAGES")
MAX_HELDOUT_REPROJECTION_ERROR = 1.5
_intrinsic_results = []
_stereo_results = []
_camera_ids = None
_image_sizes = None
_diagnostics = None
_reserved_indices = []
_board_validation = None
MAX_BOARD_SHAPE_RMS_MM = 2.0
MAX_BOARD_SCALE_ERROR_PERCENT = 2.0
MAX_BOARD_PLANARITY_RMS_MM = 1.0


def _load_backend(backend):
    if backend not in ("webcam", "optitrack"):
        raise ValueError(f"Unsupported calibration backend: {backend}")
    config = importlib.import_module(f"handtrack.cameras.{backend}.config")
    globals().update({name: getattr(config, name) for name in _CONFIG_NAMES})
    module = "multi_webcam" if backend == "webcam" else "multi_mjpeg"
    return importlib.import_module(f"handtrack.cameras.{backend}.{module}").CameraManager


def calibrate_intrinsics(all_corners, all_ids, board, img_size, num_cameras):
    global _intrinsic_results, _reserved_indices
    _reserved_indices = list(range(4, len(all_corners[0]), 5))
    board_points = board.getChessboardCorners()
    _intrinsic_results = []
    for camera in range(num_cameras):
        objects, images = [], []
        for view, (corners, ids) in enumerate(zip(all_corners[camera], all_ids[camera])):
            if view in _reserved_indices:
                continue
            if corners is not None and ids is not None and len(ids) >= 6:
                objects.append(board_points[ids.flatten()].astype(np.float32))
                images.append(corners.reshape(-1, 1, 2).astype(np.float32))
        size = tuple(_image_sizes[camera]) if _image_sizes is not None else img_size
        result = calibrate_intrinsic_candidates(objects, images, size, min_inliers=12)
        _intrinsic_results.append(result)
        print(f"Camera {camera}: training RMS {result.rms:.3f} px; held-out RMS {np.sqrt(np.mean(result.heldout_errors**2)):.3f} px")
    return [result.camera_matrix for result in _intrinsic_results], [result.dist_coeffs for result in _intrinsic_results]


def calibrate_stereo_pairs(all_corners, all_ids, board, matrices, dists, img_size, num_cameras):
    global _stereo_results, _board_validation
    _board_validation = None
    _stereo_results = []
    board_points = board.getChessboardCorners()
    for a, b in itertools.combinations(range(num_cameras), 2):
        objects, images_a, images_b = [], [], []
        for view, (ca, ia, cb, ib) in enumerate(zip(all_corners[a], all_ids[a], all_corners[b], all_ids[b])):
            if view in _reserved_indices:
                continue
            if ca is None or cb is None or ia is None or ib is None:
                continue
            common, aa, bb = np.intersect1d(ia.flatten(), ib.flatten(), return_indices=True)
            if len(common) >= 6:
                objects.append(board_points[common].astype(np.float32))
                images_a.append(ca[aa].reshape(-1, 1, 2).astype(np.float32))
                images_b.append(cb[bb].reshape(-1, 1, 2).astype(np.float32))
        if len(objects) < 11:
            continue
        _stereo_results.append(calibrate_stereo_robust(a, b, objects, images_a, images_b,
                                                       matrices[a], dists[a], matrices[b], dists[b],
                                                       img_size, min_inliers=8))
    if num_cameras == 1:
        return [np.eye(3)], [np.zeros((3, 1))]
    rotations, translations, _ = solve_camera_pose_graph(num_cameras, _stereo_results)
    _board_validation = validate_heldout_board_reconstruction(
        all_corners, all_ids, board_points, matrices, dists, rotations, translations,
        _reserved_indices)
    return rotations, translations


def save_calibration(num_cameras, img_size, num_captured, camera_matrices, dist_coeffs_list,
                     R_matrices, T_vectors, *, intrinsic_results=None):
    results = intrinsic_results if intrinsic_results is not None else _intrinsic_results
    data = {"schema_version": 2, "num_cameras": num_cameras, "img_size": img_size,
            "image_sizes": _image_sizes if _image_sizes is not None else np.tile(img_size, (num_cameras, 1)),
            "camera_ids": _camera_ids if _camera_ids is not None else list(range(num_cameras)),
            "num_captures": num_captured, "coordinate_units": "meters",
            "calibration_type": "charuco", "charuco_square_length": CHARUCO_SQUARE_LENGTH,
            "charuco_marker_length": CHARUCO_MARKER_LENGTH}
    if _diagnostics and all(item.get("uid") for item in _diagnostics):
        data["camera_uid_high"] = [item["uid"][0] for item in _diagnostics]
        data["camera_uid_low"] = [item["uid"][1] for item in _diagnostics]
        data["camera_serials"] = np.asarray([str(item.get("serial", "")) for item in _diagnostics])
    quality = bool(results) and len(results) == num_cameras
    for index in range(num_cameras):
        for prefix, values in (("camera_matrix", camera_matrices), ("dist_coeffs", dist_coeffs_list),
                               ("R", R_matrices), ("T", T_vectors)):
            value = np.asarray(values[index])
            if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError(f"Camera {index} has an invalid {prefix} value")
            data[f"{prefix}_{index}"] = value
        if index < len(results):
            result = results[index]
            heldout = np.asarray(result.heldout_errors)
            quality &= bool(heldout.size and np.isfinite(heldout).all()
                            and np.max(heldout) <= MAX_HELDOUT_REPROJECTION_ERROR)
            data[f"heldout_errors_{index}"] = heldout
            data[f"per_view_errors_{index}"] = result.per_view_errors
            data[f"inlier_indices_{index}"] = result.inlier_indices
            data[f"outlier_indices_{index}"] = result.outlier_indices
            data[f"heldout_indices_{index}"] = result.heldout_indices
            data[f"std_intrinsics_{index}"] = result.std_intrinsics
            data[f"model_flags_{index}"] = result.model_flags
            data[f"candidate_flags_{index}"] = result.candidate_flags
            data[f"candidate_heldout_rms_{index}"] = result.candidate_heldout_rms
    for pair in _stereo_results:
        heldout = np.asarray(pair.heldout_errors)
        quality &= bool(heldout.size and np.isfinite(heldout).all()
                        and np.max(heldout) <= MAX_HELDOUT_REPROJECTION_ERROR)
        data[f"stereo_heldout_{pair.camera_a}_{pair.camera_b}"] = heldout
    if num_cameras > 1:
        # A connected stereo graph is required before publishing extrinsics.
        if not _stereo_results:
            quality = False
        else:
            try:
                solve_camera_pose_graph(num_cameras, _stereo_results)
            except ValueError:
                quality = False
    if num_cameras > 1:
        validation = _board_validation
        quality &= validation is not None
        if validation is not None:
            metrics = (validation.shape_rms_mm, abs(validation.scale_error_percent), validation.planarity_rms_mm)
            quality &= bool(validation.num_views >= 3 and np.isfinite(metrics).all()
                            and all(value <= limit for value, limit in zip(metrics,
                                (MAX_BOARD_SHAPE_RMS_MM, MAX_BOARD_SCALE_ERROR_PERCENT, MAX_BOARD_PLANARITY_RMS_MM))))
            data.update(board_shape_rms_mm=validation.shape_rms_mm,
                        board_scale_error_percent=validation.scale_error_percent,
                        board_planarity_rms_mm=validation.planarity_rms_mm,
                        board_validation_views=validation.num_views,
                        board_validation_points=validation.num_points,
                        reserved_view_indices=np.asarray(_reserved_indices, dtype=int))
    data["quality_passed"] = quality
    destination = Path(CALIBRATION_FILE)
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidate = destination.with_name(destination.name + ".candidate.npz")
    np.savez_compressed(candidate, **data)
    if not quality:
        raise ValueError(f"Calibration failed held-out quality checks; previous calibration preserved. Candidate: {candidate}")
    from mavis_track.calibration import load_camera_calibration
    load_camera_calibration(candidate, count=num_cameras, image_sizes=data["image_sizes"], camera_ids=data["camera_ids"])
    backup = destination.parent / ("calib_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".npz")
    if destination.exists():
        backup.write_bytes(destination.read_bytes())
    else:
        backup.write_bytes(candidate.read_bytes())
    os.replace(candidate, destination)
    print(f"Validated calibration saved: {destination}")


def main(backend):
    global _camera_ids, _image_sizes, _diagnostics, _intrinsic_results, _stereo_results, _reserved_indices, _board_validation
    manager_class = _load_backend(backend)
    original = "webcam" if backend == "webcam" else "optitrack_cam_py"
    module = importlib.import_module(f"unity_hand_tracking.{original}.calibration")
    _camera_ids = _image_sizes = _diagnostics = None
    _intrinsic_results, _stereo_results = [], []
    _reserved_indices, _board_validation = [], None
    managers = []
    def create_manager(*args, **kwargs):
        manager = manager_class(*args, **kwargs)
        managers.append(manager)
        return manager
    def solve_intrinsics(*args):
        global _camera_ids, _image_sizes, _diagnostics
        manager = managers[-1]
        _camera_ids = getattr(manager, "camera_indices", list(range(manager.num_cameras)))
        _image_sizes = np.asarray([manager.get_resolution(i) for i in range(manager.num_cameras)])
        _diagnostics = manager.get_camera_diagnostics() if hasattr(manager, "get_camera_diagnostics") else None
        # Mixed dimensions require a separately designed stereo observation
        # transform; fail explicitly rather than reuse camera 0's pixel scale.
        if not np.all(_image_sizes == _image_sizes[0]):
            raise ValueError("Stereo calibration currently requires matching camera resolutions")
        return calibrate_intrinsics(*args)
    saved = False
    def publish(*args, **kwargs):
        nonlocal saved
        save_calibration(*args, **kwargs)
        saved = True
    replacements = {"CameraManager": create_manager, "calibrate_intrinsics": solve_intrinsics,
                    "calibrate_stereo_pairs": calibrate_stereo_pairs, "save_calibration": publish}
    previous = {name: getattr(module, name) for name in replacements}
    for name, value in replacements.items():
        setattr(module, name, value)
    try:
        module._capture_main()
        return 0 if saved else 1
    finally:
        for name, value in previous.items():
            setattr(module, name, value)
        for manager in managers:
            manager.stop_all()
