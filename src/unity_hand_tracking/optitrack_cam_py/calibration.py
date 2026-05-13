"""
calibration.py — Multi-camera ChArUco board calibration using multi_mjpeg.

Uses ChArUco boards instead of standard checkerboards. ChArUco boards combine
ArUco markers with a chessboard pattern, allowing calibration even when the
board is partially occluded or partially out of frame.

Uses CameraManager from multi_mjpeg.py as the camera backend.

To generate a printable ChArUco board:
    python generate_charuco_board.py
"""

import concurrent.futures
import os
import time
from datetime import datetime

import cv2
import numpy as np

try:
    from .config import (
        ARUCO_DICT,
        CALIBRATION_DIR,
        CALIBRATION_FILE,
        CHARUCO_MARKER_LENGTH,
        CHARUCO_SQUARE_LENGTH,
        CHARUCO_SQUARES_X,
        CHARUCO_SQUARES_Y,
        NUM_CALIBRATION_IMAGES,
    )
    from .multi_mjpeg import CameraManager
except ImportError:
    from config import (
        ARUCO_DICT,
        CALIBRATION_DIR,
        CALIBRATION_FILE,
        CHARUCO_MARKER_LENGTH,
        CHARUCO_SQUARE_LENGTH,
        CHARUCO_SQUARES_X,
        CHARUCO_SQUARES_Y,
        NUM_CALIBRATION_IMAGES,
    )
    from multi_mjpeg import CameraManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Ordered list of ArUco dictionaries to try during auto-detection
_ARUCO_DICTS_TO_TRY = [
    ("DICT_4X4_50", cv2.aruco.DICT_4X4_50),
    ("DICT_4X4_100", cv2.aruco.DICT_4X4_100),
    ("DICT_4X4_250", cv2.aruco.DICT_4X4_250),
    ("DICT_5X5_50", cv2.aruco.DICT_5X5_50),
    ("DICT_5X5_100", cv2.aruco.DICT_5X5_100),
    ("DICT_5X5_250", cv2.aruco.DICT_5X5_250),
    ("DICT_6X6_50", cv2.aruco.DICT_6X6_50),
    ("DICT_6X6_100", cv2.aruco.DICT_6X6_100),
    ("DICT_6X6_250", cv2.aruco.DICT_6X6_250),
    ("DICT_7X7_50", cv2.aruco.DICT_7X7_50),
    ("DICT_7X7_100", cv2.aruco.DICT_7X7_100),
    ("DICT_7X7_250", cv2.aruco.DICT_7X7_250),
]


def build_charuco_detector(dict_name=None):
    """Create the ChArUco board and detector objects.

    Parameters
    ----------
    dict_name : str | None
        If provided, use this specific ArUco dictionary name.
        Otherwise, use the ARUCO_DICT from config.
    """
    if dict_name is not None:
        dict_id = getattr(cv2.aruco, dict_name)
    else:
        dict_id = getattr(cv2.aruco, ARUCO_DICT, cv2.aruco.DICT_5X5_250)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

    board = cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        CHARUCO_SQUARE_LENGTH,
        CHARUCO_MARKER_LENGTH,
        aruco_dict,
    )

    detector_params = cv2.aruco.DetectorParameters()
    charuco_params = cv2.aruco.CharucoParameters()
    detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)

    return board, detector


def auto_detect_dictionary(frame):
    """Try all common ArUco dictionaries to find which one the board uses.

    Returns
    -------
    dict_name : str | None
        Name of the matching dictionary, or None if no match.
    num_corners : int
        Number of ChArUco corners detected with the best dictionary.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    best_name = None
    best_count = 0

    for name, dict_id in _ARUCO_DICTS_TO_TRY:
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        board = cv2.aruco.CharucoBoard(
            (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
            CHARUCO_SQUARE_LENGTH,
            CHARUCO_MARKER_LENGTH,
            aruco_dict,
        )
        detector_params = cv2.aruco.DetectorParameters()
        charuco_params = cv2.aruco.CharucoParameters()
        detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)

        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(
            gray
        )

        n_corners = len(charuco_corners) if charuco_corners is not None else 0
        if n_corners > best_count:
            best_count = n_corners
            best_name = name

    return best_name, best_count


def detect_charuco(args):
    """
    Try to find the ChArUco board in *frame*.

    Returns
    -------
    found : bool
    corners : np.ndarray | None
    ids : np.ndarray | None
    """
    frame, detector = args
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(
        gray
    )

    # We need at least 6 corners for a robust calibration frame
    if charuco_corners is not None and len(charuco_corners) >= 6:
        return True, (charuco_corners, charuco_ids)
    return False, (None, None)


def annotate_frame(frame, cam_index, found, corners_data):
    """Draw camera label, ChArUco overlay, and border colour."""
    display = frame.copy()

    cv2.putText(
        display,
        f"Camera {cam_index}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    if found:
        charuco_corners, charuco_ids = corners_data
        cv2.aruco.drawDetectedCornersCharuco(
            display, charuco_corners, charuco_ids, (0, 255, 0)
        )
        border_colour = (0, 255, 0)
    else:
        border_colour = (0, 0, 255)

    cv2.rectangle(
        display,
        (0, 0),
        (frame.shape[1] - 1, frame.shape[0] - 1),
        border_colour,
        5,
    )
    return display


def build_grid(frames):
    """Stitch list of frames into a 2x2 grid image."""
    if len(frames) != 4:
        # Fallback if not exactly 4 cameras
        grid_cols = 2
        rows = []
        for start in range(0, len(frames), grid_cols):
            row = frames[start : start + grid_cols]
            while len(row) < grid_cols:
                row.append(np.zeros_like(frames[0]))
            rows.append(np.hstack(row))
        return np.vstack(rows)

    # 2x2 grid for exactly 4 cameras
    top_row = np.hstack((frames[0], frames[1]))
    bottom_row = np.hstack((frames[2], frames[3]))
    return np.vstack((top_row, bottom_row))


# ---------------------------------------------------------------------------
# Calibration math
# ---------------------------------------------------------------------------
def calibrate_intrinsics(all_corners, all_ids, board, img_size, num_cameras):
    """Calibrate each camera independently using ChArUco corners.

    The ChArUco API handles the object-point ↔ image-point correspondence
    internally through the board definition and detected corner IDs,
    so we don't need to build object point arrays ourselves.
    """
    camera_matrices = []
    dist_coeffs_list = []

    print("\n" + "=" * 60)
    print("INDIVIDUAL CAMERA CALIBRATION (ChArUco)")
    print("=" * 60)

    for idx in range(num_cameras):
        print(f"\nCalibrating camera {idx}...")

        # Build matched object points from the board definition + detected IDs
        obj_points = []
        img_points = []
        board_corners_3d = board.getChessboardCorners()

        for corners, ids in zip(all_corners[idx], all_ids[idx]):
            if corners is not None and ids is not None and len(corners) >= 6:
                # Map each detected corner ID to its 3D board coordinate
                obj_pts = np.array(
                    [board_corners_3d[i] for i in ids.flatten()], dtype=np.float32
                )
                img_pts = corners.reshape(-1, 1, 2).astype(np.float32)
                obj_points.append(obj_pts.reshape(-1, 1, 3))
                img_points.append(img_pts)

        if len(obj_points) < 5:
            print(
                f"  Not enough valid frames for camera {idx} ({len(obj_points)} frames)"
            )
            camera_matrices.append(None)
            dist_coeffs_list.append(None)
            continue

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points,
            img_points,
            img_size,
            None,
            None,
        )
        if not ret:
            print(f"  Calibration FAILED for camera {idx}!")
            camera_matrices.append(None)
            dist_coeffs_list.append(None)
            continue

        camera_matrices.append(mtx)
        dist_coeffs_list.append(dist)

        # Reprojection error
        total_err = 0.0
        for i in range(len(obj_points)):
            proj, _ = cv2.projectPoints(
                obj_points[i],
                rvecs[i],
                tvecs[i],
                mtx,
                dist,
            )
            err = cv2.norm(img_points[i], proj, cv2.NORM_L2) / len(proj)
            total_err += err
        mean_err = total_err / len(obj_points)
        print(f"  Reprojection error: {mean_err:.4f} px  ({len(obj_points)} frames)")

    return camera_matrices, dist_coeffs_list


def calibrate_stereo_pairs(
    all_corners,
    all_ids,
    board,
    camera_matrices,
    dist_coeffs_list,
    img_size,
    num_cameras,
):
    """Stereo-calibrate every camera relative to camera 0 using ChArUco.

    Only uses frames where BOTH camera 0 and camera N detected the board,
    and only uses the COMMON corner IDs between the two views.
    """
    R_matrices = [np.eye(3)] + [None] * (num_cameras - 1)
    T_vectors = [np.zeros((3, 1))] + [None] * (num_cameras - 1)

    print("\n" + "=" * 60)
    print("STEREO PAIR CALIBRATION (ChArUco)")
    print("=" * 60)

    board_corners_3d = board.getChessboardCorners()

    for idx in range(1, num_cameras):
        print(f"\nCalibrating camera {idx} relative to camera 0...")

        obj_points_common = []
        img_points_0 = []
        img_points_n = []

        num_frames = len(all_corners[0])
        for f in range(num_frames):
            corners_0, ids_0 = all_corners[0][f], all_ids[0][f]
            corners_n, ids_n = all_corners[idx][f], all_ids[idx][f]

            if corners_0 is None or corners_n is None:
                continue

            # Find common corner IDs visible in BOTH cameras
            ids_0_flat = ids_0.flatten()
            ids_n_flat = ids_n.flatten()
            common_ids = np.intersect1d(ids_0_flat, ids_n_flat)

            if len(common_ids) < 6:
                continue

            # Build matched arrays using only common IDs
            obj_pts = np.array(
                [board_corners_3d[i] for i in common_ids], dtype=np.float32
            )

            # Map from id -> index in each detection
            id_to_idx_0 = {int(id_val): i for i, id_val in enumerate(ids_0_flat)}
            id_to_idx_n = {int(id_val): i for i, id_val in enumerate(ids_n_flat)}

            img_pts_0 = np.array(
                [corners_0[id_to_idx_0[int(cid)]].flatten() for cid in common_ids],
                dtype=np.float32,
            )
            img_pts_n = np.array(
                [corners_n[id_to_idx_n[int(cid)]].flatten() for cid in common_ids],
                dtype=np.float32,
            )

            obj_points_common.append(obj_pts.reshape(-1, 1, 3))
            img_points_0.append(img_pts_0.reshape(-1, 1, 2))
            img_points_n.append(img_pts_n.reshape(-1, 1, 2))

        if len(obj_points_common) < 5:
            print(
                f"  Not enough common frames ({len(obj_points_common)}). Stereo calibration FAILED!"
            )
            continue

        ret, *_, R, T, _E, _F = cv2.stereoCalibrate(
            obj_points_common,
            img_points_0,
            img_points_n,
            camera_matrices[0],
            dist_coeffs_list[0],
            camera_matrices[idx],
            dist_coeffs_list[idx],
            img_size,
            flags=cv2.CALIB_FIX_INTRINSIC,
        )
        if ret:
            R_matrices[idx] = R
            T_vectors[idx] = T
            print(f"  Stereo reprojection error: {ret:.4f} px")
            print(f"  Baseline distance: {np.linalg.norm(T):.1f} mm")
            print(f"  Common frames used: {len(obj_points_common)}")
        else:
            print("  Stereo calibration FAILED!")

    return R_matrices, T_vectors


def save_calibration(
    num_cameras,
    img_size,
    num_captured,
    camera_matrices,
    dist_coeffs_list,
    R_matrices,
    T_vectors,
):
    """Persist calibration to .npz (latest + timestamped backup)."""
    print("\n" + "=" * 60)
    print("SAVING CALIBRATION DATA")
    print("=" * 60)

    os.makedirs(CALIBRATION_DIR, exist_ok=True)

    data = {
        "num_cameras": num_cameras,
        "camera_ids": list(range(num_cameras)),
        "img_size": img_size,
        "num_captures": num_captured,
        "calibration_type": "charuco",
        "charuco_squares_x": CHARUCO_SQUARES_X,
        "charuco_squares_y": CHARUCO_SQUARES_Y,
        "charuco_square_length": CHARUCO_SQUARE_LENGTH,
        "charuco_marker_length": CHARUCO_MARKER_LENGTH,
    }
    for idx in range(num_cameras):
        data[f"camera_matrix_{idx}"] = camera_matrices[idx]
        data[f"dist_coeffs_{idx}"] = dist_coeffs_list[idx]
        data[f"R_{idx}"] = R_matrices[idx]
        data[f"T_{idx}"] = T_vectors[idx]

    np.savez(CALIBRATION_FILE, **data)
    print(f"✓ Calibration saved to: {CALIBRATION_FILE}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = os.path.join(CALIBRATION_DIR, f"calib_backup_{ts}.npz")
    np.savez(backup, **data)
    print(f"✓ Backup saved to: {backup}")


# ---------------------------------------------------------------------------
# Main capture + calibrate loop
# ---------------------------------------------------------------------------
def main():
    # --- Camera setup via CameraManager ---
    mgr = CameraManager()
    if mgr.num_cameras == 0:
        print("No cameras found. Exiting.")
        return

    mgr.start_all(exposure=10)
    num_cameras = mgr.num_cameras
    img_size = mgr.get_resolution(0)  # (width, height)

    window_name = "OptiTrack Multi-Camera ChArUco Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("Running calibration capture. Press 'q' to quit early.")
    print(f"Board size: {CHARUCO_SQUARES_X}x{CHARUCO_SQUARES_Y} ChArUco")
    print(f"Target captures: {NUM_CALIBRATION_IMAGES}")
    print(f"Configured dictionary: {ARUCO_DICT}")
    print()

    # --- Auto-detect the ArUco dictionary from the first frame ---
    print("Hold the ChArUco board in front of any camera for auto-detection...")
    detected_dict = None
    while detected_dict is None:
        raw_frames = mgr.get_all_frames()

        # Show live preview while waiting for auto-detection
        display_frames = []
        for idx, frame in enumerate(raw_frames):
            disp = frame.copy()
            cv2.putText(
                disp,
                f"Camera {idx}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                disp,
                "Detecting dictionary...",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
            display_frames.append(disp)
        grid = build_grid(display_frames)
        h, w = grid.shape[:2]
        grid = cv2.resize(
            grid, (int(w * 0.5), int(h * 0.5)), interpolation=cv2.INTER_AREA
        )
        cv2.putText(
            grid,
            "Show the ChArUco board to any camera...",
            (20, grid.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )
        cv2.imshow(window_name, grid)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Cancelled.")
            mgr.stop_all()
            return

        # Try auto-detection on each camera
        for cam_idx, frame in enumerate(raw_frames):
            dict_name, n_corners = auto_detect_dictionary(frame)
            if dict_name is not None and n_corners >= 4:
                detected_dict = dict_name
                print(
                    f"✓ Auto-detected dictionary: {detected_dict} ({n_corners} corners from camera {cam_idx})"
                )
                break

    # Build the detector with the auto-detected dictionary
    board, detector = build_charuco_detector(dict_name=detected_dict)

    print(f"\nUsing dictionary: {detected_dict}")
    print("Auto-capture every 1 s when ALL cameras detect the board.\n")

    # We will store the raw corners and ids for each camera
    # all_corners[cam_idx] = list of charuco_corners arrays
    # all_ids[cam_idx] = list of charuco_ids arrays
    all_corners = [[] for _ in range(num_cameras)]
    all_ids = [[] for _ in range(num_cameras)]

    num_captured = 0
    last_capture_time = time.time()
    capture_delay = 1.0  # seconds (fast polling for calibration)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_cameras) as executor:
            while num_captured < NUM_CALIBRATION_IMAGES:
                raw_frames = mgr.get_all_frames()

                detections = []  # bool per camera
                corners_data_list = []  # (corners, ids) or (None, None) per camera
                display_frames = []

                # Run ChArUco detection in parallel for all cameras
                args_list = [(frame, detector) for frame in raw_frames]
                results = list(executor.map(detect_charuco, args_list))

                for idx, (frame, (found, corners_data)) in enumerate(
                    zip(raw_frames, results)
                ):
                    detections.append(found)
                    corners_data_list.append(corners_data)
                    display_frames.append(
                        annotate_frame(frame, idx, found, corners_data)
                    )

                grid = build_grid(display_frames)

                # Resize grid to fit on screen better while maintaining aspect ratio
                h, w = grid.shape[:2]
                scale = 0.5  # Scale down by 50%
                grid = cv2.resize(
                    grid, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
                )

                all_valid = all(detections)
                elapsed = time.time() - last_capture_time

                # Status text
                if all_valid:
                    if elapsed >= capture_delay:
                        # --- auto-capture ---
                        for idx in range(num_cameras):
                            charuco_corners, charuco_ids = corners_data_list[idx]
                            all_corners[idx].append(charuco_corners)
                            all_ids[idx].append(charuco_ids)
                        num_captured += 1
                        last_capture_time = time.time()
                        print(f"✓ Captured {num_captured}/{NUM_CALIBRATION_IMAGES}")
                        status_text = (
                            f"CAPTURED! ({num_captured}/{NUM_CALIBRATION_IMAGES})"
                        )
                        status_color = (0, 255, 0)
                    else:
                        countdown = int(capture_delay - elapsed)
                        status_text = (
                            f"Capturing in {countdown}... "
                            f"({num_captured}/{NUM_CALIBRATION_IMAGES})"
                        )
                        status_color = (0, 255, 255)  # yellow
                else:
                    status_text = (
                        f"Move board until all cameras see it "
                        f"({num_captured}/{NUM_CALIBRATION_IMAGES})"
                    )
                    status_color = (0, 0, 255)

                cv2.putText(
                    grid,
                    status_text,
                    (20, grid.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    status_color,
                    3,
                )

                cv2.imshow(window_name, grid)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cv2.destroyAllWindows()

        if num_captured >= 10:
            print(f"\n✓ Captured {num_captured} image sets. Starting calibration...")

            camera_matrices, dist_coeffs_list = calibrate_intrinsics(
                all_corners,
                all_ids,
                board,
                img_size,
                num_cameras,
            )

            R_matrices, T_vectors = calibrate_stereo_pairs(
                all_corners,
                all_ids,
                board,
                camera_matrices,
                dist_coeffs_list,
                img_size,
                num_cameras,
            )

            save_calibration(
                num_cameras,
                img_size,
                num_captured,
                camera_matrices,
                dist_coeffs_list,
                R_matrices,
                T_vectors,
            )

            print("\n" + "=" * 60)
            print("✓ CALIBRATION COMPLETE!")
            print("=" * 60)
        else:
            print(f"\n⚠ Not enough captures ({num_captured}). Need at least 10.")

        mgr.stop_all()


if __name__ == "__main__":
    main()
