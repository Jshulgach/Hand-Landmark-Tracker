"""
calibration.py — Multi-camera checkerboard calibration using multi_mjpeg.

Uses CameraManager from multi_mjpeg.py as the camera backend instead of
managing cameras directly.
"""

import os
import time
from datetime import datetime

import cv2
import numpy as np
from config import (
    CALIBRATION_DIR,
    CALIBRATION_FILE,
    CHECKERBOARD_COLS,
    CHECKERBOARD_ROWS,
    CHECKERBOARD_SQUARE_SIZE,
    GRID_COLS,
    NUM_CALIBRATION_IMAGES,
)
from multi_mjpeg import CameraManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_object_points_template():
    """3-D coordinates of checkerboard inner corners in board-local frame."""
    pts = np.zeros((CHECKERBOARD_COLS * CHECKERBOARD_ROWS, 3), np.float32)
    pts[:, :2] = np.mgrid[0:CHECKERBOARD_COLS, 0:CHECKERBOARD_ROWS].T.reshape(-1, 2)
    pts *= CHECKERBOARD_SQUARE_SIZE
    return pts


def detect_checkerboard(frame):
    """
    Try to find the checkerboard in *frame*.

    Returns
    -------
    found : bool
    corners : np.ndarray | None   (sub-pixel refined if found)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(
        gray,
        (CHECKERBOARD_COLS, CHECKERBOARD_ROWS),
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_FAST_CHECK
        + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return ret, (corners if ret else None)


def annotate_frame(frame, cam_index, found, corners):
    """Draw camera label, checkerboard overlay, and border colour."""
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
        cv2.drawChessboardCorners(
            display,
            (CHECKERBOARD_COLS, CHECKERBOARD_ROWS),
            corners,
            True,
        )
        for c in corners:
            x, y = c.ravel()
            cv2.circle(display, (int(x), int(y)), 2, (0, 255, 0), -1)
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


def build_grid(frames, grid_cols=GRID_COLS):
    """Stitch list of frames into a row-major grid image."""
    rows = []
    for start in range(0, len(frames), grid_cols):
        row = frames[start : start + grid_cols]
        while len(row) < grid_cols:
            row.append(np.zeros_like(frames[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# Calibration math
# ---------------------------------------------------------------------------
def calibrate_intrinsics(obj_points_all, img_points_all, img_size, num_cameras):
    """Calibrate each camera independently. Returns matrices and dist coeffs."""
    camera_matrices = []
    dist_coeffs_list = []

    print("\n" + "=" * 60)
    print("INDIVIDUAL CAMERA CALIBRATION")
    print("=" * 60)

    for idx in range(num_cameras):
        print(f"\nCalibrating camera {idx}...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points_all[idx],
            img_points_all[idx],
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
        for i in range(len(obj_points_all[idx])):
            proj, _ = cv2.projectPoints(
                obj_points_all[idx][i],
                rvecs[i],
                tvecs[i],
                mtx,
                dist,
            )
            err = cv2.norm(img_points_all[idx][i], proj, cv2.NORM_L2) / len(proj)
            total_err += err
        mean_err = total_err / len(obj_points_all[idx])
        print(f"  Reprojection error: {mean_err:.3f} px")

    return camera_matrices, dist_coeffs_list


def calibrate_stereo_pairs(
    obj_points_all,
    img_points_all,
    camera_matrices,
    dist_coeffs_list,
    img_size,
    num_cameras,
):
    """Stereo-calibrate every camera relative to camera 0."""
    R_matrices = [np.eye(3)] + [None] * (num_cameras - 1)
    T_vectors = [np.zeros((3, 1))] + [None] * (num_cameras - 1)

    print("\n" + "=" * 60)
    print("STEREO PAIR CALIBRATION")
    print("=" * 60)

    for idx in range(1, num_cameras):
        print(f"\nCalibrating camera {idx} relative to camera 0...")
        ret, *_, R, T, _E, _F = cv2.stereoCalibrate(
            obj_points_all[0],
            img_points_all[0],
            img_points_all[idx],
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
            print(f"  Baseline distance: {np.linalg.norm(T):.1f} mm")
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
        "checkerboard_size": (CHECKERBOARD_COLS, CHECKERBOARD_ROWS),
        "square_size": CHECKERBOARD_SQUARE_SIZE,
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

    mgr.start_all()
    num_cameras = mgr.num_cameras
    img_size = mgr.get_resolution(0)  # (width, height)

    window_name = "OptiTrack Multi-Camera Checkerboard Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Running calibration capture. Press 'q' to quit early.")
    print(f"Board size: {CHECKERBOARD_COLS}x{CHECKERBOARD_ROWS}")
    print(f"Target captures: {NUM_CALIBRATION_IMAGES}")
    print("Auto-capture every 1 s when ALL cameras detect the board.")

    obj_template = build_object_points_template()
    obj_points_all = [[] for _ in range(num_cameras)]
    img_points_all = [[] for _ in range(num_cameras)]

    num_captured = 0
    last_capture_time = time.time()
    capture_delay = 1.0  # seconds (fast polling for calibration)

    try:
        while num_captured < NUM_CALIBRATION_IMAGES:
            raw_frames = mgr.get_all_frames()

            detections = []  # bool per camera
            corners_list = []  # corners or None per camera
            display_frames = []

            for idx, frame in enumerate(raw_frames):
                found, corners = detect_checkerboard(frame)
                detections.append(found)
                corners_list.append(corners)
                display_frames.append(annotate_frame(frame, idx, found, corners))

            grid = build_grid(display_frames)

            all_valid = all(detections)
            elapsed = time.time() - last_capture_time

            # Status text
            if all_valid:
                if elapsed >= capture_delay:
                    # --- auto-capture ---
                    for idx in range(num_cameras):
                        obj_points_all[idx].append(obj_template)
                        img_points_all[idx].append(corners_list[idx])
                    num_captured += 1
                    last_capture_time = time.time()
                    print(f"✓ Captured {num_captured}/{NUM_CALIBRATION_IMAGES}")
                    status_text = f"CAPTURED! ({num_captured}/{NUM_CALIBRATION_IMAGES})"
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
                obj_points_all,
                img_points_all,
                img_size,
                num_cameras,
            )

            R_matrices, T_vectors = calibrate_stereo_pairs(
                obj_points_all,
                img_points_all,
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
