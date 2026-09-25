"""Legacy one-step multi-camera ChArUco calibration runtime.

This preserves the previous behavior:
- One synchronized capture stream for all cameras.
- A frame set is accepted only when all cameras detect the board.
- Intrinsics and extrinsics are solved from the same synchronized sets.

The implementation reuses the current solver/save pipeline so published artifacts
remain schema-compatible with the rest of the application.
"""

from __future__ import annotations

import concurrent.futures
import time

import cv2

from ._calibration_runtime import (
    ARUCO_DICT,
    CAMERA_EXPOSURE,
    INTRINSIC_CAPTURE_INTERVAL_SEC,
    MIN_INTRINSIC_INLIERS,
    NUM_CALIBRATION_IMAGES,
    _capture_snapshots,
    _load_backend,
    annotate_frame,
    auto_detect_dictionary,
    build_charuco_detector,
    build_grid,
    calibrate_intrinsics,
    calibrate_stereo_pairs,
    detect_charuco,
    save_calibration,
    save_raw_observations,
    validate_heldout_board_reconstruction,
)


def main(backend: str = "webcam"):
    """Run one-step synchronized calibration for the requested camera backend."""
    CameraManager = _load_backend(backend)
    manager = CameraManager()
    if manager.num_cameras == 0:
        print("No cameras found. Exiting.")
        return

    manager.start_all(exposure=CAMERA_EXPOSURE)
    num_cameras = manager.num_cameras
    img_size = manager.get_resolution(0)

    window_name = f"{backend.title()} One-Step ChArUco Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("Running one-step calibration capture. Press 'q' to quit early.")
    print(f"Board size: {build_charuco_detector()[0].getChessboardSize()}")
    print(f"Target synchronized captures: {NUM_CALIBRATION_IMAGES}")
    print(f"Configured dictionary: {ARUCO_DICT}")
    print()

    print("Hold the ChArUco board in front of any camera for auto-detection...")
    detected_dict = None
    while detected_dict is None:
        raw_frames = manager.get_all_frames()
        display_frames = []
        for index, frame in enumerate(raw_frames):
            display = frame.copy()
            cv2.putText(
                display,
                f"Camera {index}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                display,
                "Detecting dictionary...",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
            display_frames.append(display)

        grid = build_grid(display_frames)
        height, width = grid.shape[:2]
        grid = cv2.resize(
            grid,
            (int(width * 0.5), int(height * 0.5)),
            interpolation=cv2.INTER_AREA,
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
            manager.stop_all()
            return

        for camera_index, frame in enumerate(raw_frames):
            dict_name, corner_count = auto_detect_dictionary(frame)
            if dict_name is not None and corner_count >= 4:
                detected_dict = dict_name
                print(
                    "✓ Auto-detected dictionary: "
                    f"{detected_dict} ({corner_count} corners from camera {camera_index})"
                )
                break

    board, detector = build_charuco_detector(dict_name=detected_dict)

    synchronized_corners = [[] for _ in range(num_cameras)]
    synchronized_ids = [[] for _ in range(num_cameras)]
    synchronized_metadata = [[] for _ in range(num_cameras)]

    captures = 0
    last_capture_time = 0.0

    print("\nOne-step synchronized capture enabled.")
    print("A set is captured only when all cameras see the board.")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_cameras) as executor:
            while captures < NUM_CALIBRATION_IMAGES:
                snapshots = _capture_snapshots(manager)
                frames = [snapshot["frame"] for snapshot in snapshots]
                results = list(
                    executor.map(detect_charuco, [(frame, detector) for frame in frames])
                )

                detections = []
                corners_data = []
                display_frames = []
                for index, (frame, (found, data)) in enumerate(zip(frames, results)):
                    detections.append(found)
                    corners_data.append(data)
                    display_frames.append(annotate_frame(frame, index, found, data))

                now = time.time()
                all_valid = all(detections)
                if (
                    all_valid
                    and now - last_capture_time >= INTRINSIC_CAPTURE_INTERVAL_SEC
                ):
                    for camera in range(num_cameras):
                        corners, ids = corners_data[camera]
                        synchronized_corners[camera].append(corners.copy())
                        synchronized_ids[camera].append(ids.copy())
                        synchronized_metadata[camera].append(
                            {
                                key: value
                                for key, value in snapshots[camera].items()
                                if key != "frame"
                            }
                        )
                    captures += 1
                    last_capture_time = now
                    print(f"✓ Captured synchronized set {captures}/{NUM_CALIBRATION_IMAGES}")

                grid = build_grid(display_frames)
                height, width = grid.shape[:2]
                grid = cv2.resize(
                    grid,
                    (int(width * 0.5), int(height * 0.5)),
                    interpolation=cv2.INTER_AREA,
                )

                status_text = (
                    f"SYNC CAPTURE {captures}/{NUM_CALIBRATION_IMAGES}"
                    if all_valid
                    else f"All cameras must see board ({captures}/{NUM_CALIBRATION_IMAGES})"
                )
                status_color = (0, 255, 0) if all_valid else (0, 0, 255)
                cv2.putText(
                    grid,
                    status_text,
                    (20, grid.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    status_color,
                    2,
                )

                cv2.imshow(window_name, grid)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt

        min_required = max(10, MIN_INTRINSIC_INLIERS + 1)
        if captures < min_required:
            raise RuntimeError(
                f"Not enough synchronized captures: {captures}. Need at least {min_required}."
            )

        observation_file = save_raw_observations(
            num_cameras,
            img_size,
            synchronized_corners,
            synchronized_ids,
            synchronized_metadata,
            synchronized_corners,
            synchronized_ids,
            synchronized_metadata,
        )
        camera_matrices, dist_coeffs_list, intrinsic_results = calibrate_intrinsics(
            synchronized_corners,
            synchronized_ids,
            board,
            img_size,
            num_cameras,
        )
        R_matrices, T_vectors, pair_results, loop_rms = calibrate_stereo_pairs(
            synchronized_corners,
            synchronized_ids,
            board,
            camera_matrices,
            dist_coeffs_list,
            img_size,
            num_cameras,
        )
        board_validation = validate_heldout_board_reconstruction(
            synchronized_corners,
            synchronized_ids,
            board.getChessboardCorners(),
            camera_matrices,
            dist_coeffs_list,
            R_matrices,
            T_vectors,
            pair_results[0].heldout_indices,
        )
        identities = [metadata[0] for metadata in synchronized_metadata]
        save_calibration(
            num_cameras,
            img_size,
            captures,
            camera_matrices,
            dist_coeffs_list,
            R_matrices,
            T_vectors,
            camera_source_indices=[item["source_index"] for item in identities],
            camera_source_names=[item["name"] for item in identities],
            camera_identities=identities,
            intrinsic_results=intrinsic_results,
            pair_results=pair_results,
            pose_graph_rms=loop_rms,
            observation_file=observation_file,
            board_validation=board_validation,
        )
        print("\n" + "=" * 60)
        print("✓ ONE-STEP CALIBRATION COMPLETE!")
        print("=" * 60)
    except KeyboardInterrupt:
        print("\nCalibration cancelled; the existing published calibration was unchanged.")
    except Exception as exc:
        print(f"\nCalibration failed: {exc}")
        print("The existing published calibration was unchanged.")
        raise
    finally:
        cv2.destroyAllWindows()
        manager.stop_all()


if __name__ == "__main__":
    main()
