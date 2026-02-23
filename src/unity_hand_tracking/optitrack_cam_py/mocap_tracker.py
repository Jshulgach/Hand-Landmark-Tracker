"""
Multi-camera hand tracking with 3D triangulation.

Uses CameraManager (multi_mjpeg.py) for OptiTrack camera capture.
MediaPipe hand detection is parallelised across cameras using threads.
Calibration is loaded from the .npz file produced by calibration.py.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import mediapipe as mp
import numpy as np
from config import (
    CALIBRATION_FILE,
    HAND_MATCH_THRESHOLD,
    MAX_HANDS,
    MAX_REPROJECTION_ERROR,
    MIN_CAMERAS_FOR_TRIANGULATION,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MODEL_COMPLEXITY,
    NUM_LANDMARKS,
    TRIANGULATION_METHOD,
)
from multi_mjpeg import CameraManager


class MultiCameraTracker:
    """Tracks hands across multiple calibrated OptiTrack cameras."""

    def __init__(self):
        # Will be populated on initialize_cameras()
        self.cam_mgr: CameraManager | None = None
        self.num_cameras = 0
        self.camera_ids = []
        self.num_landmarks = NUM_LANDMARKS
        self.img_width = 0
        self.img_height = 0

        # MediaPipe Hands instances (one per camera — not thread-safe to share)
        self.hands_detectors: list[mp.solutions.hands.Hands] = []

        # Calibration data
        self.camera_matrices = []
        self.dist_coeffs = []
        self.R_matrices = []
        self.T_vectors = []
        self.projection_matrices = []

        # Set of camera indices used for triangulation (all enabled by default)
        self._enabled_cameras: set[int] | None = None  # None = all enabled

        # Thread pool for parallel MediaPipe inference
        self._executor: ThreadPoolExecutor | None = None

        # FPS / timing stats
        self._t_prev = time.perf_counter()
        self._fps = 0.0
        self._capture_ms = 0.0
        self._detection_ms = 0.0
        self._triangulation_ms = 0.0
        self._camera_confidence = {}

    # ------------------------------------------------------------------ #
    # Calibration
    # ------------------------------------------------------------------ #
    def load_calibration(self):
        """Load camera calibration data from .npz file."""
        try:
            data = np.load(CALIBRATION_FILE, allow_pickle=True)
            n = int(data["num_cameras"])

            for idx in range(n):
                self.camera_matrices.append(data[f"camera_matrix_{idx}"])
                self.dist_coeffs.append(data[f"dist_coeffs_{idx}"])
                self.R_matrices.append(data[f"R_{idx}"])
                self.T_vectors.append(data[f"T_{idx}"])

            self._compute_projection_matrices()
            print(f"✓ Loaded calibration for {n} cameras from {CALIBRATION_FILE}")
        except Exception as e:
            print(f"✗ Error loading calibration: {e}")
            raise

    def _compute_projection_matrices(self):
        """P = K @ [R | T] for each camera."""
        self.projection_matrices = []
        for idx in range(len(self.camera_matrices)):
            RT = np.hstack([self.R_matrices[idx], self.T_vectors[idx]])
            P = self.camera_matrices[idx] @ RT
            self.projection_matrices.append(P)

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #
    def initialize_cameras(
        self,
        min_det_conf=MIN_DETECTION_CONFIDENCE,
        min_track_conf=MIN_TRACKING_CONFIDENCE,
    ) -> bool:
        """
        Start CameraManager, create per-camera MediaPipe detectors,
        and spin up the thread pool.
        """
        # --- cameras via CameraManager ---
        self.cam_mgr = CameraManager()
        self.num_cameras = self.cam_mgr.num_cameras

        if self.num_cameras == 0:
            print("No cameras detected.")
            return False

        self.cam_mgr.start_all()
        self.camera_ids = list(range(self.num_cameras))

        w, h = self.cam_mgr.get_resolution(0)
        self.img_width = w
        self.img_height = h

        # --- calibration ---
        self.load_calibration()

        # Sanity check: calibration camera count must match hardware
        if len(self.camera_matrices) != self.num_cameras:
            print(
                f"⚠ Calibration has {len(self.camera_matrices)} cameras "
                f"but {self.num_cameras} are connected."
            )

        # --- MediaPipe (one instance per camera for thread safety) ---
        self.hands_detectors = []
        for _ in range(self.num_cameras):
            hands = mp.solutions.hands.Hands(
                max_num_hands=MAX_HANDS,
                model_complexity=MODEL_COMPLEXITY,
                min_detection_confidence=min_det_conf,
                min_tracking_confidence=min_track_conf,
            )
            self.hands_detectors.append(hands)

        # --- thread pool (one worker per camera) ---
        self._executor = ThreadPoolExecutor(
            max_workers=self.num_cameras,
            thread_name_prefix="mp_detect",
        )

        print(f"✓ {self.num_cameras} cameras initialised with threaded MediaPipe")
        return True

    def update_mp_params(self, min_det_conf, min_track_conf):
        """Re-initialize MediaPipe detectors with new confidence thresholds."""
        if not self.hands_detectors:
            return

        for hands in self.hands_detectors:
            hands.close()
        self.hands_detectors.clear()

        for _ in range(self.num_cameras):
            hands = mp.solutions.hands.Hands(
                max_num_hands=MAX_HANDS,
                model_complexity=MODEL_COMPLEXITY,
                min_detection_confidence=min_det_conf,
                min_tracking_confidence=min_track_conf,
            )
            self.hands_detectors.append(hands)
        print(
            f"Updated MediaPipe params: det={min_det_conf:.2f}, track={min_track_conf:.2f}"
        )

    # ------------------------------------------------------------------ #
    # Capture
    # ------------------------------------------------------------------ #
    def capture_frames(self):
        """Grab the latest BGR frame from every camera via CameraManager."""
        t0 = time.perf_counter()
        frames = self.cam_mgr.get_all_frames()
        self._capture_ms = (time.perf_counter() - t0) * 1000
        return frames

    # ------------------------------------------------------------------ #
    # Hand detection (parallelised)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _detect_single_camera(detector, frame):
        """Run MediaPipe on one frame. Designed to run in a thread."""
        if frame is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return detector.process(rgb)

    def detect_hands_all_cameras(self, frames):
        """Detect hands in all cameras **in parallel**."""
        t0 = time.perf_counter()

        futures = []
        for idx, frame in enumerate(frames):
            fut = self._executor.submit(
                self._detect_single_camera,
                self.hands_detectors[idx],
                frame,
            )
            futures.append(fut)

        all_results = [f.result() for f in futures]
        self._detection_ms = (time.perf_counter() - t0) * 1000
        return all_results

    # ------------------------------------------------------------------ #
    # 2-D landmark extraction
    # ------------------------------------------------------------------ #
    def extract_landmarks_2d(self, results):
        """
        Extract + undistort 2D landmarks from MediaPipe results.

        Returns
        -------
        list[list[tuple(ndarray, ndarray, list[float])]]
            Per camera → per hand → (landmarks_px (21,2), camera_confidence, landmark_confidences)
        """
        all_landmarks_2d = []
        conf_debug = {}

        h, w = self.img_height, self.img_width

        for cam_idx, result in enumerate(results):
            if result is None or not result.multi_hand_landmarks:
                all_landmarks_2d.append([])
                conf_debug[cam_idx] = 0.0
                continue

            try:
                camera_conf = result.multi_handedness[0].classification[0].score
            except Exception:
                camera_conf = 0.0

            conf_debug[cam_idx] = camera_conf

            camera_hands = []
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks_norm = np.array(
                    [[lm.x, lm.y] for lm in hand_landmarks.landmark]
                )

                # Extract individual landmark confidences (visibility/presence)
                # MediaPipe Hands doesn't explicitly expose per-landmark confidence in the same way Pose does,
                # but we can use the z-coordinate (depth) or just the overall camera confidence as a fallback.
                # For now, we'll use the overall camera confidence for all landmarks, but structure it
                # so it can be easily updated if MediaPipe adds per-landmark confidence.
                landmark_confs = [camera_conf] * self.num_landmarks

                # Normalised → pixel
                landmarks_px = landmarks_norm * [w, h]

                # Undistort
                landmarks_px = landmarks_px.reshape(-1, 1, 2)
                landmarks_undist = cv2.undistortPoints(
                    landmarks_px,
                    self.camera_matrices[cam_idx],
                    self.dist_coeffs[cam_idx],
                    P=self.camera_matrices[cam_idx],
                )
                landmarks_undist = landmarks_undist.reshape(-1, 2)

                camera_hands.append((landmarks_undist, camera_conf, landmark_confs))

            all_landmarks_2d.append(camera_hands)

        # if conf_debug:
        #     line = " | ".join(
        #         f"Cam {cid}: {conf_debug[cid]:.2f}" for cid in sorted(conf_debug.keys())
        #     )
        #     print(f"\rHand Confidence → {line}", end="", flush=True)

        return all_landmarks_2d

    # ------------------------------------------------------------------ #
    # Hand matching across cameras
    # ------------------------------------------------------------------ #
    def match_hands_across_cameras(self, all_landmarks_2d):
        """
        Greedy matching: for each hand in the reference camera, find the closest
        hand in every other enabled camera by centroid distance.

        Returns list of dicts  {cam_idx: hand_idx_in_that_camera}
        """
        # Determine which cameras are active for triangulation
        if self._enabled_cameras is not None:
            active = sorted(self._enabled_cameras & set(range(self.num_cameras)))
        else:
            active = list(range(self.num_cameras))

        if len(active) < MIN_CAMERAS_FOR_TRIANGULATION:
            return []

        # Use first enabled camera as reference
        ref_cam = active[0]
        if not all_landmarks_2d[ref_cam]:
            return []

        matched_groups = []

        for hand_idx_ref, (landmarks_ref, _conf_ref, _lm_confs_ref) in enumerate(
            all_landmarks_2d[ref_cam]
        ):
            group = {ref_cam: hand_idx_ref}
            centroid_ref = np.mean(landmarks_ref, axis=0)

            for cam_idx in active:
                if cam_idx == ref_cam:
                    continue
                if not all_landmarks_2d[cam_idx]:
                    continue

                best_idx = None
                best_dist = float("inf")

                for hand_idx, (landmarks, _conf, _lm_confs) in enumerate(
                    all_landmarks_2d[cam_idx]
                ):
                    centroid = np.mean(landmarks, axis=0)
                    dist = np.linalg.norm(centroid - centroid_ref) / self.img_width
                    if dist < best_dist and dist < HAND_MATCH_THRESHOLD:
                        best_dist = dist
                        best_idx = hand_idx

                if best_idx is not None:
                    group[cam_idx] = best_idx

            if len(group) >= MIN_CAMERAS_FOR_TRIANGULATION:
                matched_groups.append(group)

        return matched_groups

    # ------------------------------------------------------------------ #
    # Triangulation
    # ------------------------------------------------------------------ #
    def _reprojection_error(self, pt3d, points_2d, camera_indices):
        """Compute average reprojection error of a 3D point across cameras."""
        pt_h = np.append(pt3d, 1.0)
        total_error = 0.0
        for k in range(len(points_2d)):
            P = self.projection_matrices[camera_indices[k]]
            projected = P @ pt_h
            projected = projected[:2] / projected[2]
            total_error += np.linalg.norm(projected - points_2d[k])
        return total_error / len(points_2d)

    def _triangulate_n_views(self, points_2d, projection_matrices):
        """
        Triangulate a 3D point from N views using Direct Linear Transform (DLT) and SVD.
        points_2d: list of (2,) arrays
        projection_matrices: list of (3, 4) projection matrices
        """
        A = []
        for pt, P in zip(points_2d, projection_matrices):
            u, v = pt[0], pt[1]
            A.append(u * P[2, :] - P[0, :])
            A.append(v * P[2, :] - P[1, :])
        A = np.array(A)
        _, _, Vh = np.linalg.svd(A)
        X = Vh[-1, :]
        return X[:3] / X[3]

    def triangulate_landmark(self, points_2d, camera_indices, camera_confidences=None):
        """
        Triangulate one landmark from ≥2 views. Returns (pt3d, used_cameras, error) or (None, [], inf).

        TRIANGULATION_METHOD controls behavior:
          - "simple_average":   ref-based pairs, unweighted mean
          - "weighted_average": ref-based pairs, weighted by camera_confidence
          - "reprojection":     all pairs, weighted by inverse reprojection error
          - "weighted_error":   finds the single pair with the lowest reprojection error
          - "best_triplet":     finds the triplet of cameras with the lowest reprojection error using N-view DLT
        """
        n = len(points_2d)
        if n < 2:
            return None, [], float("inf")

        method = TRIANGULATION_METHOD
        if method == "best_triplet":
            best_pt3d = None
            best_error = float("inf")
            best_cams = []

            if n >= 3:
                import itertools

                for combo in itertools.combinations(range(n), 3):
                    pts = [points_2d[idx] for idx in combo]
                    Ps = [
                        self.projection_matrices[camera_indices[idx]] for idx in combo
                    ]
                    pt3d = self._triangulate_n_views(pts, Ps)

                    # Calculate reprojection error for this triplet
                    error = self._reprojection_error(
                        pt3d,
                        pts,
                        [camera_indices[idx] for idx in combo],
                    )

                    if error < best_error:
                        best_error = error
                        best_pt3d = pt3d
                        best_cams = [camera_indices[idx] for idx in combo]
                return best_pt3d, best_cams, best_error
            else:
                # Fallback to best pair if < 3 cameras available
                method = "weighted_error"

        if method == "weighted_error":
            best_pt3d = None
            best_error = float("inf")
            best_pair = []

            for i in range(n):
                for j in range(i + 1, n):
                    pt4d = cv2.triangulatePoints(
                        self.projection_matrices[camera_indices[i]],
                        self.projection_matrices[camera_indices[j]],
                        points_2d[i].reshape(2, 1),
                        points_2d[j].reshape(2, 1),
                    )
                    pt3d = (pt4d[:3] / pt4d[3]).flatten()

                    # Calculate reprojection error for this pair
                    error = self._reprojection_error(
                        pt3d,
                        [points_2d[i], points_2d[j]],
                        [camera_indices[i], camera_indices[j]],
                    )

                    if error < best_error:
                        best_error = error
                        best_pt3d = pt3d
                        best_pair = [camera_indices[i], camera_indices[j]]

            return best_pt3d, best_pair, best_error

        elif method == "reprojection":
            # --- All unique camera pairs ---
            candidates = []
            for i in range(n):
                for j in range(i + 1, n):
                    pt4d = cv2.triangulatePoints(
                        self.projection_matrices[camera_indices[i]],
                        self.projection_matrices[camera_indices[j]],
                        points_2d[i].reshape(2, 1),
                        points_2d[j].reshape(2, 1),
                    )
                    pt3d = (pt4d[:3] / pt4d[3]).flatten()
                    candidates.append(pt3d)

            if not candidates:
                return None, [], float("inf")

            # --- Weight by inverse reprojection error ---
            weights = []
            avg_errors = []
            for pt3d in candidates:
                avg_error = self._reprojection_error(pt3d, points_2d, camera_indices)
                avg_errors.append(avg_error)
                weights.append(1.0 / (avg_error + 1e-6))

            weights = np.array(weights)
            weights /= weights.sum()
            final_pt3d = np.average(candidates, axis=0, weights=weights)
            final_error = self._reprojection_error(
                final_pt3d, points_2d, camera_indices
            )
            return final_pt3d, camera_indices, final_error

        elif method in ("simple_average", "weighted_average"):
            # --- Original ref-based pair approach ---
            tri_pts = []
            ref_idx = camera_indices[0]
            ref_pt = points_2d[0].reshape(2, 1)

            for i in range(1, len(points_2d)):
                other_idx = camera_indices[i]
                other_pt = points_2d[i].reshape(2, 1)

                pt4d = cv2.triangulatePoints(
                    self.projection_matrices[ref_idx],
                    self.projection_matrices[other_idx],
                    ref_pt,
                    other_pt,
                )
                pt3d = pt4d[:3] / pt4d[3]
                tri_pts.append(pt3d.flatten())

            if (
                camera_confidences is not None
                and method == "weighted_average"
            ):
                weights = np.array(camera_confidences[1:], dtype=np.float64)
                wsum = weights.sum()
                if wsum > 0:
                    weights /= wsum
                    final_pt3d = np.average(tri_pts, axis=0, weights=weights)
                    final_error = self._reprojection_error(
                        final_pt3d, points_2d, camera_indices
                    )
                    return final_pt3d, camera_indices, final_error

            final_pt3d = np.mean(tri_pts, axis=0)
            final_error = self._reprojection_error(
                final_pt3d, points_2d, camera_indices
            )
            return final_pt3d, camera_indices, final_error

        else:
            # --- Fallback: single pair ---
            pt4d = cv2.triangulatePoints(
                self.projection_matrices[camera_indices[0]],
                self.projection_matrices[camera_indices[1]],
                points_2d[0].reshape(2, 1),
                points_2d[1].reshape(2, 1),
            )
            final_pt3d = (pt4d[:3] / pt4d[3]).flatten()
            final_error = self._reprojection_error(
                final_pt3d, points_2d[:2], camera_indices[:2]
            )
            return final_pt3d, camera_indices[:2], final_error

    def _evaluate_cameras(self, landmarks_3d, all_landmarks_2d, matched_group):
        """
        Evaluate each camera's view of the 3D hand by projecting it back to 2D.
        Returns:
            camera_errors: dict of cam_idx -> list of 21 errors (or inf if missing)
            valid_landmarks: dict of cam_idx -> list of 21 booleans
        """
        camera_errors = {}
        valid_landmarks = {}

        for cam_idx, hand_idx in matched_group.items():
            landmarks_2d, _, _ = all_landmarks_2d[cam_idx][hand_idx]
            P = self.projection_matrices[cam_idx]

            errors = []
            valids = []
            for lm_idx in range(self.num_landmarks):
                pt3d = landmarks_3d[lm_idx]
                if np.all(pt3d == 0):
                    errors.append(float("inf"))
                    valids.append(False)
                    continue

                pt_h = np.append(pt3d, 1.0)
                projected = P @ pt_h
                projected = projected[:2] / projected[2]

                err = np.linalg.norm(projected - landmarks_2d[lm_idx])
                errors.append(err)
                valids.append(err <= MAX_REPROJECTION_ERROR)

            camera_errors[cam_idx] = errors
            valid_landmarks[cam_idx] = valids

        return camera_errors, valid_landmarks

    def triangulate_hand(self, all_landmarks_2d, matched_group):
        """Triangulate all 21 landmarks for one matched hand → (21,3)."""
        landmarks_3d = []
        cam_indices = list(matched_group.keys())

        for lm_idx in range(self.num_landmarks):
            pts_2d = []
            valid_cams = []
            confs = []

            for cam_idx in cam_indices:
                hand_idx = matched_group[cam_idx]
                landmarks, camera_conf, lm_confs = all_landmarks_2d[cam_idx][hand_idx]

                # Only use this camera's landmark if its confidence is above a threshold
                # (Currently using camera_conf as a proxy for landmark confidence)
                if lm_confs[lm_idx] > 0.1:  # Basic threshold
                    pts_2d.append(landmarks[lm_idx])
                    confs.append(lm_confs[lm_idx])
                    valid_cams.append(cam_idx)

            pt3d, used_cams, error = self.triangulate_landmark(
                pts_2d, valid_cams, confs
            )

            # Thumb occlusion safety check
            # If it's a thumb landmark and the reprojection error is too high, it's likely a guess
            if lm_idx in [1, 2, 3, 4] and error > MAX_REPROJECTION_ERROR:
                pt3d = None  # Drop the landmark, let Kalman filter predict

            if pt3d is not None:
                landmarks_3d.append(pt3d)
            else:
                landmarks_3d.append(np.zeros(3))

        landmarks_3d_array = np.array(landmarks_3d)

        # Evaluate cameras to find the true "best" cameras and valid landmarks
        camera_errors, valid_landmarks = self._evaluate_cameras(
            landmarks_3d_array, all_landmarks_2d, matched_group
        )

        # Determine the "optimal" cameras used for this hand based on number of valid landmarks
        cam_scores = []
        for cam_idx in cam_indices:
            num_valid = sum(valid_landmarks[cam_idx])
            # Average error of valid landmarks (or inf if none)
            avg_err = (
                np.mean(
                    [
                        e
                        for e, v in zip(
                            camera_errors[cam_idx], valid_landmarks[cam_idx]
                        )
                        if v
                    ]
                )
                if num_valid > 0
                else float("inf")
            )
            cam_scores.append((cam_idx, num_valid, avg_err))

        # Sort by num_valid (descending), then avg_err (ascending)
        cam_scores.sort(key=lambda x: (-x[1], x[2]))

        # If we are using best_triplet, highlight the top 3 cameras, otherwise top 2
        num_best_cams = 3 if TRIANGULATION_METHOD == "best_triplet" else 2
        best_cams = [x[0] for x in cam_scores[:num_best_cams]]

        # Debug print to help tune MAX_REPROJECTION_ERROR
        # print(f"Cam scores (cam_idx, num_valid, avg_err): {cam_scores}")

        return landmarks_3d_array, best_cams, valid_landmarks

    # ------------------------------------------------------------------ #
    # Main per-frame pipeline
    # ------------------------------------------------------------------ #
    def process_frame(self):
        """
        Capture → detect (parallel) → match → triangulate.

        Returns
        -------
        frames : list[ndarray]
        triangulated_hands : list[tuple(ndarray (21,3), list[int], dict)]
        all_results : list[mediapipe result | None]
        valid_2d_landmarks : dict[cam_idx][hand_idx_in_cam] -> list[bool]
        """
        frames = self.capture_frames()
        all_results = self.detect_hands_all_cameras(frames)

        t0 = time.perf_counter()
        all_landmarks_2d = self.extract_landmarks_2d(all_results)
        matched_groups = self.match_hands_across_cameras(all_landmarks_2d)

        triangulated_hands = []
        valid_2d_landmarks = {cam_idx: {} for cam_idx in range(self.num_cameras)}

        for group in matched_groups:
            lm3d, best_cams, valid_lms = self.triangulate_hand(all_landmarks_2d, group)
            triangulated_hands.append((lm3d, best_cams, valid_lms))

            for cam_idx, hand_idx_in_cam in group.items():
                valid_2d_landmarks[cam_idx][hand_idx_in_cam] = valid_lms[cam_idx]

        self._triangulation_ms = (time.perf_counter() - t0) * 1000

        # FPS
        now = time.perf_counter()
        dt = now - self._t_prev
        self._fps = 1.0 / dt if dt > 0 else 0.0
        self._t_prev = now

        return frames, triangulated_hands, all_results, valid_2d_landmarks

    def get_fps_stats(self):
        return {
            "fps": self._fps,
            "capture_ms": self._capture_ms,
            "detection_ms": self._detection_ms,
            "triangulation_ms": self._triangulation_ms,
        }

    def set_exposure(self, cam_index: int, value: int):
        """Set exposure for a specific camera at runtime."""
        if self.cam_mgr:
            self.cam_mgr.set_exposure(cam_index, value)

    def set_enabled_cameras(self, enabled: set[int] | None):
        """
        Set which cameras participate in triangulation.

        Parameters
        ----------
        enabled : set of camera indices, or None to enable all.
        """
        self._enabled_cameras = enabled

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #
    def cleanup(self):
        """Release cameras, detectors, and thread pool."""
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None

        for hands in self.hands_detectors:
            hands.close()
        self.hands_detectors.clear()

        if self.cam_mgr:
            self.cam_mgr.stop_all()
            self.cam_mgr = None

        print("Multi-camera tracker cleanup complete.")
