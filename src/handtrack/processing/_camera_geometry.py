"""Pure geometry functions for multi-camera placement evaluation.

All functions operate on numpy arrays with no side effects.  Units follow the
calibration file convention (typically meters from the ChArUco board definition)
unless otherwise noted.

References
----------
- Olague & Mohr, "Optimal Camera Placement for Accurate Reconstruction",
  Pattern Recognition 35:927-944, 2002.
- Hartley & Sturm, "Triangulation", CVIU 68(2):146-157, 1997.
- Hartley & Zisserman, Multiple View Geometry, 2nd ed., Cambridge, 2004.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


# ---------------------------------------------------------------------------
# Basic camera helpers
# ---------------------------------------------------------------------------

def camera_center(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """World-space camera centre: C = -R^T @ T.

    Parameters
    ----------
    R : (3, 3) rotation matrix.
    T : (3, 1) translation vector.

    Returns
    -------
    (3,) camera centre in world coordinates.
    """
    return (-R.T @ T).ravel()


def compute_fov(K: np.ndarray, img_size: tuple[int, int]) -> tuple[float, float]:
    """Horizontal and vertical field of view in radians.

    Parameters
    ----------
    K : (3, 3) intrinsic matrix.
    img_size : (width, height) in pixels.
    """
    w, h = img_size
    fx, fy = K[0, 0], K[1, 1]
    fov_x = 2.0 * np.arctan(w / (2.0 * fx))
    fov_y = 2.0 * np.arctan(h / (2.0 * fy))
    return float(fov_x), float(fov_y)


def projection_matrix(K: np.ndarray, R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """P = K @ [R | T], shape (3, 4)."""
    RT = np.hstack([R, T.reshape(3, 1)])
    return K @ RT


# ---------------------------------------------------------------------------
# Visibility
# ---------------------------------------------------------------------------

def is_visible(
    point: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    img_size: tuple[int, int],
    near: float,
    far: float,
) -> bool:
    """Check whether a 3D point projects inside the image and depth range.

    Parameters
    ----------
    point : (3,) world coordinate.
    K, R, T : camera intrinsics / extrinsics.
    img_size : (width, height).
    near, far : clipping distances along the camera's optical axis.
    """
    # Transform to camera frame
    p_cam = R @ point.ravel() + T.ravel()
    depth = p_cam[2]
    if depth < near or depth > far:
        return False
    # Project
    px = K @ p_cam
    u = px[0] / px[2]
    v = px[1] / px[2]
    w, h = img_size
    return 0 <= u < w and 0 <= v < h


def visibility_mask(
    points: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    img_size: tuple[int, int],
    near: float,
    far: float,
) -> np.ndarray:
    """Vectorised visibility check for an array of 3D points.

    Parameters
    ----------
    points : (N, 3) world coordinates.
    K, R, T : camera parameters.
    img_size : (width, height).
    near, far : depth clipping range.

    Returns
    -------
    (N,) boolean array.
    """
    # Camera frame: p_cam = R @ p + t  for each row p
    p_cam = (R @ points.T).T + T.ravel()  # (N, 3)
    depth = p_cam[:, 2]
    # Project to pixel
    px = (K @ p_cam.T).T  # (N, 3)
    u = px[:, 0] / px[:, 2]
    v = px[:, 1] / px[:, 2]
    w, h = img_size
    mask = (
        (depth >= near)
        & (depth <= far)
        & (u >= 0)
        & (u < w)
        & (v >= 0)
        & (v < h)
    )
    return mask


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def coverage_map(
    grid_points: np.ndarray,
    cameras: list[dict],
    near: float,
    far: float,
) -> np.ndarray:
    """Count how many cameras see each grid point.

    Parameters
    ----------
    grid_points : (N, 3) world coordinates.
    cameras : list of dicts with keys ``K``, ``R``, ``T``, ``img_size``.
    near, far : depth clipping range.

    Returns
    -------
    (N,) integer array of camera counts.
    """
    counts = np.zeros(len(grid_points), dtype=np.int32)
    for cam in cameras:
        mask = visibility_mask(
            grid_points, cam["K"], cam["R"], cam["T"], cam["img_size"], near, far
        )
        counts += mask.astype(np.int32)
    return counts


# ---------------------------------------------------------------------------
# DLT matrix and triangulation quality
# ---------------------------------------------------------------------------

def build_dlt_matrix(
    point: np.ndarray,
    projection_matrices: list[np.ndarray],
) -> np.ndarray:
    """Build the DLT measurement matrix A for a 3D point seen by N cameras.

    This constructs the *same* A used by the existing ``_triangulate_n_views``
    in ``mocap_tracker.py``, but here we evaluate it at a *known* 3D point
    (projected to each camera) rather than from measured 2D detections.

    Parameters
    ----------
    point : (3,) known world coordinate.
    projection_matrices : list of (3, 4) matrices P_i = K_i [R_i | T_i].

    Returns
    -------
    A : (2N, 4) matrix.
    """
    pt_h = np.append(point, 1.0)
    rows = []
    for P in projection_matrices:
        proj = P @ pt_h
        u = proj[0] / proj[2]
        v = proj[1] / proj[2]
        rows.append(u * P[2, :] - P[0, :])
        rows.append(v * P[2, :] - P[1, :])
    return np.array(rows, dtype=np.float64)


def triangulation_covariance(A: np.ndarray, sigma_px: float) -> np.ndarray:
    """3x3 covariance of the triangulated 3D point.

    The DLT system AX = 0 is solved by the right singular vector of A
    corresponding to the smallest singular value.  The 3D point in
    Euclidean coordinates is X[:3]/X[3].  The uncertainty of this estimate
    is derived by propagating pixel noise through the Jacobian of the
    inhomogeneous system obtained by fixing the homogeneous scale.

    We construct the inhomogeneous Jacobian J (2N x 3) by taking the
    first three columns of A minus the fourth column scaled by the
    projected coordinates, then compute  Cov = sigma^2 (J^T J)^{-1}.

    Parameters
    ----------
    A : (2N, 4) DLT matrix.
    sigma_px : assumed isotropic pixel noise standard deviation.

    Returns
    -------
    (3, 3) covariance matrix in world units squared.
    """
    # Solve for the 3D point via SVD
    _, s, Vh = np.linalg.svd(A)
    X_h = Vh[-1]
    if abs(X_h[3]) < 1e-15:
        return np.full((3, 3), np.inf)
    # Inhomogeneous Jacobian: derivative of (A[:,:3] x + A[:,3]) = 0
    # at the solution x = X[:3]/X[3].  This gives J = A[:,:3] + outer product.
    # Simpler equivalent: J_i = P_i[:2,:3] - u_i * P_i[2,:3] (row-pair per cam)
    # which is exactly A[:, :3] when we substitute the projected coords.
    # The Jacobian of the inhomogeneous system is the first 3 cols of A
    # (since the 4th column encodes the constant term).
    J = A[:, :3]
    JtJ = J.T @ J
    try:
        JtJ_inv = np.linalg.inv(JtJ)
    except np.linalg.LinAlgError:
        return np.full((3, 3), np.inf)
    return sigma_px ** 2 * JtJ_inv


def condition_number(A: np.ndarray) -> float:
    """Condition number of the inhomogeneous DLT Jacobian.

    The full (2N x 4) DLT matrix A always has a near-zero smallest singular
    value (the null space that encodes the 3D solution).  The meaningful
    condition number is that of the (2N x 3) Jacobian J = A[:, :3], which
    governs how pixel noise maps to 3D positional error.
    """
    J = A[:, :3]
    s = np.linalg.svd(J, compute_uv=False)
    if s[-1] < 1e-15:
        return np.inf
    return float(s[0] / s[-1])


def rms_uncertainty(cov: np.ndarray) -> float:
    """RMS positional uncertainty = sqrt(trace(Cov_3x3))."""
    return float(np.sqrt(np.trace(cov)))


def worst_axis_uncertainty(cov: np.ndarray) -> float:
    """Worst-axis uncertainty = sqrt(max eigenvalue of Cov_3x3)."""
    eigvals = np.linalg.eigvalsh(cov)
    return float(np.sqrt(max(eigvals)))


def anisotropy_ratio(cov: np.ndarray) -> float:
    """Ratio of worst to best axis uncertainty (> 1 means elongated ellipsoid)."""
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)
    if eigvals[0] < 1e-30:
        return np.inf
    return float(np.sqrt(eigvals[-1] / eigvals[0]))


# ---------------------------------------------------------------------------
# Ray intersection angles
# ---------------------------------------------------------------------------

def ray_intersection_angles(
    point: np.ndarray,
    camera_centers: list[np.ndarray],
) -> np.ndarray:
    """Pairwise convergence angles (degrees) between rays from cameras to point.

    Returns
    -------
    angles : 1-D array of length C(N, 2), one per camera pair.
    """
    rays = [point.ravel() - c.ravel() for c in camera_centers]
    rays = [r / np.linalg.norm(r) for r in rays]
    angles = []
    for r_i, r_j in combinations(rays, 2):
        cos_a = np.clip(np.dot(r_i, r_j), -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cos_a)))
    return np.array(angles, dtype=np.float64)


def min_intersection_angle(
    point: np.ndarray,
    camera_centers: list[np.ndarray],
) -> float:
    """Minimum pairwise convergence angle in degrees."""
    angles = ray_intersection_angles(point, camera_centers)
    if len(angles) == 0:
        return 0.0
    return float(np.min(angles))


# ---------------------------------------------------------------------------
# Baseline / depth precision
# ---------------------------------------------------------------------------

def pairwise_baselines(camera_centers: list[np.ndarray]) -> dict:
    """Compute baseline distances between all camera pairs.

    Returns
    -------
    dict mapping (i, j) -> baseline distance (same units as camera centres).
    """
    baselines = {}
    for (i, c_i), (j, c_j) in combinations(enumerate(camera_centers), 2):
        baselines[(i, j)] = float(np.linalg.norm(c_i - c_j))
    return baselines


def depth_uncertainty(
    focal_length_px: float,
    baseline: float,
    depth: float,
    sigma_px: float,
) -> float:
    """Expected depth uncertainty for a stereo pair.

    sigma_Z = Z^2 * sigma_px / (f * B)

    Parameters
    ----------
    focal_length_px : focal length in pixels.
    baseline : distance between camera centres.
    depth : working distance along viewing direction.
    sigma_px : pixel noise standard deviation.
    """
    if baseline < 1e-12 or focal_length_px < 1e-12:
        return np.inf
    return depth ** 2 * sigma_px / (focal_length_px * baseline)


# ---------------------------------------------------------------------------
# Epipolar geometry
# ---------------------------------------------------------------------------

def _skew(t: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix [t]_x."""
    t = t.ravel()
    return np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0],
    ], dtype=np.float64)


def essential_matrix(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """E = [T]_x R for the relative pose between two cameras.

    Parameters
    ----------
    R : (3, 3) relative rotation.
    T : (3, 1) relative translation.
    """
    return _skew(T) @ R


def fundamental_matrix(
    K_i: np.ndarray,
    K_j: np.ndarray,
    E: np.ndarray,
) -> np.ndarray:
    """F = K_j^{-T} E K_i^{-1}."""
    return np.linalg.inv(K_j).T @ E @ np.linalg.inv(K_i)


def epipolar_consistency(F: np.ndarray) -> float:
    """Consistency residual: |det(F)| should be ~0 for a valid fundamental matrix."""
    return float(abs(np.linalg.det(F)))


# ---------------------------------------------------------------------------
# Intrinsic properties report helpers
# ---------------------------------------------------------------------------

def intrinsic_properties(
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    img_size: tuple[int, int],
) -> dict:
    """Extract human-readable intrinsic properties from a calibration.

    Returns a dict with fx, fy, cx, cy, fov_h, fov_v, fov_diag (degrees),
    focal_ratio, distortion_magnitude, principal_point_offset.
    """
    w, h = img_size
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    fov_h, fov_v = compute_fov(K, img_size)
    fov_diag = 2.0 * np.arctan(
        np.sqrt(w ** 2 + h ** 2) / (2.0 * np.sqrt(fx * fy))
    )
    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "fov_h_deg": float(np.degrees(fov_h)),
        "fov_v_deg": float(np.degrees(fov_v)),
        "fov_diag_deg": float(np.degrees(fov_diag)),
        "focal_ratio": float(fx / fy),
        "distortion_magnitude": float(np.linalg.norm(dist_coeffs)),
        "principal_point_offset_px": float(
            np.sqrt((cx - w / 2) ** 2 + (cy - h / 2) ** 2)
        ),
    }


def ground_sampling_distance(focal_length_px: float, depth: float) -> float:
    """GSD = depth / focal_length  (mm-per-pixel when both are in mm)."""
    if focal_length_px < 1e-12:
        return np.inf
    return depth / focal_length_px
