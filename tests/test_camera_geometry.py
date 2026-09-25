"""Tests for handtrack.processing._camera_geometry using synthetic cameras."""

import math

import numpy as np

from handtrack.processing._camera_geometry import (
    anisotropy_ratio,
    build_dlt_matrix,
    camera_center,
    compute_fov,
    condition_number,
    coverage_map,
    depth_uncertainty,
    epipolar_consistency,
    essential_matrix,
    fundamental_matrix,
    ground_sampling_distance,
    intrinsic_properties,
    is_visible,
    min_intersection_angle,
    pairwise_baselines,
    projection_matrix,
    ray_intersection_angles,
    rms_uncertainty,
    triangulation_covariance,
    visibility_mask,
    worst_axis_uncertainty,
)

# ---------------------------------------------------------------------------
# Helpers — build synthetic cameras
# ---------------------------------------------------------------------------

def _make_intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


IMG_SIZE = (640, 480)

# Two cameras looking at the origin from opposite sides along the X-axis,
# separated by 0.5 units.
# OpenCV convention: R transforms world->camera.  Camera z-axis = optical axis.
# For a camera at position C looking toward the origin, the camera z-axis in
# world coords is  z_w = (origin - C) / |origin - C|.  We build R so that
# row 2 of R (the camera z-axis expressed in world) equals z_w.
def _look_at(position, target=np.zeros(3)):
    """Build R, T for a camera at *position* looking at *target*."""
    z = target - position
    z = z / np.linalg.norm(z)
    up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        up = np.array([0.0, 0.0, 1.0])
        x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])  # (3,3) — rows are camera axes in world
    T = (-R @ position).reshape(3, 1)
    return R, T


def _two_camera_setup():
    K = _make_intrinsics()
    R0, T0 = _look_at(np.array([-0.25, 0.0, 0.0]))
    R1, T1 = _look_at(np.array([0.25, 0.0, 0.0]))

    cameras = [
        {"K": K, "R": R0, "T": T0, "dist": np.zeros(5), "img_size": IMG_SIZE},
        {"K": K, "R": R1, "T": T1, "dist": np.zeros(5), "img_size": IMG_SIZE},
    ]
    for c in cameras:
        c["P"] = projection_matrix(c["K"], c["R"], c["T"])
        c["C"] = camera_center(c["R"], c["T"])
    return cameras


def _four_camera_setup():
    """Four cameras at corners of a square in the XY plane, all looking at origin."""
    K = _make_intrinsics()
    positions = [
        np.array([-0.2, -0.2, 0.3]),
        np.array([0.2, -0.2, 0.3]),
        np.array([0.2, 0.2, 0.3]),
        np.array([-0.2, 0.2, 0.3]),
    ]
    cameras = []
    for pos in positions:
        R, T = _look_at(pos)
        cam = {
            "K": K, "R": R, "T": T,
            "dist": np.zeros(5), "img_size": IMG_SIZE,
        }
        cam["P"] = projection_matrix(K, R, T)
        cam["C"] = camera_center(R, T)
        cameras.append(cam)
    return cameras


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCameraCenter:
    def test_identity(self):
        R = np.eye(3)
        T = np.array([[1], [2], [3]], dtype=np.float64)
        C = camera_center(R, T)
        np.testing.assert_allclose(C, [-1, -2, -3])

    def test_roundtrip(self):
        cameras = _two_camera_setup()
        np.testing.assert_allclose(cameras[0]["C"], [-0.25, 0, 0], atol=1e-10)
        np.testing.assert_allclose(cameras[1]["C"], [0.25, 0, 0], atol=1e-10)


class TestFOV:
    def test_known_fov(self):
        K = _make_intrinsics(fx=320, fy=240)
        fov_h, fov_v = compute_fov(K, (640, 480))
        assert abs(math.degrees(fov_h) - 90.0) < 0.1
        assert abs(math.degrees(fov_v) - 90.0) < 0.1

    def test_symmetric(self):
        K = _make_intrinsics(fx=500, fy=500)
        fov_h, fov_v = compute_fov(K, (640, 480))
        assert fov_h > fov_v  # wider image -> wider horizontal FOV


class TestVisibility:
    def test_origin_visible_from_both(self):
        cameras = _two_camera_setup()
        for cam in cameras:
            assert is_visible(
                np.array([0, 0, 0]), cam["K"], cam["R"], cam["T"],
                cam["img_size"], near=0.01, far=1.0,
            )

    def test_behind_camera(self):
        cameras = _two_camera_setup()
        # Point far behind camera 0 (at -0.25 looking +X, so -1.0 is behind)
        assert not is_visible(
            np.array([-1.0, 0, 0]), cameras[0]["K"], cameras[0]["R"],
            cameras[0]["T"], cameras[0]["img_size"], near=0.01, far=2.0,
        )

    def test_vectorised_matches_scalar(self):
        cameras = _two_camera_setup()
        cam = cameras[0]
        pts = np.array([[0, 0, 0], [-1.0, 0, 0], [0, 0.01, 0]])
        mask = visibility_mask(pts, cam["K"], cam["R"], cam["T"], cam["img_size"], 0.01, 1.0)
        for i, pt in enumerate(pts):
            expected = is_visible(pt, cam["K"], cam["R"], cam["T"], cam["img_size"], 0.01, 1.0)
            assert mask[i] == expected, f"Mismatch at point {i}"


class TestCoverage:
    def test_origin_seen_by_both(self):
        cameras = _two_camera_setup()
        pts = np.array([[0.0, 0.0, 0.0]])
        counts = coverage_map(pts, cameras, near=0.01, far=1.0)
        assert counts[0] == 2


class TestDLT:
    # Use a slightly off-centre point to avoid the degenerate case where the
    # point projects exactly onto the principal point of opposing cameras,
    # making the DLT rows linearly dependent.
    _TEST_PT = np.array([0.05, 0.03, 0.01])

    def test_condition_number_orthogonal(self):
        """Two cameras at 180 deg should have finite conditioning at a generic point."""
        cameras = _two_camera_setup()
        Ps = [c["P"] for c in cameras]
        A = build_dlt_matrix(self._TEST_PT, Ps)
        cn = condition_number(A)
        assert np.isfinite(cn)
        assert cn < 1e6

    def test_covariance_finite(self):
        cameras = _two_camera_setup()
        A = build_dlt_matrix(self._TEST_PT, [c["P"] for c in cameras])
        cov = triangulation_covariance(A, sigma_px=1.0)
        assert np.all(np.isfinite(cov))
        assert rms_uncertainty(cov) > 0
        assert worst_axis_uncertainty(cov) > 0

    def test_more_cameras_reduces_uncertainty(self):
        """Four cameras should yield lower uncertainty than two."""
        two = _two_camera_setup()
        four = _four_camera_setup()

        A2 = build_dlt_matrix(self._TEST_PT, [c["P"] for c in two])
        A4 = build_dlt_matrix(self._TEST_PT, [c["P"] for c in four])

        rms2 = rms_uncertainty(triangulation_covariance(A2, 1.0))
        rms4 = rms_uncertainty(triangulation_covariance(A4, 1.0))
        assert rms4 < rms2


class TestRayAngles:
    def test_opposing_cameras_180(self):
        cameras = _two_camera_setup()
        pt = np.array([0, 0, 0])
        angles = ray_intersection_angles(pt, [c["C"] for c in cameras])
        # Cameras on opposite sides -> ~180 deg
        assert abs(angles[0] - 180.0) < 1.0

    def test_orthogonal_cameras_90(self):
        C0 = np.array([1, 0, 0])
        C1 = np.array([0, 1, 0])
        pt = np.array([0, 0, 0])
        angles = ray_intersection_angles(pt, [C0, C1])
        assert abs(angles[0] - 90.0) < 0.1

    def test_min_angle(self):
        Cs = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        pt = np.array([0, 0, 0])
        ma = min_intersection_angle(pt, Cs)
        assert abs(ma - 90.0) < 0.1


class TestBaselines:
    def test_two_cameras(self):
        cameras = _two_camera_setup()
        bl = pairwise_baselines([c["C"] for c in cameras])
        assert abs(bl[(0, 1)] - 0.5) < 1e-6

    def test_depth_uncertainty_formula(self):
        # sigma_Z = Z^2 * sigma_px / (f * B)
        sigma = depth_uncertainty(500.0, 0.5, 1.0, 1.0)
        expected = 1.0 ** 2 * 1.0 / (500.0 * 0.5)
        assert abs(sigma - expected) < 1e-10


class TestEpipolar:
    def test_det_F_near_zero(self):
        cameras = _two_camera_setup()
        R_rel = cameras[1]["R"] @ cameras[0]["R"].T
        T_rel = cameras[1]["T"] - cameras[1]["R"] @ cameras[0]["R"].T @ cameras[0]["T"]
        E = essential_matrix(R_rel, T_rel)
        F = fundamental_matrix(cameras[0]["K"], cameras[1]["K"], E)
        res = epipolar_consistency(F)
        assert res < 1e-6


class TestIntrinsics:
    def test_properties(self):
        K = _make_intrinsics()
        props = intrinsic_properties(K, np.zeros(5), IMG_SIZE)
        assert abs(props["focal_ratio"] - 1.0) < 1e-6
        assert props["fov_h_deg"] > 0
        assert props["distortion_magnitude"] == 0.0

    def test_gsd(self):
        # At depth 1.0 with f=500px -> GSD = 1/500 = 0.002 per pixel
        gsd = ground_sampling_distance(500.0, 1.0)
        assert abs(gsd - 0.002) < 1e-10


class TestAnisotropy:
    def test_isotropic_near_one(self):
        cov = np.eye(3) * 0.01
        assert abs(anisotropy_ratio(cov) - 1.0) < 1e-6

    def test_elongated(self):
        cov = np.diag([0.01, 0.01, 1.0])
        assert anisotropy_ratio(cov) > 5.0
