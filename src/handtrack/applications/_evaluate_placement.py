"""Quantitative camera placement evaluation from a saved calibration file.

Computes control-volume metrics (coverage, triangulation uncertainty,
convergence angles, DLT condition number, baseline/depth ratios, epipolar
consistency) entirely from intrinsic + extrinsic calibration data.  No live
cameras are required.

Methodology follows Olague & Mohr (2002) for covariance-based placement
analysis, with convergence angle bounds from the ACM VRST optimal placement
paper (40-140 deg acceptable range).

Usage
-----
    handtracker evaluate-placement --backend optitrack
    handtracker evaluate-placement --backend webcam --grid-spacing 5 --output-dir ./placement_report
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import sys
from itertools import combinations
from pathlib import Path
from typing import Sequence, TextIO

import numpy as np

from handtrack.processing._camera_geometry import (
    anisotropy_ratio,
    build_dlt_matrix,
    camera_center,
    condition_number,
    coverage_map,
    depth_uncertainty,
    epipolar_consistency,
    essential_matrix,
    fundamental_matrix,
    ground_sampling_distance,
    intrinsic_properties,
    pairwise_baselines,
    projection_matrix,
    ray_intersection_angles,
    rms_uncertainty,
    triangulation_covariance,
    worst_axis_uncertainty,
)

_CONFIG_MODULES = {
    "optitrack": "handtrack.cameras.optitrack.config",
    "webcam": "handtrack.cameras.webcam.config",
}

# Acceptable convergence angle range (ACM VRST mocap placement paper)
_MIN_CONVERGENCE_DEG = 40.0
_MAX_CONVERGENCE_DEG = 140.0


class _Tee:
    """Write report output to both the terminal and a UTF-8 text file."""

    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


# ---------------------------------------------------------------------------
# Calibration loading
# ---------------------------------------------------------------------------


def _calibration_path(backend: str) -> Path:
    """Resolve a backend calibration without requiring live camera modules."""
    try:
        cfg = importlib.import_module(_CONFIG_MODULES[backend])
    except ModuleNotFoundError as exc:
        # Placement analysis is offline. Some checkouts intentionally keep the
        # vendor-backed camera modules out of source control while retaining
        # calibration_data, so config.py must not be a hard dependency here.
        if exc.name != _CONFIG_MODULES[backend]:
            raise
        return (
            Path(__file__).resolve().parents[1]
            / "cameras"
            / backend
            / "calibration_data"
            / "multi_camera_calib_latest.npz"
        )
    return Path(cfg.CALIBRATION_FILE)


def _load_calibration(backend: str) -> dict:
    """Load all calibration arrays from the .npz file for *backend*."""
    path = _calibration_path(backend)
    if not path.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {path}\n"
            f"Run  handtracker calibrate --backend {backend}  first."
        )
    from mavis_track._files import load_npz
    data = load_npz(path)
    n = int(data["num_cameras"])
    img_size = tuple(data["img_size"])

    cameras = []
    for i in range(n):
        K = data[f"camera_matrix_{i}"]
        R = data[f"R_{i}"]
        T = data[f"T_{i}"]
        dist = data[f"dist_coeffs_{i}"]
        P = projection_matrix(K, R, T)
        C = camera_center(R, T)
        cameras.append(
            {
                "K": K,
                "R": R,
                "T": T,
                "dist": dist,
                "P": P,
                "C": C,
                "img_size": img_size,
            }
        )
    return {"cameras": cameras, "img_size": img_size, "n": n, "path": path}


# ---------------------------------------------------------------------------
# Volume definition
# ---------------------------------------------------------------------------


def _auto_volume(cameras: list[dict], margin: float) -> tuple[np.ndarray, np.ndarray]:
    """Bounding box from camera positions, pushed forward along mean gaze and
    expanded by *margin*.  Returns (min_corner, max_corner)."""
    centres = np.array([c["C"] for c in cameras])

    # Mean gaze direction (third row of R is the camera z-axis in world)
    gaze_dirs = np.array([c["R"][2, :] for c in cameras])
    mean_gaze = gaze_dirs.mean(axis=0)
    mean_gaze /= np.linalg.norm(mean_gaze) + 1e-12

    # Place the analysis volume in front of the cameras
    centroid = centres.mean(axis=0)
    # Estimate working distance from camera spread
    spread = np.ptp(centres, axis=0).max()
    working_dist = max(spread * 0.5, 0.3)  # at least 300mm equivalent

    volume_centre = centroid + mean_gaze * working_dist
    half_extent = margin
    lo = volume_centre - half_extent
    hi = volume_centre + half_extent
    return lo, hi


def _make_grid(
    lo: np.ndarray,
    hi: np.ndarray,
    spacing: float,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Create a regular 3D grid of points.

    Returns (points (N,3), grid_shape (nx, ny, nz)).
    """
    axes = [np.arange(lo[i], hi[i] + spacing * 0.5, spacing) for i in range(3)]
    grid = np.meshgrid(*axes, indexing="ij")
    shape = grid[0].shape
    points = np.column_stack([g.ravel() for g in grid])
    return points, shape


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _compute_all(
    cameras: list[dict],
    grid_points: np.ndarray,
    grid_shape: tuple[int, int, int],
    near: float,
    far: float,
    sigma_px: float,
    min_cameras: int,
) -> dict:
    """Run every metric on the grid.  Returns a results dict."""
    n_points = len(grid_points)
    # --- 1. Coverage ---
    coverage = coverage_map(grid_points, cameras, near, far)

    # Pre-compute per-camera visibility for reuse
    from handtrack.processing._camera_geometry import visibility_mask as _vis

    vis_masks = []
    for cam in cameras:
        vis_masks.append(
            _vis(grid_points, cam["K"], cam["R"], cam["T"], cam["img_size"], near, far)
        )
    vis_masks = np.array(vis_masks)  # (n_cams, n_points)

    # Indices where we have enough cameras for triangulation
    valid = coverage >= min_cameras
    valid_indices = np.where(valid)[0]

    # --- 2-4. Uncertainty, condition number, convergence angles ---
    rms_unc = np.full(n_points, np.nan)
    worst_unc = np.full(n_points, np.nan)
    aniso = np.full(n_points, np.nan)
    cond_num = np.full(n_points, np.nan)
    min_angle = np.full(n_points, np.nan)
    mean_angle = np.full(n_points, np.nan)

    all_P = [cam["P"] for cam in cameras]
    all_C = [cam["C"] for cam in cameras]

    for idx in valid_indices:
        pt = grid_points[idx]
        visible_cams = np.where(vis_masks[:, idx])[0]
        if len(visible_cams) < min_cameras:
            continue

        # Projection matrices for visible cameras
        Ps = [all_P[c] for c in visible_cams]
        Cs = [all_C[c] for c in visible_cams]

        # DLT matrix
        A = build_dlt_matrix(pt, Ps)
        cond_num[idx] = condition_number(A)

        # Covariance
        cov = triangulation_covariance(A, sigma_px)
        rms_unc[idx] = rms_uncertainty(cov)
        worst_unc[idx] = worst_axis_uncertainty(cov)
        aniso[idx] = anisotropy_ratio(cov)

        # Ray angles
        angles = ray_intersection_angles(pt, Cs)
        if len(angles) > 0:
            min_angle[idx] = float(np.min(angles))
            mean_angle[idx] = float(np.mean(angles))

    return {
        "grid_points": grid_points,
        "grid_shape": grid_shape,
        "coverage": coverage,
        "valid_mask": valid,
        "rms_uncertainty": rms_unc,
        "worst_uncertainty": worst_unc,
        "anisotropy": aniso,
        "condition_number": cond_num,
        "min_angle": min_angle,
        "mean_angle": mean_angle,
    }


# ---------------------------------------------------------------------------
# Pairwise & per-camera metrics (not grid-based)
# ---------------------------------------------------------------------------


def _pairwise_metrics(
    cameras: list[dict],
    sigma_px: float,
    sample_depths: list[float],
) -> dict:
    """Baseline distances, depth precision, and epipolar consistency."""
    centres = [cam["C"] for cam in cameras]
    baselines = pairwise_baselines(centres)

    depth_prec = {}
    for (i, j), B in baselines.items():
        fx_i = cameras[i]["K"][0, 0]
        fx_j = cameras[j]["K"][0, 0]
        f_avg = (fx_i + fx_j) / 2.0
        depth_prec[(i, j)] = {
            z: depth_uncertainty(f_avg, B, z, sigma_px) for z in sample_depths
        }

    epipolar = {}
    for i, j in combinations(range(len(cameras)), 2):
        R_rel = cameras[j]["R"] @ cameras[i]["R"].T
        T_rel = cameras[j]["T"] - cameras[j]["R"] @ cameras[i]["R"].T @ cameras[i]["T"]
        E = essential_matrix(R_rel, T_rel)
        F = fundamental_matrix(cameras[i]["K"], cameras[j]["K"], E)
        epipolar[(i, j)] = epipolar_consistency(F)

    return {
        "baselines": baselines,
        "depth_precision": depth_prec,
        "epipolar_residuals": epipolar,
    }


def _per_camera_intrinsics(cameras: list[dict]) -> list[dict]:
    """Intrinsic property reports per camera."""
    return [
        intrinsic_properties(cam["K"], cam["dist"], cam["img_size"]) for cam in cameras
    ]


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------


def _print_report(
    calib: dict,
    results: dict,
    pairwise: dict,
    intrinsics_list: list[dict],
    grid_spacing: float,
    near: float,
    far: float,
    sigma_px: float,
    min_cameras: int,
    sample_depths: list[float],
    unit_scale: float,
    unit_label: str,
) -> None:
    """Print a structured text report to stdout."""
    n_cams = calib["n"]
    grid_shape = results["grid_shape"]
    coverage = results["coverage"]
    valid = results["valid_mask"]
    n_total = len(coverage)
    n_valid = int(valid.sum())

    # Volume
    voxel_vol = (grid_spacing * unit_scale) ** 3  # in mm^3
    control_vol_mm3 = n_valid * voxel_vol
    control_vol_cm3 = control_vol_mm3 / 1e3  # mm^3 -> cm^3

    print()
    print("=" * 70)
    print("  CAMERA PLACEMENT EVALUATION REPORT")
    print("=" * 70)

    # --- Parameters ---
    print(f"\n  Calibration file : {calib['path']}")
    print(f"  Cameras          : {n_cams}")
    print(f"  Image size       : {calib['img_size'][0]} x {calib['img_size'][1]} px")
    print(f"  Grid spacing     : {grid_spacing * unit_scale:.1f} {unit_label}")
    print(f"  Grid dimensions  : {grid_shape[0]} x {grid_shape[1]} x {grid_shape[2]}")
    print(
        f"  Near / far clip  : {near * unit_scale:.0f} / {far * unit_scale:.0f} {unit_label}"
    )
    print(f"  Pixel noise (σ)  : {sigma_px:.2f} px")
    print(f"  Min cameras      : {min_cameras}")

    # --- 1. Control volume ---
    print(f"\n{'─' * 70}")
    print("  1. CONTROL VOLUME (N-camera coverage)")
    print(f"{'─' * 70}")
    for k in range(n_cams + 1):
        count = int((coverage == k).sum())
        pct = 100.0 * count / n_total if n_total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"    {k} cameras : {count:>7} voxels  ({pct:5.1f}%)  {bar}")
    print(
        f"\n    Control volume (>={min_cameras} cameras) : {control_vol_cm3:,.1f} cm³"
    )
    print(f"    Voxels in control volume          : {n_valid:,} / {n_total:,}")

    # --- 2. Triangulation uncertainty ---
    print(f"\n{'─' * 70}")
    print("  2. TRIANGULATION UNCERTAINTY  [Olague & Mohr 2002]")
    print(f"{'─' * 70}")
    rms_v = results["rms_uncertainty"][valid]
    worst_v = results["worst_uncertainty"][valid]
    aniso_v = results["anisotropy"][valid]
    if len(rms_v) > 0:
        rms_v_mm = rms_v * unit_scale
        worst_v_mm = worst_v * unit_scale
        print("    RMS positional uncertainty:")
        print(f"      mean   : {np.nanmean(rms_v_mm):.4f} {unit_label}")
        print(f"      median : {np.nanmedian(rms_v_mm):.4f} {unit_label}")
        print(f"      95th ‰ : {np.nanpercentile(rms_v_mm, 95):.4f} {unit_label}")
        print(f"      max    : {np.nanmax(rms_v_mm):.4f} {unit_label}")
        print("    Worst-axis uncertainty:")
        print(f"      mean   : {np.nanmean(worst_v_mm):.4f} {unit_label}")
        print(f"      max    : {np.nanmax(worst_v_mm):.4f} {unit_label}")
        print("    Anisotropy ratio (worst/best axis):")
        print(f"      mean   : {np.nanmean(aniso_v):.2f}")
        print(f"      max    : {np.nanmax(aniso_v):.2f}")
    else:
        print("    No valid voxels for uncertainty analysis.")

    # --- 3. Ray intersection angles ---
    print(f"\n{'─' * 70}")
    print("  3. RAY CONVERGENCE ANGLES  [ACM VRST: 40-140° acceptable]")
    print(f"{'─' * 70}")
    min_a = results["min_angle"][valid]
    mean_a = results["mean_angle"][valid]
    if len(min_a) > 0:
        in_range = (min_a >= _MIN_CONVERGENCE_DEG) & (min_a <= _MAX_CONVERGENCE_DEG)
        pct_good = 100.0 * np.nansum(in_range) / len(min_a)
        print("    Min convergence angle across volume:")
        print(f"      mean   : {np.nanmean(min_a):.1f}°")
        print(f"      median : {np.nanmedian(min_a):.1f}°")
        print(f"      min    : {np.nanmin(min_a):.1f}°")
        print("    Mean convergence angle:")
        print(f"      mean   : {np.nanmean(mean_a):.1f}°")
        print(f"    Voxels in acceptable range (40-140°) : {pct_good:.1f}%")
    else:
        print("    No valid voxels for angle analysis.")

    # --- 4. DLT condition number ---
    print(f"\n{'─' * 70}")
    print("  4. DLT CONDITION NUMBER  [Olague & Mohr 2002]")
    print(f"{'─' * 70}")
    cn = results["condition_number"][valid]
    if len(cn) > 0:
        well_cond = cn < 100
        print("    Condition number across control volume:")
        print(f"      mean   : {np.nanmean(cn):.1f}")
        print(f"      median : {np.nanmedian(cn):.1f}")
        print(f"      95th ‰ : {np.nanpercentile(cn, 95):.1f}")
        print(f"      max    : {np.nanmax(cn):.1f}")
        print(
            f"    Well-conditioned (cond < 100) : {100.0 * np.nansum(well_cond) / len(cn):.1f}%"
        )
    else:
        print("    No valid voxels for condition number analysis.")

    # --- 5. Baselines & depth precision ---
    print(f"\n{'─' * 70}")
    print("  5. BASELINE-TO-DEPTH RATIO & DEPTH PRECISION  [Szeliski 2022]")
    print(f"{'─' * 70}")
    baselines = pairwise["baselines"]
    depth_prec = pairwise["depth_precision"]
    for (i, j), B in sorted(baselines.items()):
        print(f"    Cameras ({i}, {j}):  baseline = {B * unit_scale:.1f} {unit_label}")
        for z in sample_depths:
            sigma_z = depth_prec[(i, j)][z]
            bdr = B / z if z > 0 else 0
            print(
                f"      Z={z * unit_scale:.0f}{unit_label}:  σ_Z = {sigma_z * unit_scale:.3f} {unit_label}   B/Z = {bdr:.3f}"
            )

    # --- 6-7. Per-camera intrinsics ---
    print(f"\n{'─' * 70}")
    print("  6-7. PER-CAMERA INTRINSIC PROPERTIES")
    print(f"{'─' * 70}")
    for idx, props in enumerate(intrinsics_list):
        gsd_vals = [
            ground_sampling_distance(props["fx"], z) * unit_scale for z in sample_depths
        ]
        print(f"    Camera {idx}:")
        print(f"      Focal length     : fx={props['fx']:.1f}  fy={props['fy']:.1f} px")
        print(
            f"      Principal point  : ({props['cx']:.1f}, {props['cy']:.1f}) px"
            f"  offset={props['principal_point_offset_px']:.1f} px"
        )
        print(
            f"      FOV              : {props['fov_h_deg']:.1f}° x {props['fov_v_deg']:.1f}°"
            f"  (diag {props['fov_diag_deg']:.1f}°)"
        )
        print(
            f"      Focal ratio      : {props['focal_ratio']:.4f}  (1.0 = square pixels)"
        )
        print(f"      Distortion |k|   : {props['distortion_magnitude']:.4f}")
        gsd_strs = [f"{g:.3f}" for g in gsd_vals]
        depth_strs = [f"{z * unit_scale:.0f}" for z in sample_depths]
        print(
            f"      GSD ({unit_label}/px)     : "
            + "  ".join(f"@{d}{unit_label}={g}" for d, g in zip(depth_strs, gsd_strs))
        )

    # --- 8. Epipolar consistency ---
    print(f"\n{'─' * 70}")
    print("  8. EPIPOLAR CONSISTENCY  [Zhang 1998]")
    print(f"{'─' * 70}")
    for (i, j), res in sorted(pairwise["epipolar_residuals"].items()):
        status = "OK" if res < 1e-6 else "WARN" if res < 1e-3 else "FAIL"
        print(f"    Cameras ({i}, {j}):  |det(F)| = {res:.2e}  [{status}]")

    # --- 9. VICON comparison summary ---
    print(f"\n{'─' * 70}")
    print("  9. SUMMARY — VICON COMPARISON TABLE")
    print(f"{'─' * 70}")
    rms_mean = np.nanmean(rms_v * unit_scale) if len(rms_v) > 0 else float("nan")
    rms_worst = np.nanmax(worst_v * unit_scale) if len(worst_v) > 0 else float("nan")
    med_cond = np.nanmedian(cn) if len(cn) > 0 else float("nan")
    min_ang = np.nanmin(min_a) if len(min_a) > 0 else float("nan")
    mean_baseline = np.mean(list(baselines.values())) * unit_scale

    fmt = "    {:<40s} {:>15s}   {:>20s}"
    print(fmt.format("Metric", "This System", "VICON Vero (ref)"))
    print(fmt.format("─" * 40, "─" * 15, "─" * 20))
    print(
        fmt.format(
            "Control volume", f"{control_vol_cm3:,.1f} cm³", "~64M cm³ (4×4×4 m)"
        )
    )
    print(
        fmt.format(
            "Mean positional accuracy", f"{rms_mean:.3f} {unit_label}", "0.08-0.26 mm"
        )
    )
    print(fmt.format("Worst-case accuracy", f"{rms_worst:.3f} {unit_label}", "~0.3 mm"))
    print(fmt.format("Median condition number", f"{med_cond:.1f}", "N/A"))
    print(fmt.format("Min convergence angle", f"{min_ang:.1f}°", "N/A"))
    print(fmt.format("Mean baseline", f"{mean_baseline:.1f} {unit_label}", "varies"))
    print(fmt.format("Number of cameras", str(n_cams), "8-12 typical"))
    print(
        fmt.format(
            "Image resolution",
            f"{calib['img_size'][0]}×{calib['img_size'][1]}",
            "16MP (Vero)",
        )
    )
    print()
    print("  References:")
    print("    Merriaux et al., Sensors 17(7):1591, 2017")
    print("    Windolf et al., J Biomechanics 41(16):2776-2780, 2008")
    print("    Topley & Richards, PMC systematic review, 2022")
    print()


# ---------------------------------------------------------------------------
# Visualisation (optional matplotlib)
# ---------------------------------------------------------------------------


def _save_visualisations(
    calib: dict,
    results: dict,
    pairwise: dict,
    output_dir: Path,
    unit_scale: float,
    unit_label: str,
    near: float,
    far: float,
) -> list[Path]:
    """Generate and save matplotlib figures.  Returns list of saved paths."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError:
        print("  [warn] matplotlib not installed — skipping visualisations.")
        print("         Install with:  pip install matplotlib")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    grid = results["grid_points"] * unit_scale
    shape = results["grid_shape"]
    coverage = results["coverage"]

    # Reshape to 3D for slicing
    cov_3d = coverage.reshape(shape)
    xs = np.unique(grid[:, 0])
    ys = np.unique(grid[:, 1])
    zs = np.unique(grid[:, 2])

    # --- Camera layout ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    centres = np.array([c["C"] for c in calib["cameras"]]) * unit_scale
    for idx, c in enumerate(centres):
        ax.scatter(*c, s=100, marker="^", zorder=5)
        ax.text(c[0], c[1], c[2], f"  cam{idx}", fontsize=9)
        # Draw gaze direction
        R = calib["cameras"][idx]["R"]
        gaze = R[2, :] * (far * unit_scale * 0.3)
        ax.quiver(
            c[0],
            c[1],
            c[2],
            gaze[0],
            gaze[1],
            gaze[2],
            color="gray",
            alpha=0.5,
            arrow_length_ratio=0.1,
        )
    ax.set_xlabel(f"X ({unit_label})")
    ax.set_ylabel(f"Y ({unit_label})")
    ax.set_zlabel(f"Z ({unit_label})")
    ax.set_title("Camera Layout")
    p = output_dir / "camera_layout.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # --- Coverage slices (XY, XZ, YZ at centre) ---
    mid = [s // 2 for s in shape]
    slice_configs = [
        ("XY", cov_3d[:, :, mid[2]], xs, ys, "X", "Y", f"Z={zs[mid[2]]:.0f}"),
        ("XZ", cov_3d[:, mid[1], :], xs, zs, "X", "Z", f"Y={ys[mid[1]]:.0f}"),
        ("YZ", cov_3d[mid[0], :, :], ys, zs, "Y", "Z", f"X={xs[mid[0]]:.0f}"),
    ]
    for name, slc, ax_x, ax_y, lbl_x, lbl_y, title_extra in slice_configs:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(
            slc.T,
            origin="lower",
            aspect="auto",
            extent=[ax_x[0], ax_x[-1], ax_y[0], ax_y[-1]],
            cmap="YlOrRd",
            vmin=0,
            vmax=len(calib["cameras"]),
        )
        ax.set_xlabel(f"{lbl_x} ({unit_label})")
        ax.set_ylabel(f"{lbl_y} ({unit_label})")
        ax.set_title(f"Coverage ({name} slice, {title_extra} {unit_label})")
        plt.colorbar(im, ax=ax, label="# cameras")
        p = output_dir / f"coverage_{name.lower()}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    # --- Uncertainty slices (XY at centre) ---
    rms_3d = results["rms_uncertainty"].reshape(shape) * unit_scale
    slc = rms_3d[:, :, mid[2]]
    valid_vals = slc[np.isfinite(slc)]
    if len(valid_vals) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(
            slc.T,
            origin="lower",
            aspect="auto",
            extent=[xs[0], xs[-1], ys[0], ys[-1]],
            cmap="viridis",
            norm=Normalize(vmin=0, vmax=np.nanpercentile(valid_vals, 95)),
        )
        ax.set_xlabel(f"X ({unit_label})")
        ax.set_ylabel(f"Y ({unit_label})")
        ax.set_title(f"RMS Uncertainty (XY slice, Z={zs[mid[2]]:.0f} {unit_label})")
        plt.colorbar(im, ax=ax, label=f"σ ({unit_label})")
        p = output_dir / "uncertainty_xy.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    # --- Condition number slice ---
    cn_3d = results["condition_number"].reshape(shape)
    slc = cn_3d[:, :, mid[2]]
    valid_vals = slc[np.isfinite(slc)]
    if len(valid_vals) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(
            slc.T,
            origin="lower",
            aspect="auto",
            extent=[xs[0], xs[-1], ys[0], ys[-1]],
            cmap="inferno",
            norm=Normalize(vmin=0, vmax=min(np.nanpercentile(valid_vals, 95), 200)),
        )
        ax.set_xlabel(f"X ({unit_label})")
        ax.set_ylabel(f"Y ({unit_label})")
        ax.set_title(
            f"DLT Condition Number (XY slice, Z={zs[mid[2]]:.0f} {unit_label})"
        )
        plt.colorbar(im, ax=ax, label="cond(A)")
        p = output_dir / "condition_number_xy.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    # --- Min convergence angle slice ---
    ang_3d = results["min_angle"].reshape(shape)
    slc = ang_3d[:, :, mid[2]]
    valid_vals = slc[np.isfinite(slc)]
    if len(valid_vals) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(
            slc.T,
            origin="lower",
            aspect="auto",
            extent=[xs[0], xs[-1], ys[0], ys[-1]],
            cmap="RdYlGn",
            vmin=0,
            vmax=180,
        )
        ax.set_xlabel(f"X ({unit_label})")
        ax.set_ylabel(f"Y ({unit_label})")
        ax.set_title(f"Min Convergence Angle (XY, Z={zs[mid[2]]:.0f} {unit_label})")
        plt.colorbar(im, ax=ax, label="degrees")
        # Mark acceptable range
        ax.text(
            0.02,
            0.98,
            "Acceptable: 40°-140°",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        p = output_dir / "convergence_angle_xy.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    # --- Baseline bar chart ---
    baselines = pairwise["baselines"]
    if baselines:
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = [f"({i},{j})" for i, j in sorted(baselines.keys())]
        vals = [baselines[k] * unit_scale for k in sorted(baselines.keys())]
        ax.barh(labels, vals, color="steelblue")
        ax.set_xlabel(f"Baseline ({unit_label})")
        ax.set_title("Pairwise Camera Baselines")
        p = output_dir / "baselines.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_evaluation(args: argparse.Namespace) -> int:
    """Compute metrics, print the report, and optionally save figures."""
    # Determine unit scaling — the ChArUco board is defined in meters,
    # so calibration translations are in meters.  Display in mm.
    unit_scale = 1000.0  # meters -> mm
    unit_label = "mm"

    # Load calibration
    calib = _load_calibration(args.backend)
    cameras = calib["cameras"]

    # Define volume
    lo, hi = _auto_volume(cameras, margin=args.margin)
    grid_points, grid_shape = _make_grid(lo, hi, args.grid_spacing)

    print(
        f"[evaluate-placement] Analysing {len(grid_points):,} voxels "
        f"({grid_shape[0]}×{grid_shape[1]}×{grid_shape[2]}) ..."
    )

    # Compute grid-based metrics
    results = _compute_all(
        cameras,
        grid_points,
        grid_shape,
        near=args.near,
        far=args.far,
        sigma_px=args.sigma_px,
        min_cameras=args.min_cameras,
    )

    # Sample depths for baseline/depth analysis
    sample_depths = [0.200, 0.400, 0.600, 0.800]

    # Pairwise metrics
    pairwise = _pairwise_metrics(cameras, args.sigma_px, sample_depths)

    # Per-camera intrinsics
    intrinsics_list = _per_camera_intrinsics(cameras)

    # Print report
    _print_report(
        calib,
        results,
        pairwise,
        intrinsics_list,
        grid_spacing=args.grid_spacing,
        near=args.near,
        far=args.far,
        sigma_px=args.sigma_px,
        min_cameras=args.min_cameras,
        sample_depths=sample_depths,
        unit_scale=unit_scale,
        unit_label=unit_label,
    )

    # Visualisations
    if args.output_dir:
        out = Path(args.output_dir)
        print(f"[evaluate-placement] Saving visualisations to {out}/ ...")
        saved = _save_visualisations(
            calib,
            results,
            pairwise,
            out,
            unit_scale=unit_scale,
            unit_label=unit_label,
            near=args.near,
            far=args.far,
        )
        for p in saved:
            print(f"  ✓ {p}")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate camera placement quality from calibration data.",
    )
    parser.add_argument(
        "--backend",
        choices=("optitrack", "webcam"),
        required=True,
        help="camera backend whose calibration to analyse",
    )
    parser.add_argument(
        "--grid-spacing",
        type=float,
        default=0.010,
        help="grid spacing in calibration units (default 0.010 = 10 mm if board in meters)",
    )
    parser.add_argument(
        "--near",
        type=float,
        default=0.100,
        help="near clipping distance in calibration units (default 0.100)",
    )
    parser.add_argument(
        "--far",
        type=float,
        default=1.000,
        help="far clipping distance in calibration units (default 1.000)",
    )
    parser.add_argument(
        "--sigma-px",
        type=float,
        default=3.0,
        help="assumed pixel noise standard deviation (default 3.0, conservative for MediaPipe)",
    )
    parser.add_argument(
        "--min-cameras",
        type=int,
        default=2,
        help="minimum cameras for a voxel to be 'in' the control volume (default 2)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.250,
        help="half-extent of the analysis volume around the centroid (default 0.250)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="directory for the text report and visualisation figures",
    )
    args = parser.parse_args(argv)

    if not args.output_dir:
        return _run_evaluation(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "placement_evaluation.txt"
    with report_path.open("w", encoding="utf-8") as report_file:
        with contextlib.redirect_stdout(_Tee(sys.stdout, report_file)):
            result = _run_evaluation(args)
            print(f"[evaluate-placement] Report saved to {report_path}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
