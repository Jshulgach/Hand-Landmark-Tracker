"""Anatomical constraint for PIP joint landmarks.

MediaPipe misplaces PIP joints (landmarks 6, 10, 14, 18) when fingers are
straight but MCP is bent.  When a finger is nearly straight the PIP must lie
on the MCP→DIP line at an anatomically proportional distance (~63%).  This
module provides a single pure function that corrects PIP positions with smooth
blending so bent fingers are unaffected.
"""

import numpy as np

# Landmark indices per finger: (MCP, PIP, DIP)
_FINGER_JOINTS = {
    "index": (5, 6, 7),
    "middle": (9, 10, 11),
    "ring": (13, 14, 15),
    "pinky": (17, 18, 19),
}

# Anatomical MCP→PIP / MCP→DIP bone-length ratios (population averages).
_PIP_RATIOS = {
    "index": 0.63,
    "middle": 0.63,
    "ring": 0.62,
    "pinky": 0.64,
}


def _bend_angle(a, b, c):
    """Bend angle at joint *b* in degrees.  0° = straight, 90° = bent."""
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0  # degenerate — treat as straight
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    geometric = np.degrees(np.arccos(cos))
    # Convert: geometric 180° (straight) → bend 0°, geometric 90° → bend 90°
    return 180.0 - geometric


def enforce_pip_constraints(
    landmarks_3d,
    straightness_threshold=15.0,
    blend_range=25.0,
):
    """Correct PIP positions when fingers are nearly straight.

    Parameters
    ----------
    landmarks_3d : ndarray (21, 3)
        Triangulated hand landmarks.
    straightness_threshold : float
        Bend angle (degrees) below which correction activates.
        0° = fully straight, 90° = bent at right angle.
    blend_range : float
        Degrees over which correction blends from 100% to 0%.
        At 0° bend → full correction.
        At straightness_threshold → correction starts tapering.
        At straightness_threshold + blend_range → no correction.

    Returns
    -------
    ndarray (21, 3)
        Corrected landmarks (copy — original is not modified).
    """
    out = landmarks_3d.copy()

    for name, (mcp_i, pip_i, dip_i) in _FINGER_JOINTS.items():
        mcp = out[mcp_i]
        pip_ = out[pip_i]
        dip = out[dip_i]

        # Skip missing landmarks
        if np.linalg.norm(mcp) < 1e-6 or np.linalg.norm(dip) < 1e-6:
            continue

        mcp_dip_len = np.linalg.norm(dip - mcp)
        if mcp_dip_len < 1e-6:
            continue

        # Ideal PIP position on the MCP→DIP line
        ratio = _PIP_RATIOS[name]
        ideal_pip = mcp + ratio * (dip - mcp)

        # --- Bone-length sanity clamp ---
        mcp_pip_len = np.linalg.norm(pip_ - mcp)
        actual_ratio = mcp_pip_len / mcp_dip_len
        if actual_ratio < 0.30 or actual_ratio > 0.80:
            out[pip_i] = ideal_pip
            continue

        # --- Straightness-based blending ---
        bend = _bend_angle(mcp, pip_, dip)
        if bend > straightness_threshold:
            continue  # finger is bent enough — don't touch

        # Linear blend: 100% correction at 0° bend, tapering to 0% at threshold
        blend_factor = 1.0 - (bend / straightness_threshold)
        blend_factor = np.clip(blend_factor, 0.0, 1.0)
        out[pip_i] = (1.0 - blend_factor) * pip_ + blend_factor * ideal_pip

    return out
