"""Slab extraction — slice a 3D point cloud along a principal axis."""
from __future__ import annotations

import numpy as np

# Axis index map
_AXIS = {"X": 0, "Y": 1, "Z": 2}
# The two projected axes for each slice normal
_PROJ = {
    "X": (1, 2, "Y", "Z"),   # slice normal X → project onto Y, Z
    "Y": (0, 2, "X", "Z"),   # slice normal Y → project onto X, Z
    "Z": (0, 1, "X", "Y"),   # slice normal Z → project onto X, Y
}


def extract_slab(
    pts: np.ndarray,
    axis: str,
    position_m: float,
    thickness_m: float,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Extract points within ±thickness/2 of a plane normal to `axis`.

    Args:
        pts: (N, 3) float32/64 point cloud
        axis: 'X', 'Y', or 'Z'
        position_m: slice position along the chosen axis
        thickness_m: full slab thickness in metres

    Returns:
        pts_slab:  (M, 3) points inside the slab
        uv:        (M, 2) 2D projected coordinates
        u_label:   label for the first projected axis (e.g. "Y")
        v_label:   label for the second projected axis (e.g. "Z")
    """
    axis = axis.upper()
    ax = _AXIS[axis]
    u_idx, v_idx, u_label, v_label = _PROJ[axis]

    half = thickness_m / 2.0
    coord = pts[:, ax]
    mask = (coord >= position_m - half) & (coord <= position_m + half)
    pts_slab = pts[mask]
    uv = pts_slab[:, [u_idx, v_idx]]
    return pts_slab, uv, u_label, v_label


def plan_projection(
    pts: np.ndarray,
    n_max: int = 75_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return X, Y coordinates for a top-down plan view (sub-sampled).

    Returns:
        x, y  — 1D arrays for Plotly
    """
    if len(pts) > n_max:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), n_max, replace=False, shuffle=False)
        sub = pts[idx]
    else:
        sub = pts
    return sub[:, 0], sub[:, 1]


def cloud_bounds(pts: np.ndarray) -> dict:
    """Return dict with xmin/xmax/ymin/ymax/zmin/zmax."""
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    return dict(
        xmin=float(mn[0]), xmax=float(mx[0]),
        ymin=float(mn[1]), ymax=float(mx[1]),
        zmin=float(mn[2]), zmax=float(mx[2]),
    )
