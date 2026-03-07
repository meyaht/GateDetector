"""Gate detection algorithms — pure numpy + scipy, no external CV libraries.

Detection pipeline
------------------
1. Rasterize 2D point slice to binary occupancy grid (30 mm cells by default).
2. Compute row-sum and column-sum projections to find dense linear spans (steel beams).
3. Pair horizontal beam spans (top + bottom) with overlapping vertical beam spans
   (left + right) to form rectangular gate candidates.
4. Within each candidate rectangle, count circular pipe cross-sections using
   connected-component analysis and circularity filtering.
5. Score by: side completeness, pipe count, opening area.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
from scipy.ndimage import binary_dilation, label, find_objects


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Gate:
    gate_id:          str
    axis:             str            # 'X' (slice normal X, projected onto YZ) or 'Y'
    position_m:       float          # slice centre position
    thickness_m:      float          # slab thickness used for detection
    bbox_2d:          list           # [u0, v0, u1, v1] in projected coords (m)
    bbox_3d:          list           # [xmin, ymin, zmin, xmax, ymax, zmax]
    pipe_count:       int   = 0
    pipe_locs_2d:     list  = field(default_factory=list)   # [[u, v], ...]
    opening_area_m2:  float = 0.0
    confidence:       float = 0.0    # 0–1
    source:           str   = "auto" # 'auto' | 'manual'
    label:            str   = ""
    notes:            str   = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Gate":
        return cls(**d)


# ---------------------------------------------------------------------------
# Rasterisation helpers
# ---------------------------------------------------------------------------

def _rasterize(
    uv: np.ndarray,
    cell_m: float,
) -> tuple[np.ndarray, float, float, float, float]:
    """Convert 2D points to a binary occupancy grid.

    Returns:
        grid:           bool 2D array [rows=V, cols=U]
        u_min, u_max:   real-world extents of the U axis
        v_min, v_max:   real-world extents of the V axis
    """
    if len(uv) == 0:
        return np.zeros((1, 1), dtype=bool), 0, 1, 0, 1

    u_min, v_min = uv.min(axis=0)
    u_max, v_max = uv.max(axis=0)

    # Add a small border
    border = cell_m * 2
    u0 = u_min - border
    v0 = v_min - border

    n_u = max(1, int(np.ceil((u_max - u_min + 2 * border) / cell_m)))
    n_v = max(1, int(np.ceil((v_max - v_min + 2 * border) / cell_m)))

    ui = np.clip(((uv[:, 0] - u0) / cell_m).astype(int), 0, n_u - 1)
    vi = np.clip(((uv[:, 1] - v0) / cell_m).astype(int), 0, n_v - 1)

    grid = np.zeros((n_v, n_u), dtype=bool)
    grid[vi, ui] = True

    # Dilate to connect nearby points (handles slight gaps in beam surfaces)
    struct = np.ones((3, 3), dtype=bool)
    grid = binary_dilation(grid, structure=struct, iterations=2)

    return grid, u0, u0 + n_u * cell_m, v0, v0 + n_v * cell_m


def _grid_to_world(idx: int, origin: float, cell_m: float) -> float:
    return origin + (idx + 0.5) * cell_m


# ---------------------------------------------------------------------------
# Linear feature (beam) detection
# ---------------------------------------------------------------------------

def _find_dense_spans(
    projection: np.ndarray,
    grid_size: int,
    min_fill_fraction: float = 0.20,
    min_span_cells: int = 3,
    merge_gap_cells: int = 4,
) -> list[tuple[int, int]]:
    """Given a 1D occupancy projection (sum of a 2D grid along one axis),
    find contiguous spans where fill fraction exceeds the threshold.

    projection: 1D array — number of occupied cells per row or column
    grid_size:  the dimension perpendicular to the projection axis (to compute fill)
    Returns list of (start, end) index pairs (inclusive).
    """
    norm = projection / max(grid_size, 1)
    dense = norm >= min_fill_fraction

    spans = []
    in_span = False
    start = 0
    for i, d in enumerate(dense):
        if d and not in_span:
            start = i
            in_span = True
        elif not d and in_span:
            if i - start >= min_span_cells:
                spans.append((start, i - 1))
            in_span = False
    if in_span and len(dense) - start >= min_span_cells:
        spans.append((start, len(dense) - 1))

    # Merge nearby spans
    if not spans:
        return []
    merged = [spans[0]]
    for s, e in spans[1:]:
        if s - merged[-1][1] <= merge_gap_cells:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


# ---------------------------------------------------------------------------
# Gate rectangle assembly
# ---------------------------------------------------------------------------

def _spans_overlap(a: tuple, b: tuple, min_overlap_cells: int = 5) -> bool:
    """Do two 1D spans overlap by at least min_overlap_cells?"""
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    return (hi - lo) >= min_overlap_cells


def _find_gate_rects(
    h_spans: list[tuple[int, int]],    # horizontal beam row spans (V direction)
    v_spans: list[tuple[int, int]],    # vertical beam col spans (U direction)
    grid: np.ndarray,
    cell_m: float,
    u_origin: float,
    v_origin: float,
    min_gate_w_cells: int,
    max_gate_w_cells: int,
    min_gate_h_cells: int,
    max_gate_h_cells: int,
) -> list[dict]:
    """Assemble gate rectangles from detected beam spans.

    Strategy: every pair of horizontal beams (top + bottom) combined with
    every pair of vertical beams (left + right) that geometrically form
    a plausible gate rectangle.
    """
    n_rows, n_cols = grid.shape
    candidates = []

    # Pair horizontal beams (top row, bottom row)
    for i, h1 in enumerate(h_spans):
        for h2 in h_spans[i + 1:]:
            # h1 is lower row index (lower V = lower Z typically)
            bot, top = h1, h2
            gate_h = top[0] - bot[1]   # gap between the two beams
            if not (min_gate_h_cells <= gate_h <= max_gate_h_cells):
                continue

            # Pair vertical beams that span the gate height
            for j, v1 in enumerate(v_spans):
                for v2 in v_spans[j + 1:]:
                    left, right = v1, v2
                    gate_w = right[0] - left[1]
                    if not (min_gate_w_cells <= gate_w <= max_gate_w_cells):
                        continue

                    # Check that the vertical beams span the gate height zone
                    gate_v_range = (bot[0], top[1])
                    if not (_spans_overlap(left, gate_v_range, min_overlap_cells=3) and
                            _spans_overlap(right, gate_v_range, min_overlap_cells=3)):
                        continue

                    # Opening bbox in grid coords
                    u0_c = left[1] + 1
                    u1_c = right[0] - 1
                    v0_c = bot[1] + 1
                    v1_c = top[0] - 1

                    # Convert to world coords (wu/wv prefix avoids shadowing loop vars v1/u0)
                    wu0 = _grid_to_world(u0_c, u_origin, cell_m)
                    wu1 = _grid_to_world(u1_c, u_origin, cell_m)
                    wv0 = _grid_to_world(v0_c, v_origin, cell_m)
                    wv1 = _grid_to_world(v1_c, v_origin, cell_m)

                    confidence = 1.0

                    candidates.append(dict(
                        bbox_2d=[wu0, wv0, wu1, wv1],
                        opening_w=wu1 - wu0,
                        opening_h=wv1 - wv0,
                        opening_area_m2=(wu1 - wu0) * (wv1 - wv0),
                        confidence=confidence,
                        u0_c=u0_c, u1_c=u1_c, v0_c=v0_c, v1_c=v1_c,
                    ))

    return candidates


# ---------------------------------------------------------------------------
# Pipe detection inside a gate
# ---------------------------------------------------------------------------

def _count_pipes_in_gate(
    uv: np.ndarray,
    bbox_2d: list,
    cell_m: float = 0.020,
    min_pipe_r_m: float = 0.025,   # NPS 0.5 = 21mm OD
    max_pipe_r_m: float = 0.400,   # NPS 16 = 406mm OD
    min_circularity: float = 0.35,
) -> tuple[int, list]:
    """Count pipe cross-sections inside a gate using connected-component circularity.

    Returns:
        pipe_count: estimated number of pipes
        locs: [[u_centre, v_centre], ...] of each pipe
    """
    u0, v0, u1, v1 = bbox_2d
    mask = (
        (uv[:, 0] >= u0) & (uv[:, 0] <= u1) &
        (uv[:, 1] >= v0) & (uv[:, 1] <= v1)
    )
    sub = uv[mask]
    if len(sub) < 5:
        return 0, []

    # Fine raster inside the gate
    border = cell_m * 3
    uo = u0 - border
    vo = v0 - border
    n_u = max(1, int(np.ceil((u1 - u0 + 2 * border) / cell_m)))
    n_v = max(1, int(np.ceil((v1 - v0 + 2 * border) / cell_m)))

    ui = np.clip(((sub[:, 0] - uo) / cell_m).astype(int), 0, n_u - 1)
    vi = np.clip(((sub[:, 1] - vo) / cell_m).astype(int), 0, n_v - 1)

    grid = np.zeros((n_v, n_u), dtype=bool)
    grid[vi, ui] = True

    struct = np.ones((3, 3), dtype=bool)
    dilated = binary_dilation(grid, structure=struct, iterations=1)
    labeled, n_comp = label(dilated)
    slices = find_objects(labeled)

    min_area = np.pi * (min_pipe_r_m ** 2) / (cell_m ** 2)
    max_area = np.pi * (max_pipe_r_m ** 2) / (cell_m ** 2)

    pipe_count = 0
    locs = []

    for i, sl in enumerate(slices):
        comp = (labeled == i + 1)
        area_cells = comp.sum()
        if not (min_area <= area_cells <= max_area):
            continue

        # Bounding box of component
        vr = sl[0]
        ur = sl[1]
        h = vr.stop - vr.start
        w = ur.stop - ur.start
        if h == 0 or w == 0:
            continue

        # Circularity: 4π·area / perimeter² — approximate perimeter from bbox
        aspect = min(h, w) / max(h, w)
        if aspect < min_circularity:
            continue

        # Centre in world coords
        uc = uo + (ur.start + ur.stop) / 2 * cell_m
        vc = vo + (vr.start + vr.stop) / 2 * cell_m
        pipe_count += 1
        locs.append([float(uc), float(vc)])

    return pipe_count, locs


# ---------------------------------------------------------------------------
# Top-level detect function
# ---------------------------------------------------------------------------

def detect_gates(
    uv: np.ndarray,
    axis: str,
    position_m: float,
    thickness_m: float,
    pts3d: np.ndarray,
    cell_m: float = 0.030,
    min_gate_w: float = 0.3,
    max_gate_w: float = 8.0,
    min_gate_h: float = 0.3,
    max_gate_h: float = 6.0,
    min_beam_fill: float = 0.20,
) -> list[Gate]:
    """Detect pipe rack gates in a 2D cross-section slice.

    Args:
        uv:           (M, 2) 2D projected points from slab extraction
        axis:         'X' or 'Y' (slice normal axis)
        position_m:   slice position
        thickness_m:  slab thickness
        pts3d:        corresponding (M, 3) 3D points (for bbox_3d)
        cell_m:       raster cell size in metres
        min/max_gate_w/h: gate size bounds in metres

    Returns:
        list of Gate objects, sorted by confidence desc then pipe_count desc
    """
    if len(uv) < 10:
        return []

    grid, u_origin, u_end, v_origin, v_end = _rasterize(uv, cell_m)
    n_rows, n_cols = grid.shape

    min_gate_w_cells = max(1, int(min_gate_w / cell_m))
    max_gate_w_cells = int(max_gate_w / cell_m)
    min_gate_h_cells = max(1, int(min_gate_h / cell_m))
    max_gate_h_cells = int(max_gate_h / cell_m)

    # Row sums → horizontal beam spans (structures running along U, span in V)
    row_sums = grid.sum(axis=1).astype(float)
    h_spans = _find_dense_spans(row_sums, n_cols, min_fill_fraction=min_beam_fill)

    # Column sums → vertical beam spans
    col_sums = grid.sum(axis=0).astype(float)
    v_spans = _find_dense_spans(col_sums, n_rows, min_fill_fraction=min_beam_fill)

    if not h_spans or not v_spans:
        return []

    rects = _find_gate_rects(
        h_spans, v_spans, grid, cell_m,
        u_origin, v_origin,
        min_gate_w_cells, max_gate_w_cells,
        min_gate_h_cells, max_gate_h_cells,
    )

    gates = []
    seen_bboxes: list[list] = []

    for r in rects:
        bbox2 = r["bbox_2d"]

        # Deduplicate: skip if very similar bbox already captured
        is_dup = False
        for sb in seen_bboxes:
            if (abs(bbox2[0] - sb[0]) < cell_m * 5 and
                abs(bbox2[1] - sb[1]) < cell_m * 5 and
                abs(bbox2[2] - sb[2]) < cell_m * 5 and
                abs(bbox2[3] - sb[3]) < cell_m * 5):
                is_dup = True
                break
        if is_dup:
            continue
        seen_bboxes.append(bbox2)

        pipe_count, pipe_locs = _count_pipes_in_gate(uv, bbox2)

        # 3D bounding box
        axis_up = axis.upper()
        u_idx = {"X": 1, "Y": 0}[axis_up]
        v_idx = 2   # V is always Z
        ax_idx = {"X": 0, "Y": 1}[axis_up]

        b3 = [0.0] * 6
        b3[ax_idx]     = position_m - thickness_m / 2
        b3[ax_idx + 3] = position_m + thickness_m / 2
        b3[u_idx]     = bbox2[0]
        b3[u_idx + 3] = bbox2[2]
        b3[v_idx]     = bbox2[1]
        b3[v_idx + 3] = bbox2[3]

        g = Gate(
            gate_id=f"GATE_{uuid.uuid4().hex[:6].upper()}",
            axis=axis_up,
            position_m=position_m,
            thickness_m=thickness_m,
            bbox_2d=bbox2,
            bbox_3d=b3,
            pipe_count=pipe_count,
            pipe_locs_2d=pipe_locs,
            opening_area_m2=r["opening_area_m2"],
            confidence=r["confidence"],
            source="auto",
        )
        gates.append(g)

    gates.sort(key=lambda g: (-g.confidence, -g.pipe_count))
    return gates
