"""Gate detection algorithms — pure numpy + scipy, no external CV libraries.

Detection pipeline (v2 — local segment detection)
--------------------------------------------------
1. Rasterize 2D point slice to binary occupancy grid (30 mm cells by default).
2. Per-row gap-filling (tolerates LiDAR blind spots up to ~0.6 m).
3. Per-row run detection: find horizontal runs >= 0.3 m, anywhere in the row.
4. Group nearby beam rows into horizontal bands.
5. Same column-wise for vertical bands.
6. Pair top+bottom h_bands within gate size limits to form opening candidates.
7. Look for matching v_bands on left/right sides; assign confidence by confirmed sides.
8. Count pipe cross-sections inside each candidate (unchanged from v1).

v1 code (global row/column projection) is commented out below for reference.
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
# Rasterisation helpers (shared by v1 and v2)
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

    border = cell_m * 2
    u0 = u_min - border
    v0 = v_min - border

    n_u = max(1, int(np.ceil((u_max - u_min + 2 * border) / cell_m)))
    n_v = max(1, int(np.ceil((v_max - v_min + 2 * border) / cell_m)))

    ui = np.clip(((uv[:, 0] - u0) / cell_m).astype(int), 0, n_u - 1)
    vi = np.clip(((uv[:, 1] - v0) / cell_m).astype(int), 0, n_v - 1)

    grid = np.zeros((n_v, n_u), dtype=bool)
    grid[vi, ui] = True

    struct = np.ones((3, 3), dtype=bool)
    grid = binary_dilation(grid, structure=struct, iterations=2)

    return grid, u0, u0 + n_u * cell_m, v0, v0 + n_v * cell_m


def _grid_to_world(idx: int, origin: float, cell_m: float) -> float:
    return origin + (idx + 0.5) * cell_m


# ---------------------------------------------------------------------------
# v2: Local segment detection (gap-tolerant)
# ---------------------------------------------------------------------------

def _fill_gaps_1d(arr: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill runs of False up to max_gap cells in a 1D boolean array.

    Handles LiDAR blind spots: a beam interrupted by a shadow still reads
    as a single continuous run after filling.
    Only fills interior gaps (not leading/trailing False regions).
    """
    out = arr.copy()
    n = len(out)
    i = 0
    while i < n:
        if not out[i]:
            j = i
            while j < n and not out[j]:
                j += 1
            # Only fill if gap is bounded on both sides and short enough
            if j - i <= max_gap and i > 0 and j < n:
                out[i:j] = True
            i = max(i + 1, j)
        else:
            i += 1
    return out


def _runs_1d(arr: np.ndarray, min_run: int) -> list[tuple[int, int]]:
    """Return (start, end) inclusive pairs for True runs of length >= min_run."""
    runs = []
    n = len(arr)
    i = 0
    while i < n:
        if arr[i]:
            j = i + 1
            while j < n and arr[j]:
                j += 1
            if j - i >= min_run:
                runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def _find_h_bands(
    grid: np.ndarray,
    max_gap_cells: int = 20,
    min_run_cells: int = 10,
    merge_row_gap: int = 4,
) -> list[dict]:
    """Find horizontal beam bands.

    For each row, gap-fill then find horizontal runs of >= min_run_cells.
    Group rows with runs into bands (consecutive rows within merge_row_gap).

    Returns list of dicts: {row_min, row_max, col_min, col_max}
    """
    beam_rows = []
    for r in range(grid.shape[0]):
        filled = _fill_gaps_1d(grid[r], max_gap_cells)
        runs = _runs_1d(filled, min_run_cells)
        if runs:
            c0 = min(s for s, _ in runs)
            c1 = max(e for _, e in runs)
            beam_rows.append((r, c0, c1))

    if not beam_rows:
        return []

    bands = []
    cur = dict(row_min=beam_rows[0][0], row_max=beam_rows[0][0],
               col_min=beam_rows[0][1], col_max=beam_rows[0][2])
    for r, c0, c1 in beam_rows[1:]:
        if r - cur['row_max'] <= merge_row_gap:
            cur['row_max'] = r
            cur['col_min'] = min(cur['col_min'], c0)
            cur['col_max'] = max(cur['col_max'], c1)
        else:
            bands.append(cur)
            cur = dict(row_min=r, row_max=r, col_min=c0, col_max=c1)
    bands.append(cur)
    return bands


def _find_v_bands(
    grid: np.ndarray,
    max_gap_cells: int = 20,
    min_run_cells: int = 10,
    merge_col_gap: int = 4,
) -> list[dict]:
    """Find vertical beam bands (same logic transposed).

    Returns list of dicts: {col_min, col_max, row_min, row_max}
    """
    beam_cols = []
    for c in range(grid.shape[1]):
        filled = _fill_gaps_1d(grid[:, c], max_gap_cells)
        runs = _runs_1d(filled, min_run_cells)
        if runs:
            r0 = min(s for s, _ in runs)
            r1 = max(e for _, e in runs)
            beam_cols.append((c, r0, r1))

    if not beam_cols:
        return []

    bands = []
    cur = dict(col_min=beam_cols[0][0], col_max=beam_cols[0][0],
               row_min=beam_cols[0][1], row_max=beam_cols[0][2])
    for c, r0, r1 in beam_cols[1:]:
        if c - cur['col_max'] <= merge_col_gap:
            cur['col_max'] = c
            cur['row_min'] = min(cur['row_min'], r0)
            cur['row_max'] = max(cur['row_max'], r1)
        else:
            bands.append(cur)
            cur = dict(col_min=c, col_max=c, row_min=r0, row_max=r1)
    bands.append(cur)
    return bands


def _find_gate_rects_v2(
    h_bands: list[dict],
    v_bands: list[dict],
    cell_m: float,
    u_origin: float,
    v_origin: float,
    min_gate_w_cells: int,
    max_gate_w_cells: int,
    min_gate_h_cells: int,
    max_gate_h_cells: int,
    v_side_tol_cells: int = 15,    # how close a v_band edge must be to the h overlap edge
    v_coverage_frac: float = 0.30, # vertical band must cover >= 30% of gate height
) -> list[dict]:
    """Assemble gate rectangles from horizontal and vertical beam bands.

    Every pair of h_bands (bottom + top) whose opening height and overlapping
    width fall within size limits forms a candidate.  Vertical bands are then
    searched for matching left/right sides.  Confidence is 0.5 (top+bottom
    only), 0.75 (three sides), or 1.0 (all four sides).
    """
    candidates = []

    for i, b1 in enumerate(h_bands):
        for b2 in h_bands[i + 1:]:
            # Ensure bot < top by row index
            bot, top = (b1, b2) if b1['row_min'] < b2['row_min'] else (b2, b1)

            gate_h_cells = top['row_min'] - bot['row_max']
            if not (min_gate_h_cells <= gate_h_cells <= max_gate_h_cells):
                continue

            # U overlap between top and bottom bands
            u_lo = max(bot['col_min'], top['col_min'])
            u_hi = min(bot['col_max'], top['col_max'])
            gate_w_cells = u_hi - u_lo
            if not (min_gate_w_cells <= gate_w_cells <= max_gate_w_cells):
                continue

            v0_c = bot['row_max']
            v1_c = top['row_min']

            # Find best matching left and right vertical bands
            left_v = None
            right_v = None
            for vb in v_bands:
                overlap = min(vb['row_max'], v1_c) - max(vb['row_min'], v0_c)
                if overlap < gate_h_cells * v_coverage_frac:
                    continue
                col_center = (vb['col_min'] + vb['col_max']) / 2
                # Left side
                if abs(col_center - u_lo) <= v_side_tol_cells:
                    if left_v is None or abs(col_center - u_lo) < abs(
                            (left_v['col_min'] + left_v['col_max']) / 2 - u_lo):
                        left_v = vb
                # Right side
                if abs(col_center - u_hi) <= v_side_tol_cells:
                    if right_v is None or abs(col_center - u_hi) < abs(
                            (right_v['col_min'] + right_v['col_max']) / 2 - u_hi):
                        right_v = vb

            n_sides = 2 + (1 if left_v else 0) + (1 if right_v else 0)
            confidence = n_sides / 4.0

            u0_c = left_v['col_max']  if left_v  else u_lo
            u1_c = right_v['col_min'] if right_v else u_hi

            wu0 = _grid_to_world(u0_c, u_origin, cell_m)
            wu1 = _grid_to_world(u1_c, u_origin, cell_m)
            wv0 = _grid_to_world(v0_c, v_origin, cell_m)
            wv1 = _grid_to_world(v1_c, v_origin, cell_m)

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
# v1: Global row/column projection — RETIRED
# Kept for reference. Fatal flaw: normalises by full grid width, so a 5m beam
# in a 60m-wide slice fills only 8% of a row and never exceeds the 20% threshold.
# ---------------------------------------------------------------------------

# def _find_dense_spans(
#     projection: np.ndarray,
#     grid_size: int,
#     min_fill_fraction: float = 0.20,
#     min_span_cells: int = 3,
#     merge_gap_cells: int = 4,
# ) -> list[tuple[int, int]]:
#     norm = projection / max(grid_size, 1)
#     dense = norm >= min_fill_fraction
#     spans = []
#     in_span = False
#     start = 0
#     for i, d in enumerate(dense):
#         if d and not in_span:
#             start = i; in_span = True
#         elif not d and in_span:
#             if i - start >= min_span_cells:
#                 spans.append((start, i - 1))
#             in_span = False
#     if in_span and len(dense) - start >= min_span_cells:
#         spans.append((start, len(dense) - 1))
#     if not spans:
#         return []
#     merged = [spans[0]]
#     for s, e in spans[1:]:
#         if s - merged[-1][1] <= merge_gap_cells:
#             merged[-1] = (merged[-1][0], e)
#         else:
#             merged.append((s, e))
#     return merged
#
#
# def _spans_overlap(a, b, min_overlap_cells=5):
#     return (min(a[1], b[1]) - max(a[0], b[0])) >= min_overlap_cells
#
#
# def _find_gate_rects_v1(h_spans, v_spans, grid, cell_m, u_origin, v_origin,
#                          min_gate_w_cells, max_gate_w_cells,
#                          min_gate_h_cells, max_gate_h_cells):
#     n_rows, n_cols = grid.shape
#     candidates = []
#     for i, h1 in enumerate(h_spans):
#         for h2 in h_spans[i + 1:]:
#             bot, top = h1, h2
#             gate_h = top[0] - bot[1]
#             if not (min_gate_h_cells <= gate_h <= max_gate_h_cells):
#                 continue
#             for j, v1 in enumerate(v_spans):
#                 for v2 in v_spans[j + 1:]:
#                     left, right = v1, v2
#                     gate_w = right[0] - left[1]
#                     if not (min_gate_w_cells <= gate_w <= max_gate_w_cells):
#                         continue
#                     gate_v_range = (bot[0], top[1])
#                     if not (_spans_overlap(left, gate_v_range, 3) and
#                             _spans_overlap(right, gate_v_range, 3)):
#                         continue
#                     u0_c = left[1] + 1; u1_c = right[0] - 1
#                     v0_c = bot[1] + 1;  v1_c = top[0] - 1
#                     wu0 = _grid_to_world(u0_c, u_origin, cell_m)
#                     wu1 = _grid_to_world(u1_c, u_origin, cell_m)
#                     wv0 = _grid_to_world(v0_c, v_origin, cell_m)
#                     wv1 = _grid_to_world(v1_c, v_origin, cell_m)
#                     candidates.append(dict(
#                         bbox_2d=[wu0, wv0, wu1, wv1],
#                         opening_w=wu1-wu0, opening_h=wv1-wv0,
#                         opening_area_m2=(wu1-wu0)*(wv1-wv0),
#                         confidence=1.0,
#                         u0_c=u0_c, u1_c=u1_c, v0_c=v0_c, v1_c=v1_c,
#                     ))
#     return candidates


# ---------------------------------------------------------------------------
# Pipe detection inside a gate (unchanged)
# ---------------------------------------------------------------------------

def _count_pipes_in_gate(
    uv: np.ndarray,
    bbox_2d: list,
    cell_m: float = 0.020,
    min_pipe_r_m: float = 0.025,
    max_pipe_r_m: float = 0.400,
    min_circularity: float = 0.35,
) -> tuple[int, list]:
    u0, v0, u1, v1 = bbox_2d
    mask = (
        (uv[:, 0] >= u0) & (uv[:, 0] <= u1) &
        (uv[:, 1] >= v0) & (uv[:, 1] <= v1)
    )
    sub = uv[mask]
    if len(sub) < 5:
        return 0, []

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
        vr = sl[0]
        ur = sl[1]
        h = vr.stop - vr.start
        w = ur.stop - ur.start
        if h == 0 or w == 0:
            continue
        aspect = min(h, w) / max(h, w)
        if aspect < min_circularity:
            continue
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
    min_beam_fill: float = 0.20,   # retained in signature for compatibility, unused in v2
) -> list[Gate]:
    """Detect pipe rack gates in a 2D cross-section slice (v2 algorithm)."""
    if len(uv) < 10:
        return []

    grid, u_origin, u_end, v_origin, v_end = _rasterize(uv, cell_m)

    min_gate_w_cells = max(1, int(min_gate_w / cell_m))
    max_gate_w_cells = int(max_gate_w / cell_m)
    min_gate_h_cells = max(1, int(min_gate_h / cell_m))
    max_gate_h_cells = int(max_gate_h / cell_m)

    h_bands = _find_h_bands(grid, max_gap_cells=20, min_run_cells=10, merge_row_gap=4)
    v_bands = _find_v_bands(grid, max_gap_cells=20, min_run_cells=10, merge_col_gap=4)

    if not h_bands:
        return []

    rects = _find_gate_rects_v2(
        h_bands, v_bands, cell_m,
        u_origin, v_origin,
        min_gate_w_cells, max_gate_w_cells,
        min_gate_h_cells, max_gate_h_cells,
    )

    gates = []
    seen_bboxes: list[list] = []

    for r in rects:
        bbox2 = [float(v) for v in r["bbox_2d"]]

        # Deduplicate
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

        axis_up = axis.upper()
        u_idx = {"X": 1, "Y": 0}[axis_up]
        v_idx = 2
        ax_idx = {"X": 0, "Y": 1}[axis_up]

        b3 = [0.0] * 6
        b3[ax_idx]     = position_m - thickness_m / 2
        b3[ax_idx + 3] = position_m + thickness_m / 2
        b3[u_idx]      = bbox2[0]
        b3[u_idx + 3]  = bbox2[2]
        b3[v_idx]      = bbox2[1]
        b3[v_idx + 3]  = bbox2[3]

        g = Gate(
            gate_id=f"GATE_{uuid.uuid4().hex[:6].upper()}",
            axis=axis_up,
            position_m=position_m,
            thickness_m=thickness_m,
            bbox_2d=bbox2,
            bbox_3d=b3,
            pipe_count=pipe_count,
            pipe_locs_2d=pipe_locs,
            opening_area_m2=float(r["opening_area_m2"]),
            confidence=float(r["confidence"]),
            source="auto",
        )
        gates.append(g)

    gates.sort(key=lambda g: (-g.confidence, -g.pipe_count))
    return gates
