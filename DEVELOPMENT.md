# GateDetector — Development Notes

## Project Overview
Standalone Dash app for identifying pipe rack gate openings in large refinery point clouds.
Companion to NeuralPipe-v2. Runs on port 8051.

**Primary test cloud:** `DET204839_Husky-DHT_15mm.npy` — 1.2 GB, ~105M pts, float32, 15mm voxel downsample of DHT unit E57 scan. Located at `C:/Users/zkrep/pointData/`.

**DHT unit:** Main rack ~125m long, multi-level, hundreds of expected gates. XZ and YZ planes only (not XY). Over-detect philosophy — catch everything, prune later.

---

## Architecture

| File | Role |
|------|------|
| `app.py` | Dash entry point, navbar, shared `dcc.Store`, port 8051, `debug=False` |
| `pages/detect.py` | Single combined page at `/` — load bar at top, plan + slice views, gate registry |
| `pages/load.py` | Stub only (retired — functionality merged into detect.py) |
| `cache.py` | Module-level in-memory cache for point cloud and load status |
| `gatedetector/slab.py` | Slab extraction, plan projection, cloud bounds |
| `gatedetector/detect.py` | Gate detection algorithms |

---

## Key Technical Decisions

### debug=False
Dash `debug=True` spawns two processes. Module-level cache is not shared between them —
background load thread writes READY in process A, poll callback reads empty cache in process B.
Fixed permanently with `debug=False`.

### Single-page design
Originally two pages (Load at `/`, Detect at `/detect`). Merged into `detect.py` at `/` so the
plan view renders immediately after load completes without page navigation. After loading a large
file, user hits F5 to trigger `init_plan`, then waits ~90s for plan view to render.

### Load strategy
- `np.load()` direct (no mmap) into memory as float32
- mmap + astype was tested and caused severe slowdown (~10 min) due to page-fault overhead vs
  direct load (~3 min) for the 1.2 GB cloud
- Background thread + 400ms poll interval — never blocks Dash main thread
- Load timer (1-second interval) shows M:SS elapsed below Load button, turns red at 3 min

### Plan rotation
- Buttons: −5° / −1° / 0° / +1° / +5°, stored as `plan_rotation` in `dcc.Store`
- Only the 75k display subsample and slab subset are rotated — NOT the full 105M-point cloud
- `_slab_mask()` computes only the rotated axis coordinate to find slab members efficiently
- `_rotate_pts()` rotates only the slab subset before passing to `extract_slab`

### Point display cap
75k points max in plan view scatter — 300k+ points produce ~20MB JSON and silently freeze
the browser tab.

---

## Detection Algorithm

### Gate geometry assumptions
- Gates occur in XZ planes (axis=Y) and YZ planes (axis=X) only — not XY
- Top and bottom edges are nearly horizontal (within ±3° of horizontal)
- At least one side (left or right) is nearly vertical
- Over-detect: better to find too many than too few
- LiDAR blind spots and shadows create gaps in what should be continuous lines

### v1: Global Row/Column Projection — RETIRED (commented out in detect.py)

**Approach:** Rasterize slice to 30mm grid → row sums → normalize by full grid width →
threshold at 20% fill fraction.

**Fatal flaw:** In a 60m-wide slice, a 5m orange gate beam fills only ~8% of a row
(5m / 60m = 8.3%). The 20% threshold kills it before rectangle assembly starts.
Works only if the slice is zoomed tightly to a single gate.

### v2: Local Segment Detection — CURRENT

**Approach:**
1. Rasterize 2D slice to 30mm binary grid (same as v1)
2. Per-row gap-filling: tolerate gaps up to 20 cells (0.6m) for LiDAR blind spots
3. Per-row run detection: find horizontal runs ≥ 10 cells (0.3m), anywhere in the row
4. Group nearby beam rows (within 4 rows = 120mm) into horizontal bands
5. Same process column-wise for vertical bands
6. Pair every top+bottom h_band combination within size limits
7. For each pair, look for matching v_bands on left and right sides
8. Confidence: 0.5 (top+bottom only) → 0.75 (+ one side) → 1.0 (all 4 sides)
9. Rectangles accepted even with only 2 confirmed sides (top+bottom) — LiDAR may not
   see all sides of a structure

**Why this works at scale:** Run detection is local — a 5m beam in a 60m-wide grid
is detected because it has a 167-cell run in its row, regardless of what the rest of
the row contains.

---

## Known Issues / Pending Work

- Load timer occasionally gets stuck at 2s; workaround: wait ~3 min, F5, wait ~90s
- Overall load UX needs redesign (F5 workflow is awkward for large clouds)
- Gate size limits (max_gate_w=8m, max_gate_h=6m) may need UI controls for large outer gates
- No angle tolerance yet for slightly non-horizontal/non-vertical gate frames
  (user can pre-rotate cloud using plan rotation buttons to compensate)
