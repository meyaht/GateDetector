"""Page 1 — Detect: image-based gate review and registry management.

No point cloud loading required.  GateDetector now displays pre-generated
PNG images produced by AutoGateDetector:
  - plan.png      — top-down XY overview of the full cloud with gate footprints
  - slice_*.png   — per-gate cross-section slices

Click the ⬜ button on any gate row to view that gate's slice image.
"""
from __future__ import annotations

import base64
import csv
import io
import json
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html, ctx

import cache

dash.register_page(__name__, path="/", title="GateDetector")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _img_src(path: Path | None) -> str:
    """Return a base64 PNG data-URL, or empty string if the file is missing."""
    if path and path.exists():
        data = base64.b64encode(path.read_bytes()).decode()
        return f"data:image/png;base64,{data}"
    return ""


def _plan_src() -> str:
    images_dir = cache.get_images_dir()
    if not images_dir:
        return ""
    gates = cache.load_gates()
    meta = next((g for g in gates if g.get("gate_id") == "_CLOUD_META_"), None)
    if meta and meta.get("plan_image"):
        return _img_src(images_dir / meta["plan_image"])
    return _img_src(images_dir / "plan.png")


def _build_table(gates: list[dict]) -> html.Div:
    meta = next((g for g in gates if g.get("gate_id") == "_CLOUD_META_"), None)
    real_gates = [g for g in gates if g.get("gate_id") != "_CLOUD_META_"]

    parts = []

    fname = cache.GATES_FILE.name
    rot = float(meta.get("plan_rotation", 0)) if meta else 0.0
    bmin = meta.get("cloud_bmin") if meta else None
    bmax = meta.get("cloud_bmax") if meta else None
    bounds = ""
    if bmin and bmax:
        bounds = (f"   X {bmin[0]:.1f}–{bmax[0]:.1f}  "
                  f"Y {bmin[1]:.1f}–{bmax[1]:.1f}  "
                  f"Z {bmin[2]:.1f}–{bmax[2]:.1f} m")
    parts.append(dbc.Alert(
        f"{fname}   |   rotation: {rot}°{bounds}",
        color="dark", className="py-1 mb-2 small font-monospace",
    ))

    if not real_gates:
        parts.append(dbc.Alert("No gates registered yet.", color="secondary", className="small py-1"))
        return html.Div(parts)

    header = html.Thead(html.Tr([
        html.Th(""),
        html.Th("ID"), html.Th("Axis"), html.Th("Pos (m)"),
        html.Th("W (m)"), html.Th("H (m)"), html.Th("Area (m²)"),
        html.Th("Pipes"), html.Th("Conf"), html.Th("Src"), html.Th("Label"), html.Th(""),
    ]))

    rows = []
    for g in real_gates:
        bbox = g.get("bbox_2d", [0, 0, 1, 1])
        w = round(bbox[2] - bbox[0], 2) if len(bbox) == 4 else "—"
        h = round(bbox[3] - bbox[1], 2) if len(bbox) == 4 else "—"
        rows.append(html.Tr([
            html.Td(dbc.Button(
                "⬜",
                id={"type": "view-gate-btn", "index": g.get("gate_id", "")},
                color="info", outline=True, size="sm", className="py-0 px-1",
                title="View slice image",
            )),
            html.Td(html.Span(g.get("gate_id", ""), className="font-monospace small")),
            html.Td(g.get("axis", "?")),
            html.Td(f"{g.get('position_m', 0):.2f}"),
            html.Td(w),
            html.Td(h),
            html.Td(f"{g.get('opening_area_m2', 0):.2f}"),
            html.Td(g.get("pipe_count", 0)),
            html.Td(f"{g.get('confidence', 0):.2f}"),
            html.Td(dbc.Badge(
                g.get("source", "?"),
                color="success" if g.get("source") == "auto" else "warning",
            )),
            html.Td(g.get("label", "")),
            html.Td(dbc.Button(
                "✕",
                id={"type": "del-gate-btn", "index": g.get("gate_id", "")},
                color="danger", outline=True, size="sm", className="py-0 px-1",
            )),
        ]))

    parts.append(dbc.Table(
        [header, html.Tbody(rows)],
        bordered=True, hover=True, responsive=True,
        size="sm", className="table-dark small",
    ))
    return html.Div(parts)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

layout = dbc.Container(fluid=True, children=[

    # ---- Dual image view -------------------------------------------------
    dbc.Row([
        dbc.Col([
            html.H6("Plan View  (XY overview)", className="text-muted mb-1 small"),
            html.Img(
                id="plan-img",
                style={"width": "100%", "maxHeight": "48vh",
                       "objectFit": "contain", "background": "#0a0a14",
                       "display": "block"},
            ),
        ], md=5),

        dbc.Col([
            dbc.Row([
                dbc.Col(dbc.Button("◀", id="nav-prev-btn", n_clicks=0,
                                   color="secondary", outline=True, size="sm",
                                   title="Previous gate  (←)"), width="auto"),
                dbc.Col(html.H6(id="slice-title",
                                children="Click ⬜ on a gate row to view its slice image",
                                className="text-muted mb-0 small"),
                        className="align-self-center"),
                dbc.Col(dbc.Button("▶", id="nav-next-btn", n_clicks=0,
                                   color="secondary", outline=True, size="sm",
                                   title="Next gate  (→)"), width="auto"),
            ], className="mb-1 align-items-center g-2"),
            html.Img(
                id="slice-img",
                style={"width": "100%", "maxHeight": "48vh",
                       "objectFit": "contain", "background": "#0a0a14",
                       "display": "block"},
            ),
        ], md=7),
    ], className="mb-3 mt-3"),

    # ---- Status line -----------------------------------------------------
    html.Div(id="detect-status", className="small text-info mb-2"),

    # ---- Import / Export / Clear (above table) ---------------------------
    dbc.Row([
        dbc.Col(dbc.Button("Import Gates JSON…", id="import-gates-btn",
                           color="info", outline=True, size="sm"), width="auto"),
        dbc.Col(dbc.Button("Export JSON", id="export-json-btn",
                           color="secondary", outline=True, size="sm"), width="auto"),
        dbc.Col(dbc.Button("Export CSV",  id="export-csv-btn",
                           color="secondary", outline=True, size="sm"), width="auto"),
        dbc.Col(dbc.Button("Clear All Gates", id="clear-gates-btn",
                           color="danger", outline=True, size="sm"), width="auto"),
        dbc.Col(html.Div(id="export-status", className="small text-info")),
    ], className="g-2 align-items-center mb-2"),

    # ---- Gate registry ---------------------------------------------------
    html.H6("Gate Registry", className="mb-1"),
    html.Div(id="gates-table", children=_build_table(cache.load_gates()), className="mb-3"),

    dcc.Download(id="gates-dl"),

    # Navigation state + dummy store for one-shot key listener setup
    dcc.Store(id="gate-nav-idx",  data=0),
    dcc.Store(id="kb-init-dummy", data=None),
])


# ---------------------------------------------------------------------------
# Init: load plan image when page loads (fires on store init)
# ---------------------------------------------------------------------------

@callback(
    Output("plan-img", "src"),
    Input("store", "data"),
)
def init_plan_image(_store):
    return _plan_src()


# ---------------------------------------------------------------------------
# Helpers for slice-based navigation
# ---------------------------------------------------------------------------

def _unique_slice_list(real_gates: list) -> list[dict]:
    """Return ordered list of unique slices (one entry per distinct slice_image).

    Each entry: {"fname": str, "axis": str, "position_m": float, "gates": [...]}.
    """
    seen: dict[str, dict] = {}
    for g in real_gates:
        fname = g.get("slice_image")
        if not fname:
            continue
        if fname not in seen:
            seen[fname] = {
                "fname":      fname,
                "axis":       g.get("axis", "?"),
                "position_m": g.get("position_m", 0),
                "gates":      [],
            }
        seen[fname]["gates"].append(g)
    return list(seen.values())


def _slice_title(sl: dict, idx: int, total: int) -> str:
    gate_ids = "  ".join(g["gate_id"] for g in sl["gates"])
    pipes    = sum(g.get("pipe_count", 0) for g in sl["gates"])
    conf     = max((g.get("confidence", 0) for g in sl["gates"]), default=0)
    return (f"[{idx+1}/{total}]  {sl['axis']}={sl['position_m']:.2f} m  "
            f"|  {len(sl['gates'])} gate(s)  {pipes} pipes  conf={conf:.2f}  "
            f"|  {gate_ids}")


# ---------------------------------------------------------------------------
# View gate slice image (⬜ button — jumps to that slice)
# ---------------------------------------------------------------------------

@callback(
    Output("slice-img",    "src"),
    Output("slice-title",  "children"),
    Output("gate-nav-idx", "data"),
    Input({"type": "view-gate-btn", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def view_gate_slice(n_clicks_list):
    triggered = ctx.triggered_id
    if not triggered or not any(n for n in n_clicks_list if n):
        return dash.no_update, dash.no_update, dash.no_update

    gate_id = triggered["index"]
    real_gates = [g for g in cache.load_gates() if g.get("gate_id") != "_CLOUD_META_"]
    g = next((g for g in real_gates if g.get("gate_id") == gate_id), None)
    if not g:
        return "", f"Gate {gate_id} not found", 0

    img_fname = g.get("slice_image")
    images_dir = cache.get_images_dir()
    if not img_fname or not images_dir:
        return "", f"{gate_id}  —  no slice image", 0

    slices = _unique_slice_list(real_gates)
    idx = next((i for i, sl in enumerate(slices) if sl["fname"] == img_fname), 0)

    src = _img_src(images_dir / img_fname)
    return src, _slice_title(slices[idx], idx, len(slices)), idx


# ---------------------------------------------------------------------------
# Delete individual gate
# ---------------------------------------------------------------------------

@callback(
    Output("gates-table",   "children",  allow_duplicate=True),
    Output("detect-status", "children",  allow_duplicate=True),
    Input({"type": "del-gate-btn", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def delete_gate(n_clicks_list):
    triggered = ctx.triggered_id
    if not triggered or not any(n for n in n_clicks_list if n):
        return dash.no_update, dash.no_update
    gates = cache.remove_gate(triggered["index"])
    return _build_table(gates), dbc.Alert(f"Removed {triggered['index']}", color="secondary", className="py-1")


# ---------------------------------------------------------------------------
# Clear all gates
# ---------------------------------------------------------------------------

@callback(
    Output("gates-table",   "children",  allow_duplicate=True),
    Output("detect-status", "children",  allow_duplicate=True),
    Input("clear-gates-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_all_gates(_):
    cache.clear_gates()
    return _build_table([]), dbc.Alert("All gates cleared.", color="info", className="py-1")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

@callback(
    Output("gates-dl",      "data"),
    Output("export-status", "children"),
    Input("export-json-btn","n_clicks"),
    Input("export-csv-btn", "n_clicks"),
    prevent_initial_call=True,
)
def export_gates(json_n, csv_n):
    gates = cache.get_gates()
    if not gates:
        return dash.no_update, dbc.Alert("No gates to export.", color="warning")

    triggered = ctx.triggered_id

    if triggered == "export-json-btn":
        return (dict(content=json.dumps(gates, indent=2), filename="gates.json"),
                f"Exported {len(gates)} gates as JSON.")

    if triggered == "export-csv-btn":
        buf = io.StringIO()
        keys = ["gate_id", "axis", "position_m", "thickness_m", "opening_area_m2",
                "pipe_count", "confidence", "source", "label", "bbox_2d", "bbox_3d", "notes"]
        writer = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for g in gates:
            row = {k: g.get(k, "") for k in keys}
            row["bbox_2d"] = json.dumps(g.get("bbox_2d", []))
            row["bbox_3d"] = json.dumps(g.get("bbox_3d", []))
            writer.writerow(row)
        return (dict(content=buf.getvalue(), filename="gates.csv"),
                f"Exported {len(gates)} gates as CSV.")

    return dash.no_update, dash.no_update


# ---------------------------------------------------------------------------
# Import gates from AutoGateDetector / GateDetector JSON
# ---------------------------------------------------------------------------

@callback(
    Output("gates-table",   "children",  allow_duplicate=True),
    Output("export-status", "children",  allow_duplicate=True),
    Output("plan-img",      "src",       allow_duplicate=True),
    Input("import-gates-btn", "n_clicks"),
    prevent_initial_call=True,
)
def import_gates(_):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select gates.json",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
    )
    root.destroy()
    if not path:
        return dash.no_update, dash.no_update, dash.no_update

    try:
        with open(path) as f:
            raw = json.load(f)
        gate_list = raw.get("gates", []) if isinstance(raw, dict) else raw
        # Keep _CLOUD_META_ and any gate with a bbox_3d
        gate_list = [g for g in gate_list
                     if g.get("gate_id") == "_CLOUD_META_" or "bbox_3d" in g]
        cache.save_gates(gate_list)

        images_dir = Path(path).parent
        cache.set_images_dir(images_dir)

        plan_src = _plan_src()

        n_real = sum(1 for g in gate_list if g.get("gate_id") != "_CLOUD_META_")
        return (
            _build_table(gate_list),
            dbc.Alert(f"Imported {n_real} gates from {Path(path).name}",
                      color="info", className="py-1"),
            plan_src,
        )
    except Exception as e:
        return dash.no_update, dbc.Alert(f"Import error: {e}", color="danger", className="py-1"), dash.no_update


# ---------------------------------------------------------------------------
# Arrow-key navigation — bind keys to ◀/▶ buttons (fires once on page load)
# ---------------------------------------------------------------------------

dash.clientside_callback(
    """
    function(_store) {
        if (!window._gd_kb_listener) {
            document.addEventListener('keydown', function(e) {
                var btn;
                if (e.key === 'ArrowRight') {
                    btn = document.getElementById('nav-next-btn');
                } else if (e.key === 'ArrowLeft') {
                    btn = document.getElementById('nav-prev-btn');
                }
                if (btn) { btn.click(); e.preventDefault(); }
            });
            window._gd_kb_listener = true;
        }
        return null;
    }
    """,
    Output("kb-init-dummy", "data"),
    Input("store", "data"),
)


@callback(
    Output("slice-img",    "src",      allow_duplicate=True),
    Output("slice-title",  "children", allow_duplicate=True),
    Output("gate-nav-idx", "data",     allow_duplicate=True),
    Input("nav-prev-btn",  "n_clicks"),
    Input("nav-next-btn",  "n_clicks"),
    State("gate-nav-idx",  "data"),
    prevent_initial_call=True,
)
def navigate_gate(_prev, _next, nav_idx):
    direction = 1 if ctx.triggered_id == "nav-next-btn" else -1
    real_gates = [g for g in cache.load_gates() if g.get("gate_id") != "_CLOUD_META_"]
    slices = _unique_slice_list(real_gates)
    if not slices:
        return dash.no_update, dash.no_update, nav_idx

    idx = (nav_idx + direction) % len(slices)
    sl = slices[idx]
    images_dir = cache.get_images_dir()
    if not images_dir:
        return "", _slice_title(sl, idx, len(slices)), idx

    src = _img_src(images_dir / sl["fname"])
    return src, _slice_title(sl, idx, len(slices)), idx
