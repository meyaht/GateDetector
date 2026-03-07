"""Page 1 — Load: import point cloud, show load status, proceed to Detect."""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html

import cache
from gatedetector.slab import cloud_bounds

dash.register_page(__name__, path="/", title="GateDetector — Load")

_SUPPORTED = ".npy  |  .e57  |  .pts  |  .xyz"

layout = dbc.Container(fluid=True, children=[
    dbc.Row([
        dbc.Col(html.H4("Load Point Cloud", className="mt-3 mb-1")),
    ]),
    dbc.Row(dbc.Col(html.P(
        "Load a pre-downsampled .npy file for best performance. "
        "Large .e57 / .pts files will be voxel-downsampled at 15 mm on load. "
        "When loaded, proceed to the Detect page.",
        className="text-muted small mb-2",
    ))),

    # File input
    dbc.Row([
        dbc.Col(dbc.Input(
            id="load-path-input",
            placeholder="Paste full path to cloud file…",
            debounce=False,
            className="font-monospace",
        ), md=8),
        dbc.Col(dbc.Button("Load", id="load-btn", color="primary", className="w-100"), md=2),
        dbc.Col(dbc.Button("Clear", id="clear-btn", color="secondary", outline=True, className="w-100"), md=2),
    ], className="g-2 mb-3"),

    dcc.Interval(id="load-poll", interval=400, n_intervals=0, disabled=True),

    # Status card — the only feedback the user needs
    dbc.Card(id="load-status-card", className="mb-3", children=[
        dbc.CardBody([
            dbc.Progress(id="load-progress", value=0, striped=True, animated=True,
                         className="mb-2", style={"height": "10px"}),
            html.Div(id="load-status", className="small"),
        ])
    ]),

    # Cloud info panel (hidden until loaded)
    html.Div(id="cloud-info-panel"),
])


# ---------------------------------------------------------------------------
# Load trigger
# ---------------------------------------------------------------------------

@callback(
    Output("load-poll",   "disabled"),
    Output("load-status", "children", allow_duplicate=True),
    Input("load-btn",     "n_clicks"),
    State("load-path-input", "value"),
    prevent_initial_call=True,
)
def trigger_load(n_clicks, path):
    if not path or not path.strip():
        return True, dbc.Alert("Enter a file path first.", color="warning", className="py-1 mb-0")
    p = Path(path.strip())
    if not p.exists():
        return True, dbc.Alert(f"File not found: {p}", color="danger", className="py-1 mb-0")

    def _load():
        try:
            cache.set_status("Starting load…", 0.0)
            suffix = p.suffix.lower()

            if suffix == ".npy":
                cache.set_status(f"Reading {p.name}…", 0.1)
                pts = np.load(str(p)).astype(np.float32)
                if pts.ndim != 2 or pts.shape[1] < 3:
                    cache.set_status("Error: .npy must be (N, 3) array.", 0.0)
                    return
                cache.set_status("Caching…", 0.9)
                cache.set_cloud(pts[:, :3])

            elif suffix == ".e57":
                # Reuse NeuralPipe's downsampler if available
                try:
                    import sys
                    sys.path.insert(0, str(Path(__file__).parents[2] / "NeuralPipe-v2"))
                    from neuralpipe.geometry.voxel_grid import downsample_to_npy
                except ImportError:
                    cache.set_status("Error: pye57 / NeuralPipe not found. Pre-downsample to .npy.", 0.0)
                    return
                out_npy = p.parent / (p.stem + "_15mm.npy")
                pts = downsample_to_npy(p, out_npy, cell_size_m=0.015,
                                        progress_callback=lambda m: cache.set_status(m, -1))
                cache.set_cloud(pts)

            elif suffix in (".pts", ".xyz", ".txt"):
                try:
                    import sys
                    sys.path.insert(0, str(Path(__file__).parents[2] / "NeuralPipe-v2"))
                    from neuralpipe.geometry.voxel_grid import downsample_to_npy
                except ImportError:
                    cache.set_status("Error: NeuralPipe not found. Pre-downsample to .npy.", 0.0)
                    return
                out_npy = p.parent / (p.stem + "_15mm.npy")
                pts = downsample_to_npy(p, out_npy, cell_size_m=0.015,
                                        progress_callback=lambda m: cache.set_status(m, -1))
                cache.set_cloud(pts)

            else:
                cache.set_status(f"Unsupported format. Use: {_SUPPORTED}", 0.0)
                return

            n = len(cache.get_cloud())
            cache.set_status(f"READY:{n}", 1.0)

        except Exception as e:
            cache.set_status(f"Error: {e}", 0.0)

    threading.Thread(target=_load, daemon=True).start()
    return False, html.Span("Loading, please wait…", className="text-info")


# ---------------------------------------------------------------------------
# Poll callback — updates progress bar and status only
# ---------------------------------------------------------------------------

@callback(
    Output("load-poll",         "disabled",  allow_duplicate=True),
    Output("load-status",       "children",  allow_duplicate=True),
    Output("load-progress",     "value"),
    Output("cloud-info-panel",  "children"),
    Output("store",             "data",      allow_duplicate=True),
    Input("load-poll",          "n_intervals"),
    State("store",              "data"),
    prevent_initial_call=True,
)
def poll_load(n, store):
    msg, progress = cache.get_status()
    store = store or {}

    bar_val = max(0, min(100, int(progress * 100))) if progress >= 0 else 50

    if not msg.startswith("READY:"):
        status_el = html.Span(msg or "Idle.", className="text-info")
        return False, status_el, bar_val, dash.no_update, dash.no_update

    # Loaded successfully
    n_pts = int(msg.split(":", 1)[1])
    pts = cache.get_cloud()
    bn = cloud_bounds(pts) if pts is not None and len(pts) > 0 else {}

    status_el = dbc.Alert(
        [html.Strong("Loaded. "), f"{n_pts:,} points — proceed to ",
         dbc.NavLink("Detect", href="/detect", className="d-inline p-0")],
        color="success", className="py-1 mb-0",
    )

    info_panel = dbc.Card(body=True, color="dark", className="mb-3", children=[
        html.H6("Cloud Summary", className="mb-2"),
        dbc.Row([
            dbc.Col([html.Strong("Points: "), f"{n_pts:,}"], md=3),
            dbc.Col([html.Strong("X: "), f"{bn.get('xmin',0):.2f} — {bn.get('xmax',0):.2f} m"], md=3),
            dbc.Col([html.Strong("Y: "), f"{bn.get('ymin',0):.2f} — {bn.get('ymax',0):.2f} m"], md=3),
            dbc.Col([html.Strong("Z: "), f"{bn.get('zmin',0):.2f} — {bn.get('zmax',0):.2f} m"], md=3),
        ], className="small"),
    ])

    store.update(
        cloud_bmin=[bn.get("xmin", 0), bn.get("ymin", 0), bn.get("zmin", 0)],
        cloud_bmax=[bn.get("xmax", 0), bn.get("ymax", 0), bn.get("zmax", 0)],
    )

    return True, status_el, 100, info_panel, store


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------

@callback(
    Output("load-status",      "children",  allow_duplicate=True),
    Output("cloud-info-panel", "children",  allow_duplicate=True),
    Output("load-progress",    "value",     allow_duplicate=True),
    Input("clear-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_cloud(_):
    cache.set_cloud(np.zeros((0, 3), dtype=np.float32))
    cache.set_status("", 0.0)
    return html.Span("Cloud cleared.", className="text-muted"), None, 0
