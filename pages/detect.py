"""Page 2 — Detect: slice viewer, auto-detect, manual gate marking, registry.

Layout
------
  [Slice Controls] (top bar)
  [Plan View XY]  |  [Slice View 2D cross-section]
  [Gate Registry Table]
  [Export Bar]

Trackpad-friendly:
  - All navigation via buttons + number inputs (no 3D manipulation required)
  - Arrow buttons to step through slice positions
  - Click plan view to jump slice position
  - Plotly 2D box-select to draw manual gates
  - Equal aspect ratio on both views
"""
from __future__ import annotations

import json
import csv
import io
import uuid

import numpy as np
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html, ctx

import cache
from gatedetector.slab import extract_slab, plan_projection, cloud_bounds
from gatedetector.detect import detect_gates, Gate

dash.register_page(__name__, path="/detect", title="GateDetector — Detect")


# ---------------------------------------------------------------------------
# Helper figures — defined BEFORE layout
# ---------------------------------------------------------------------------

def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _empty_plan() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,1)",
        xaxis=dict(title="X (m)", color="#666"),
        yaxis=dict(title="Y (m)", color="#666", scaleanchor="x"),
        margin=dict(l=40, r=10, t=10, b=40),
    )
    return fig


def _empty_slice() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,1)",
        xaxis=dict(title="U (m)", color="#666"),
        yaxis=dict(title="Z (m)", color="#666", scaleanchor="x"),
        margin=dict(l=40, r=10, t=10, b=40),
        dragmode="select",
    )
    return fig


def _plan_fig(pts: np.ndarray, axis: str, position_m: float,
              bn: dict, gates: list[dict]) -> go.Figure:
    x, y = plan_projection(pts, n_max=75_000)
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(size=1, color="#3498db", opacity=0.35),
        hovertemplate="X:%{x:.2f} Y:%{y:.2f}<extra></extra>",
        name="Cloud",
    ))

    axis_up = axis.upper()
    if axis_up == "Y":
        fig.add_shape(type="line",
                      x0=bn["xmin"], x1=bn["xmax"],
                      y0=position_m, y1=position_m,
                      line=dict(color="#e74c3c", width=2, dash="dash"))
    else:
        fig.add_shape(type="line",
                      x0=position_m, x1=position_m,
                      y0=bn["ymin"], y1=bn["ymax"],
                      line=dict(color="#e74c3c", width=2, dash="dash"))

    for g in gates:
        b3 = g.get("bbox_3d", [])
        if len(b3) != 6:
            continue
        fig.add_shape(type="rect",
                      x0=b3[0], x1=b3[3], y0=b3[1], y1=b3[4],
                      line=dict(color="#f39c12", width=1),
                      fillcolor="rgba(243,156,18,0.1)")

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,1)",
        xaxis=dict(title="X (m)", color="#aaa", range=[bn["xmin"], bn["xmax"]]),
        yaxis=dict(title="Y (m)", color="#aaa", scaleanchor="x",
                   range=[bn["ymin"], bn["ymax"]]),
        margin=dict(l=40, r=10, t=10, b=40),
        showlegend=False,
    )
    return fig


def _slice_fig(uv: np.ndarray, u_label: str, v_label: str,
               gates: list[dict], position_m: float) -> go.Figure:
    fig = go.Figure()

    if len(uv) > 0:
        n = min(200_000, len(uv))
        if n < len(uv):
            rng = np.random.default_rng(0)
            idx = rng.choice(len(uv), n, replace=False, shuffle=False)
            sub = uv[idx]
        else:
            sub = uv
        fig.add_trace(go.Scatter(
            x=sub[:, 0].tolist(), y=sub[:, 1].tolist(),
            mode="markers",
            marker=dict(size=2, color="#5dade2", opacity=0.5),
            hovertemplate=f"{u_label}:%{{x:.3f}}  Z:%{{y:.3f}}<extra></extra>",
            name="Scan",
        ))

    for g in gates:
        if abs(g.get("position_m", 0) - position_m) > 2.0:
            continue
        bbox = g.get("bbox_2d", [])
        if len(bbox) != 4:
            continue
        u0, v0, u1, v1 = bbox
        lbl   = g.get("label") or g.get("gate_id", "")
        color = "#e74c3c" if g.get("source") == "auto" else "#f39c12"
        r, g2, b = _hex_to_rgb(color)
        fig.add_shape(type="rect", x0=u0, y0=v0, x1=u1, y1=v1,
                      line=dict(color=color, width=2),
                      fillcolor=f"rgba({r},{g2},{b},0.08)")
        fig.add_annotation(x=(u0+u1)/2, y=v1, text=lbl,
                           showarrow=False, font=dict(color=color, size=10), yshift=8)
        for pu, pv in g.get("pipe_locs_2d", []):
            fig.add_shape(type="circle",
                          x0=pu-0.1, y0=pv-0.1, x1=pu+0.1, y1=pv+0.1,
                          line=dict(color="#2ecc71", width=1))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,1)",
        xaxis=dict(title=f"{u_label} (m)", color="#aaa"),
        yaxis=dict(title="Z (m)", color="#aaa", scaleanchor="x"),
        margin=dict(l=40, r=10, t=10, b=40),
        dragmode="select",
        showlegend=False,
    )
    return fig


def _build_table(gates: list[dict]) -> html.Div:
    if not gates:
        return dbc.Alert("No gates registered yet.", color="secondary", className="small py-1")

    header = html.Thead(html.Tr([
        html.Th("ID"), html.Th("Axis"), html.Th("Pos (m)"),
        html.Th("W (m)"), html.Th("H (m)"), html.Th("Area (m²)"),
        html.Th("Pipes"), html.Th("Conf"), html.Th("Src"), html.Th("Label"), html.Th(""),
    ]))

    rows = []
    for g in gates:
        bbox = g.get("bbox_2d", [0, 0, 1, 1])
        w = round(bbox[2] - bbox[0], 2) if len(bbox) == 4 else "—"
        h = round(bbox[3] - bbox[1], 2) if len(bbox) == 4 else "—"
        rows.append(html.Tr([
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
            html.Td(dbc.Button("✕",
                id={"type": "del-gate-btn", "index": g.get("gate_id", "")},
                color="danger", outline=True, size="sm", className="py-0 px-1")),
        ]))

    return dbc.Table(
        [header, html.Tbody(rows)],
        bordered=True, hover=True, responsive=True,
        size="sm", className="table-dark small",
    )


# ---------------------------------------------------------------------------
# Layout — defined AFTER helpers
# ---------------------------------------------------------------------------

layout = dbc.Container(fluid=True, children=[

    # ---- Controls bar ---------------------------------------------------
    dbc.Card(className="mb-3 mt-3", body=True, children=[
        dbc.Row([
            dbc.Col([
                html.Label("Slice axis", className="small text-muted mb-1"),
                dbc.RadioItems(
                    id="axis-radio",
                    options=[
                        {"label": "X  (YZ gate — pipe travels in X)", "value": "X"},
                        {"label": "Y  (XZ gate — pipe travels in Y)", "value": "Y"},
                    ],
                    value="Y",
                    inline=True,
                ),
            ], md=4),

            dbc.Col([
                html.Label("Slice position (m)", className="small text-muted mb-1"),
                dbc.InputGroup([
                    dbc.Button("◄◄", id="step-back10-btn", color="secondary", outline=True, size="sm"),
                    dbc.Button("◄",  id="step-back-btn",   color="secondary", outline=True, size="sm"),
                    dbc.Input(id="pos-input", type="number", step=0.1,
                              placeholder="0.0", className="text-center",
                              style={"maxWidth": "100px"}),
                    dbc.Button("►",  id="step-fwd-btn",    color="secondary", outline=True, size="sm"),
                    dbc.Button("►►", id="step-fwd10-btn",  color="secondary", outline=True, size="sm"),
                ], size="sm"),
            ], md=3),

            dbc.Col([
                html.Label("Step (m)", className="small text-muted mb-1"),
                dbc.Input(id="step-input", type="number", value=0.5, step=0.1,
                          min=0.05, size="sm", style={"maxWidth": "80px"}),
            ], md=1),

            dbc.Col([
                html.Label("Thickness (m)", className="small text-muted mb-1"),
                dbc.Input(id="thick-input", type="number", value=0.4, step=0.05,
                          min=0.05, size="sm", style={"maxWidth": "80px"}),
            ], md=1),

            dbc.Col([
                html.Label("\u00a0", className="small d-block mb-1"),
                dbc.ButtonGroup([
                    dbc.Button("Update Slice", id="slice-btn",  color="primary", size="sm"),
                    dbc.Button("Auto-Detect",  id="detect-btn", color="success", size="sm"),
                ]),
            ], md=3),
        ], className="g-2 align-items-end"),
    ]),

    # ---- Dual view -------------------------------------------------------
    dbc.Row([
        dbc.Col([
            html.H6("Plan View  (click to jump slice position)",
                    className="text-muted mb-1 small"),
            dcc.Graph(
                id="plan-graph",
                figure=_empty_plan(),
                style={"height": "45vh"},
                config={"scrollZoom": True, "displayModeBar": False},
            ),
        ], md=5),

        dbc.Col([
            html.H6(id="slice-title",
                    children="Cross-section  (box-select to mark gate)",
                    className="text-muted mb-1 small"),
            dcc.Graph(
                id="slice-graph",
                figure=_empty_slice(),
                style={"height": "45vh"},
                config={"scrollZoom": True, "displayModeBar": True,
                        "modeBarButtonsToAdd": ["select2d"]},
            ),
        ], md=7),
    ], className="mb-3"),

    # Status line
    html.Div(id="detect-status", className="small text-info mb-2"),

    # ---- Manual gate controls -------------------------------------------
    dbc.Card(className="mb-3", body=True, children=[
        dbc.Row([
            dbc.Col(html.Span("Manual gate from selection:", className="small text-muted"), md=3),
            dbc.Col(dbc.Input(id="gate-label-input",
                              placeholder="Label (optional)", size="sm"), md=4),
            dbc.Col(dbc.Button("Add Selected Gate", id="add-gate-btn",
                               color="warning", outline=True, size="sm"), md=2),
            dbc.Col(dbc.Button("Clear All Gates", id="clear-gates-btn",
                               color="danger", outline=True, size="sm"), md=2),
            dbc.Col(dbc.Button("Refresh", id="refresh-gates-btn",
                               color="secondary", outline=True, size="sm"), md=1),
        ], className="g-2 align-items-center"),
    ]),

    # ---- Gate registry ---------------------------------------------------
    html.H6("Gate Registry", className="mb-1"),
    html.Div(id="gates-table", className="mb-3"),

    # ---- Export ----------------------------------------------------------
    dbc.Row([
        dbc.Col(dbc.Button("Export JSON", id="export-json-btn",
                           color="secondary", outline=True, size="sm"), width="auto"),
        dbc.Col(dbc.Button("Export CSV",  id="export-csv-btn",
                           color="secondary", outline=True, size="sm"), width="auto"),
        dbc.Col(html.Div(id="export-status", className="small text-info")),
    ], className="g-2 align-items-center mb-4"),
    dcc.Download(id="gates-dl"),

    # Hidden store for box-select bbox
    dcc.Store(id="selection-store"),
])


# ---------------------------------------------------------------------------
# Step buttons → update position input
# ---------------------------------------------------------------------------

@callback(
    Output("pos-input", "value"),
    Input("step-back10-btn", "n_clicks"),
    Input("step-back-btn",   "n_clicks"),
    Input("step-fwd-btn",    "n_clicks"),
    Input("step-fwd10-btn",  "n_clicks"),
    Input("plan-graph",      "clickData"),
    State("pos-input",       "value"),
    State("step-input",      "value"),
    State("axis-radio",      "value"),
    prevent_initial_call=True,
)
def update_position(b10, b1, f1, f10, click, pos, step, axis):
    triggered = ctx.triggered_id
    step = float(step or 0.5)
    pos  = float(pos  or 0.0)

    if triggered == "step-back10-btn": return round(pos - step * 10, 3)
    if triggered == "step-back-btn":   return round(pos - step,      3)
    if triggered == "step-fwd-btn":    return round(pos + step,      3)
    if triggered == "step-fwd10-btn":  return round(pos + step * 10, 3)

    if triggered == "plan-graph" and click:
        pt = click["points"][0]
        return round(pt["y"] if (axis or "Y").upper() == "Y" else pt["x"], 2)

    return pos


# ---------------------------------------------------------------------------
# Slice update (Update Slice + Auto-Detect)
# ---------------------------------------------------------------------------

@callback(
    Output("plan-graph",    "figure"),
    Output("slice-graph",   "figure"),
    Output("slice-title",   "children"),
    Output("detect-status", "children"),
    Output("gates-table",   "children",  allow_duplicate=True),
    Input("slice-btn",      "n_clicks"),
    Input("detect-btn",     "n_clicks"),
    State("axis-radio",     "value"),
    State("pos-input",      "value"),
    State("thick-input",    "value"),
    prevent_initial_call=True,
)
def update_slice(slice_clicks, detect_clicks, axis, pos, thick):
    pts = cache.get_cloud()
    if pts is None or len(pts) == 0:
        msg = dbc.Alert("No cloud loaded. Go to Load page first.", color="warning")
        return _empty_plan(), _empty_slice(), "Cross-section", msg, dash.no_update

    axis_up = (axis or "Y").upper()
    pos_m   = float(pos   or 0.0)
    thick_m = float(thick or 0.4)
    bn = cloud_bounds(pts)
    triggered = ctx.triggered_id

    pts_slab, uv, u_label, v_label = extract_slab(pts, axis_up, pos_m, thick_m)
    gates = cache.get_gates()

    if triggered == "detect-btn":
        if len(uv) < 20:
            status = dbc.Alert(
                f"Too few points in slab ({len(uv)}). Adjust position or increase thickness.",
                color="warning")
        else:
            new_gates = detect_gates(uv, axis_up, pos_m, thick_m, pts_slab)
            for g in new_gates:
                cache.add_gate(g.to_dict())
            gates = cache.get_gates()
            status = dbc.Alert(
                f"Detected {len(new_gates)} gate(s) in slab  |  {len(uv):,} pts sampled.",
                color="success", className="py-1")
    else:
        status = f"{len(uv):,} pts in slab."

    plan  = _plan_fig(pts, axis_up, pos_m, bn, gates)
    slc   = _slice_fig(uv, u_label, v_label, gates, pos_m)
    title = (f"Cross-section  |  axis {axis_up}  @  {pos_m:.2f} m  "
             f"|  thick {thick_m:.2f} m  |  {len(uv):,} pts  "
             "(box-select to mark gate)")

    return plan, slc, title, status, _build_table(gates)


# ---------------------------------------------------------------------------
# Auto-init plan view on page load
# ---------------------------------------------------------------------------

@callback(
    Output("plan-graph", "figure",  allow_duplicate=True),
    Output("pos-input",  "value",   allow_duplicate=True),
    Input("store",       "data"),
    prevent_initial_call="initial_duplicate",
)
def init_plan(store):
    pts = cache.get_cloud()
    if pts is None or len(pts) == 0:
        return _empty_plan(), dash.no_update
    bn  = cloud_bounds(pts)
    mid = round((bn["ymin"] + bn["ymax"]) / 2, 2)
    return _plan_fig(pts, "Y", mid, bn, cache.get_gates()), mid


# ---------------------------------------------------------------------------
# Capture box-select bbox from slice graph
# ---------------------------------------------------------------------------

@callback(
    Output("selection-store", "data"),
    Input("slice-graph",  "selectedData"),
    State("axis-radio",   "value"),
    State("pos-input",    "value"),
    State("thick-input",  "value"),
    prevent_initial_call=True,
)
def capture_selection(sel, axis, pos, thick):
    if not sel or "range" not in sel:
        return None
    r = sel["range"]
    u0, u1 = sorted(r["x"])
    v0, v1 = sorted(r["y"])
    return dict(
        axis=axis or "Y",
        position_m=float(pos   or 0),
        thickness_m=float(thick or 0.4),
        bbox_2d=[u0, v0, u1, v1],
    )


# ---------------------------------------------------------------------------
# Add manual gate
# ---------------------------------------------------------------------------

@callback(
    Output("gates-table",    "children",  allow_duplicate=True),
    Output("detect-status",  "children",  allow_duplicate=True),
    Input("add-gate-btn",    "n_clicks"),
    State("selection-store", "data"),
    State("gate-label-input","value"),
    prevent_initial_call=True,
)
def add_manual_gate(n, sel, label):
    if not sel:
        return (dash.no_update,
                dbc.Alert("Box-select a region in the slice view first.", color="warning"))
    axis_up  = sel["axis"].upper()
    pos_m    = sel["position_m"]
    thick_m  = sel["thickness_m"]
    bbox2    = sel["bbox_2d"]
    u_idx = {"X": 1, "Y": 0}[axis_up]
    ax_idx = {"X": 0, "Y": 1}[axis_up]

    b3 = [0.0] * 6
    b3[ax_idx]     = pos_m - thick_m / 2
    b3[ax_idx + 3] = pos_m + thick_m / 2
    b3[u_idx]      = bbox2[0]
    b3[u_idx + 3]  = bbox2[2]
    b3[2]          = bbox2[1]
    b3[5]          = bbox2[3]

    w = bbox2[2] - bbox2[0]
    h = bbox2[3] - bbox2[1]
    gid = f"GATE_{uuid.uuid4().hex[:6].upper()}"
    g = dict(
        gate_id=gid, axis=axis_up, position_m=pos_m, thickness_m=thick_m,
        bbox_2d=bbox2, bbox_3d=b3, pipe_count=0, pipe_locs_2d=[],
        opening_area_m2=w * h, confidence=1.0, source="manual",
        label=label or "", notes="",
    )
    gates = cache.add_gate(g)
    return (
        _build_table(gates),
        dbc.Alert(f"Manual gate added: {gid}", color="success", className="py-1"),
    )


# ---------------------------------------------------------------------------
# Clear all gates
# ---------------------------------------------------------------------------

@callback(
    Output("gates-table",    "children",  allow_duplicate=True),
    Output("detect-status",  "children",  allow_duplicate=True),
    Input("clear-gates-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_all_gates(_):
    cache.clear_gates()
    return _build_table([]), dbc.Alert("All gates cleared.", color="info", className="py-1")


# ---------------------------------------------------------------------------
# Refresh table
# ---------------------------------------------------------------------------

@callback(
    Output("gates-table", "children", allow_duplicate=True),
    Input("refresh-gates-btn", "n_clicks"),
    prevent_initial_call=True,
)
def refresh_gates(_):
    return _build_table(cache.get_gates())


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
        return dict(content=json.dumps(gates, indent=2), filename="gates.json"), \
               f"Exported {len(gates)} gates as JSON."

    if triggered == "export-csv-btn":
        buf = io.StringIO()
        keys = ["gate_id","axis","position_m","thickness_m","opening_area_m2",
                "pipe_count","confidence","source","label","bbox_2d","bbox_3d","notes"]
        writer = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for g in gates:
            row = {k: g.get(k, "") for k in keys}
            row["bbox_2d"] = json.dumps(g.get("bbox_2d", []))
            row["bbox_3d"] = json.dumps(g.get("bbox_3d", []))
            writer.writerow(row)
        return dict(content=buf.getvalue(), filename="gates.csv"), \
               f"Exported {len(gates)} gates as CSV."

    return dash.no_update, dash.no_update
