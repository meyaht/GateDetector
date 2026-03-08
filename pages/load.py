"""Load page — functionality merged into detect.py (root page).
This file is kept as a stub to avoid breaking imports."""
import dash

dash.register_page(__name__, path="/load", title="GateDetector — Load")

layout = dash.html.Div("Load functionality has moved to the main page.", className="m-4 text-muted")
