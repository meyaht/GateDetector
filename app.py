"""GateDetector — standalone Dash app for pipe rack gate identification."""
import threading
import webbrowser

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
)
server = app.server

_nav = dbc.NavbarSimple(
    brand="GateDetector",
    brand_href="/",
    color="dark",
    dark=True,
    className="mb-0 px-3",
    children=[],
)

app.layout = html.Div([
    _nav,
    # Shared store — schema:
    #   cloud_path, cloud_bmin, cloud_bmax,
    #   gates (list of gate dicts)
    dcc.Store(id="store", storage_type="session"),
    dash.page_container,
])

if __name__ == "__main__":
    threading.Timer(1.5, lambda: webbrowser.open("http://localhost:8051")).start()
    app.run(debug=False, port=8051)
