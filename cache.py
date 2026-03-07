"""Server-side in-memory cache for GateDetector.

Holds the loaded point cloud array (avoid repeated disk reads)
and the live gate registry (persisted to gates.json on every write).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

_cloud: Optional[np.ndarray] = None   # (N, 3) float32
_load_status: str = ""                # progress string for UI polling
_load_progress: float = 0.0           # 0.0–1.0

GATES_FILE = Path(__file__).parent / "gates.json"


# ---------------------------------------------------------------------------
# Cloud
# ---------------------------------------------------------------------------

def set_cloud(pts: np.ndarray) -> None:
    global _cloud
    _cloud = pts.astype(np.float32)


def get_cloud() -> Optional[np.ndarray]:
    return _cloud


def set_status(msg: str, progress: float = -1.0) -> None:
    global _load_status, _load_progress
    _load_status = msg
    if progress >= 0:
        _load_progress = progress


def get_status() -> tuple[str, float]:
    return _load_status, _load_progress


# ---------------------------------------------------------------------------
# Gate registry
# ---------------------------------------------------------------------------

def load_gates() -> list[dict]:
    """Load gates from disk (called on startup / page refresh)."""
    if GATES_FILE.exists():
        try:
            return json.loads(GATES_FILE.read_text())
        except Exception:
            pass
    return []


def save_gates(gates: list[dict]) -> None:
    GATES_FILE.write_text(json.dumps(gates, indent=2))


def get_gates() -> list[dict]:
    return load_gates()


def add_gate(gate: dict) -> list[dict]:
    gates = load_gates()
    # Remove duplicate by gate_id if present
    gates = [g for g in gates if g.get("gate_id") != gate["gate_id"]]
    gates.append(gate)
    save_gates(gates)
    return gates


def remove_gate(gate_id: str) -> list[dict]:
    gates = [g for g in load_gates() if g.get("gate_id") != gate_id]
    save_gates(gates)
    return gates


def clear_gates() -> None:
    save_gates([])
