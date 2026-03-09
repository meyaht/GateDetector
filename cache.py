"""Server-side in-memory cache for GateDetector.

Holds the gate registry (persisted to gates.json on every write)
and the images directory for the current run's PNG outputs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

_cloud: Optional[np.ndarray] = None   # kept for backwards compat, not used by UI
_load_status: str = ""
_load_progress: float = 0.0
_images_dir: Optional[Path] = None    # directory containing plan.png + slice PNGs

GATES_FILE = Path(os.environ.get("GATEDETECTOR_GATES_FILE",
                                 Path(__file__).parent / "gates.json"))


# ---------------------------------------------------------------------------
# Images directory
# ---------------------------------------------------------------------------

def set_images_dir(p: Path) -> None:
    global _images_dir
    _images_dir = p


def get_images_dir() -> Optional[Path]:
    """Return directory containing plan.png and slice PNGs for the current run.

    Priority:
    1. Explicitly set via set_images_dir() (after import dialog)
    2. Parent of GATEDETECTOR_GATES_FILE env var (command-line launch)
    3. None (no run loaded yet)
    """
    if _images_dir:
        return _images_dir
    env_path = os.environ.get("GATEDETECTOR_GATES_FILE")
    if env_path:
        return Path(env_path).parent
    return None


# ---------------------------------------------------------------------------
# Cloud (retained for backwards compatibility)
# ---------------------------------------------------------------------------

def set_cloud(pts: np.ndarray) -> None:
    global _cloud
    _cloud = pts if pts.dtype == np.float32 else pts.astype(np.float32)


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
    """Load gates from disk. Handles both raw list and AutoGateDetector {gates:[...]} format."""
    if GATES_FILE.exists():
        try:
            raw = json.loads(GATES_FILE.read_text())
            if isinstance(raw, list):
                return raw
            if isinstance(raw, dict):
                return raw.get("gates", [])
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
