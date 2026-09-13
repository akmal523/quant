"""
explore.py — thin page script (v10.5.3, R2). Calls the shared renderer.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from quant.ui.render import page_explore

page_explore()
