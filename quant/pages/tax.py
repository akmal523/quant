"""
tax.py — thin page script (v10.7.6, Part 2). Calls the shared renderer.
"""
from __future__ import annotations  # noqa: I001 - matches the thin-page pattern

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from quant.ui.render import page_tax

page_tax()
