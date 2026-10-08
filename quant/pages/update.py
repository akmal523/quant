"""update.py — Update holdings (v10.8.2). Calls the shared renderer."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from quant.ui.pages import page_update

page_update()
