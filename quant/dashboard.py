"""
dashboard.py — Multipage navigation entry (v10.5.3, R2).

Intent: st.Page must reference FILE-based pages so tests can drive navigation
via AppTest.switch_page, so the renderers live in quant/ui/render.py and the thin
scripts under quant/pages/ call them. This module is the entry: it draws the
sidebar contract (name, tagline, version) then st.navigation over four pages.

Invariants: sidebar shows only name, tagline, version, then page nav; page order
is Today, Portfolio, Explore, Settings; navigation labels equal quant.ui.copy.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import streamlit as st

from quant import __version__
from quant.ui import copy as C

st.set_page_config(page_title="Quant-AI", layout="centered")

# (script path, title, default) in the fixed order.
_NAV = [
    ("pages/today.py", C.PAGE_TODAY, True),
    ("pages/portfolio.py", C.PAGE_PORTFOLIO, False),
    ("pages/explore.py", C.PAGE_EXPLORE, False),
    ("pages/settings.py", C.PAGE_SETTINGS, False),
]


def _startup_backfill() -> None:
    """B1/B4: one-time startup backfill for a pre-existing DB. Documented
    write-enabled connection exception (app startup); later reads stay read-only.
    """
    if st.session_state.get("_names_ensured"):
        return
    try:
        from quant.data.names import ensure_display_names

        ensure_display_names()
    except Exception:  # noqa: BLE001
        pass
    st.session_state["_names_ensured"] = True


def main() -> None:
    _startup_backfill()
    st.sidebar.title("Quant-AI")
    st.sidebar.caption("Daily portfolio management")
    st.sidebar.caption(f"Version {__version__}")
    pages = [st.Page(path, title=title, default=default)
             for path, title, default in _NAV]
    st.navigation(pages).run()


main()
