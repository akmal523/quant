"""
dashboard.py — Multipage navigation entry (v10.5.3, R2).

Intent: st.Page must reference FILE-based pages so tests can drive navigation
via AppTest.switch_page, so the renderers live in quant/ui/render.py and the thin
scripts under quant/pages/ call them. This module is the entry: it draws the
sidebar contract (name, tagline, version) then st.navigation over four pages.

Invariants: sidebar shows only name, tagline, version, then page nav; page order
is Overview, Monthly decision, My holdings, Find investments, Settings;
navigation labels equal quant.ui.copy.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import streamlit as st

from quant import __version__
from quant.ui import copy as C

st.set_page_config(page_title="Quant-AI", layout="centered")

# v10.8.0 (Phase 3): responsive layout. Tables scroll horizontally instead of
# clipping at 1280px; the layout collapses cleanly at 390px (phone).
st.markdown(
    """
    <style>
    [data-testid="stDataFrame"] { overflow-x: auto; }
    [data-testid="stDataFrame"] > div { min-width: 0; }
    @media (max-width: 420px) {
      .block-container { padding-left: 0.6rem; padding-right: 0.6rem; }
      [data-testid="stDataFrame"] { font-size: 0.8rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# (script path, title, default) in the fixed order.
_NAV = [
    ("pages/today.py", C.PAGE_TODAY, True),
    ("pages/monthly.py", C.PAGE_MONTHLY, False),
    ("pages/portfolio.py", C.PAGE_PORTFOLIO, False),
    ("pages/explore.py", C.PAGE_EXPLORE, False),
    ("pages/tax.py", C.PAGE_TAX, False),
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


def _startup_staleness() -> None:
    """App-open fallback (v10.7.0, Section 3.5): heal staleness on open.

    If the last successful daily run is older than the previous trading day and
    the runner lock is free, trigger a quiet background refresh and show a
    one-line status. If the lock is busy, show the status line only.
    """
    if st.session_state.get("_staleness_checked"):
        return
    st.session_state["_staleness_checked"] = True
    try:
        from quant.engine import daily
        from quant.ui import runner

        status = daily.staleness_status()
        if not status:
            return
        st.info(status)
        if not runner.is_running():
            import subprocess
            import sys

            subprocess.Popen(
                [sys.executable, "-m", "quant.cli", "daily"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    _startup_backfill()
    _startup_staleness()
    st.sidebar.title("Quant-AI")
    st.sidebar.caption("Daily portfolio management")
    st.sidebar.caption(f"Version {__version__}")
    pages = [st.Page(path, title=title, default=default)
             for path, title, default in _NAV]
    st.navigation(pages).run()


main()
