"""
dashboard.py — Multipage navigation entry (v10.8.2).

Intent: st.Page must reference FILE-based pages so tests can drive navigation
via AppTest.switch_page, so the renderers live in quant/ui/pages.py and the thin
scripts under quant/pages/ call them. This module is the entry: it runs the
startup hooks then st.navigation over the five pages.

Invariants: the sidebar shows the page names only (no name/tagline/version
repeat); page order is Portfolio, Update holdings, History, Full analysis,
Settings.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import streamlit as st

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
# v10.8.2 (section 4): the final five pages, in order.
_NAV = [
    ("pages/portfolio.py", "Portfolio", True),
    ("pages/update.py", "Update holdings", False),
    ("pages/history.py", "History", False),
    ("pages/analysis.py", "Full analysis", False),
    ("pages/settings.py", "Settings", False),
]


def _startup_user_data() -> None:
    """v10.8.1 (A2): migrate legacy repo state and seed the per-user data dir.

    Copy-only and idempotent; never raises. Runs once per session.
    """
    if st.session_state.get("_user_data_ready"):
        return
    try:
        from quant.data.bootstrap import seed_user_data

        seed_user_data()
    except Exception:  # noqa: BLE001
        pass
    st.session_state["_user_data_ready"] = True


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
    """Run the startup hooks and the multipage navigation (page names only)."""
    _startup_user_data()
    _startup_backfill()
    _startup_staleness()
    pages = [st.Page(path, title=title, default=default)
             for path, title, default in _NAV]
    st.navigation(pages).run()


main()
