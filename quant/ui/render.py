"""
render.py — shared renderers for the five pages (v10.8.2, one workflow).

Intent: st.Page needs FILE-based pages so AppTest.switch_page can drive them, so
the thin scripts under quant/pages/ call the five renderers in quant/ui/pages.py.
This module holds only the helpers those pages share: the read-only query
helpers, the editable broker table, and the value chart. All user-facing strings
come from quant.ui.copy.

Dependencies: streamlit, pandas, plotly, quant.*, quant.ui.copy, quant.ui.search.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import logging

import pandas as pd
import streamlit as st

from quant import paths
from quant.data.database import read_only_connection
from quant.reporting.artifacts import (
    read_actions,
    read_history,
    read_regime,
    read_value_series,
)
from quant.ui import copy as C
from quant.ui import palette as P
from quant.ui.search import discovery_candidates, label_for, load_index, search

# Re-exported for the five pages (quant/ui/pages.py reads them via this module).
__all__ = [
    "read_actions",
    "read_history",
    "read_regime",
    "read_value_series",
    "label_for",
    "q",
    "load_portfolio",
    "latest_bar_date",
]

logger = logging.getLogger(__name__)

_EDIT_COLS = ["Symbol", "Avg_Entry_Price", "Current_Value_EUR", "Broker_PnL_EUR"]
# v10.8.2 (B3): the optional savings-plan rate, saved by the same Confirm.
_PLAN_COL = "Plan_EUR_month"
_COLUMN_CONFIG = {
    # v10.7.3 (Part 2.2): the broker statement shows the company name (read-only).
    "Name": st.column_config.TextColumn("Name"),
    "Symbol": st.column_config.TextColumn(C.COLUMN_HEADERS["Symbol"]),
    "Avg_Entry_Price": st.column_config.NumberColumn(
        C.COLUMN_HEADERS["Avg_Entry_Price"], format="%.2f"),
    "Current_Value_EUR": st.column_config.NumberColumn(
        C.COLUMN_HEADERS["Current_Value_EUR"], format="%.2f"),
    "Broker_PnL_EUR": st.column_config.NumberColumn(
        C.COLUMN_HEADERS["Broker_PnL_EUR"], format="%.2f"),
    _PLAN_COL: st.column_config.NumberColumn(
        C.COLUMN_HEADERS[_PLAN_COL], format="%.2f"),
}


# ── Read-only helpers (spec 4.1: short-lived, always closed) ──────────────────

def q(sql: str, params: list | None = None) -> pd.DataFrame:
    """Run a read-only query. Returns an empty frame on any failure."""
    try:
        with read_only_connection() as conn:
            return conn.execute(sql, params or []).df()
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def load_portfolio() -> pd.DataFrame:
    """Load the broker-synced portfolio (empty frame on failure)."""
    try:
        from quant.portfolio.portfolio import load_portfolio as _lp
        return _lp(paths.DATA_PORTFOLIO)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _display_name(symbol: str) -> str:
    """Company name for a symbol (v10.7.3, Part 2): registry -> cache -> symbol."""
    try:
        from quant.data.names import display_name

        return display_name(symbol)
    except Exception:  # noqa: BLE001
        return str(symbol)


def latest_bar_date() -> str:
    """Return the latest market bar date as a string (empty when absent)."""
    df = q("SELECT MAX(Date) AS d FROM market_history")
    if df.empty or df["d"].iloc[0] is None:
        return ""
    return str(df["d"].iloc[0])


def _last_sync_date():
    """The newest holdings_meta sync_date, or None (v10.7.3, Part 3.2)."""
    try:
        with read_only_connection() as conn:
            row = conn.execute("SELECT MAX(sync_date) FROM holdings_meta").fetchone()
        return row[0] if row else None
    except Exception:  # noqa: BLE001
        return None


# ── The editable broker table (Update holdings) ───────────────────────────────

def _render_broker_editor(portfolio) -> pd.DataFrame:
    """The editable broker statement table + the add-row (v10.7.3, Part 1.6).

    The autocomplete input says what it does; selecting a match appends an
    empty-value row and shows one helper line. No silent add.
    """
    st.caption(C.COST_BASIS_HINT)
    query = st.text_input(
        C.PLACEHOLDER_ADD_HOLDING, key="add_q",
        placeholder=C.PLACEHOLDER_ADD_HOLDING, label_visibility="collapsed",
    )
    if query:
        results = search(load_index(), query, 10)
        # H3.5 (F5): also offer discovery-universe instruments (universe_master
        # rows not yet tracked); selecting one adds a held row -> enters W on
        # save, fetched at the next update.
        disc = discovery_candidates(query, 10)
        sym_by_label: dict[str, str] = {}
        labels: list[str] = []
        for r in results:
            sym_by_label[r["label"]] = r["symbol"]
            labels.append(r["label"])
        for r in disc:
            lbl = C.NOT_TRACKED_LABEL.format(label=r["label"])
            if lbl not in sym_by_label:
                sym_by_label[lbl] = r["symbol"]
                labels.append(lbl)
        if labels:
            def _add_selected() -> None:
                sym = sym_by_label.get(st.session_state.get("add_choice"))
                if not sym:
                    return
                st.session_state.setdefault("_extra", []).append({
                    "Symbol": sym, "Avg_Entry_Price": 0.0,
                    "Current_Value_EUR": 0.0, "Broker_PnL_EUR": 0.0,
                })
                st.session_state["_just_added"] = True

            st.selectbox("Matches", labels, key="add_choice", on_change=_add_selected)
            if st.session_state.get("_just_added"):
                st.caption(C.HELP_ADD_ROW)
        else:
            st.caption(C.EMPTY_NO_MATCHES.format(query=query))

    base = portfolio[_EDIT_COLS] if not portfolio.empty and \
        set(_EDIT_COLS).issubset(portfolio.columns) else pd.DataFrame(columns=_EDIT_COLS)
    base = base.copy()
    # v10.8.2 (B3): the optional savings-plan rate, saved by the same Confirm.
    if _PLAN_COL in portfolio.columns:
        base[_PLAN_COL] = portfolio[_PLAN_COL].values
    else:
        base[_PLAN_COL] = pd.NA
    extra = st.session_state.get("_extra", [])
    if extra:
        base = pd.concat([base, pd.DataFrame(extra)], ignore_index=True)
    # v10.7.3 (Part 2.2): show the company name next to the symbol (read-only).
    base.insert(0, "Name", [_display_name(s) for s in base["Symbol"].astype(str)])
    return st.data_editor(
        base, num_rows="dynamic", column_config=_COLUMN_CONFIG,
        width="stretch", key="holdings", disabled=["Name"],
    )


def _plain_save_error(exc: Exception) -> str:
    """One specific line: what was not saved and why (v10.8.1, A2)."""
    msg = str(exc).lower()
    if "lock" in msg or "conflict" in msg or "busy" in msg:
        return ("Could not save: the daily check or another tab is writing. "
                "Try again in a minute.")
    return ("Could not save your holdings. Nothing was written; "
            "your edit is still on screen.")


def _rows_differ(edited, saved) -> bool:
    """True when the edited table differs from the saved one (v10.8.1, A2)."""
    def _norm(df):
        if df is None or getattr(df, "empty", True):
            return []
        cols = [c for c in _EDIT_COLS if c in df.columns]
        return sorted(
            tuple(str(r.get(c, "")) for c in cols) for _, r in df[cols].iterrows())

    try:
        return _norm(edited) != _norm(saved)
    except Exception:  # noqa: BLE001
        return False


# ── Value chart helpers (spec 3.1; pure where possible) ───────────────────────
_RANGE_DAYS = {"1M": 31, "3M": 92, "1Y": 365, "Max": None}


def _filter_range(df: pd.DataFrame, rng: str) -> pd.DataFrame:
    """Return rows of df within the selected range (days back from the last)."""
    days = _RANGE_DAYS.get(rng or "Max")
    if not days or df.empty:
        return df
    ts = pd.to_datetime(df["review_ts"], errors="coerce")
    cutoff = ts.max() - pd.Timedelta(days=days)
    return df[ts >= cutoff]


def _rebase(values):
    """Rebase a numeric series to 100 at its first point (Growth mode)."""
    base = values.iloc[0] if hasattr(values, "iloc") else values[0]
    if not base:
        return values
    return values / base * 100.0


def _range_annotation(df: pd.DataFrame, growth: bool = False) -> str:
    """`+4.2% since 1 Jun 2026 (34.80 EUR)` from the range endpoints.

    H3.8 (M3): Growth mode is percent-only (no EUR parenthetical).
    """
    if df.empty or len(df) < 2:
        return ""
    first, last = df.iloc[0], df.iloc[-1]
    if not first["value_eur"]:
        return ""
    pct = (last["value_eur"] / first["value_eur"] - 1.0) * 100.0
    sign = "+" if pct >= 0 else ""
    if growth:
        return C.CHART_SINCE_PCT.format(sign=sign, pct=f"{pct:.1f}",
                                        date=C.fmt_date(first["review_ts"]))
    abs_ = last["value_eur"] - first["value_eur"]
    return C.CHART_SINCE.format(sign=sign, pct=f"{pct:.1f}",
                                date=C.fmt_date(first["review_ts"]), amount=f"{abs_:.2f}")


def _axis_tickformat(df) -> str:
    """H3.7 (L5): sub-3-day ranges label by hh:mm, else by day."""
    try:
        span = (df["review_ts"].max() - df["review_ts"].min()).days
    except Exception:  # noqa: BLE001
        return "%d %b"
    return "%H:%M" if span < 3 else "%d %b"


def _annotation_kwargs(text: str) -> dict:
    """H3.7 (L5): the in-plot annotation, in the top margin on a white box."""
    return dict(xref="paper", yref="paper", x=0.0, y=1.0, xanchor="left",
                yanchor="bottom", text=text, showarrow=False, align="left",
                font=dict(size=12, color=P.ANNOTATION_TEXT),
                bgcolor=P.ANNOTATION_BG)


def _render_value_chart(history, holdings) -> None:
    """Portfolio value chart: range selector, baseline, annotation, Value|Growth.

    v10.8.2 (section 5): the chart reads the append-only value series, which runs
    to the latest price date and never draws a zero for an unpriced day. It falls
    back to the review history only when the series has fewer than two points.
    """
    import plotly.graph_objects as go

    series = read_value_series()
    if series is not None and len(series) >= 2:
        df = series.copy()
        df["review_ts"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        if history is None or len(history) < 3:
            st.info(C.CHART_BUILDING)
            return
        df = history.copy()
        df["review_ts"] = pd.to_datetime(df["review_ts"], errors="coerce")
    df = df.dropna(subset=["review_ts"]).sort_values("review_ts")
    rng = st.segmented_control(C.LABEL_RANGE, list(_RANGE_DAYS),
                               default="Max", key="val_range") or "Max"
    df = _filter_range(df, rng)
    if len(df) < 2:
        st.info(C.CHART_BUILDING)
        return
    mode = st.segmented_control(C.LABEL_VIEW, [C.VALUE, C.GROWTH_VIEW],
                                default=C.VALUE, key="val_mode") or C.VALUE
    growth = mode == C.GROWTH_VIEW

    fig = go.Figure()
    port = _rebase(df["value_eur"]) if growth else df["value_eur"]
    # v10.7.0 (B8): one clean line, no area fill.
    fig.add_trace(go.Scatter(
        x=df["review_ts"], y=port, mode="lines", name="Portfolio",
        line=dict(width=2, color=P.ACCENT), fill=None))

    # v10.7.3 (Part 7.2): exactly one benchmark line, in either view. In Value
    # view it is scaled to the portfolio's start value so it overlays in EUR.
    if st.checkbox(C.LABEL_BENCHMARK, value=False, key="val_bench"):
        bench = q("SELECT Date AS d, Close FROM market_history WHERE Symbol = ? "
                  "ORDER BY Date ASC", [C.BENCHMARK_SYMBOL])
        if not bench.empty:
            bench["d"] = pd.to_datetime(bench["d"], errors="coerce")
            bench = bench.dropna(subset=["d"])
            bench = bench[bench["d"] >= df["review_ts"].min()]
            if len(bench) >= 2:
                close = bench["Close"]
                if growth:
                    y = _rebase(close)
                else:
                    b0 = float(close.iloc[0]) or 1.0
                    y = close / b0 * float(df["value_eur"].iloc[0])
                fig.add_trace(go.Scatter(
                    x=bench["d"], y=y, mode="lines",
                    name=_display_name(C.BENCHMARK_SYMBOL),
                    line=dict(width=1.5, color=P.BENCHMARK)))

    base = 100.0 if growth else float(df["value_eur"].iloc[0])
    fig.add_hline(y=base, line_dash="dot", line_color=P.BASELINE)
    ann = _range_annotation(df, growth)
    if ann:
        fig.add_annotation(**_annotation_kwargs(ann))     # H3.7 (L5)
    # v10.7.0 (B8): the axis never starts at zero; pad the visible range.
    vals = [float(v) for v in port if v is not None]
    yaxis = {}
    if vals:
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.1 or max(1.0, abs(hi) * 0.01)
        yaxis = dict(range=[lo - pad, hi + pad])
    # v10.7.3 (Part 7.1): fixed 320 px height, full container width.
    fig.update_layout(height=320, margin=dict(l=8, r=8, t=30, b=8),
                      xaxis=dict(tickformat=_axis_tickformat(df)), yaxis=yaxis,
                      showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                                  xanchor="left", x=0))
    st.plotly_chart(fig, width="stretch")
