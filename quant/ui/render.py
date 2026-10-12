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
    read_portfolio_total,
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
    # v10.8.4 (T2): the live read-only columns (from the valuation store).
    "Live value (EUR)": st.column_config.NumberColumn(
        "Live value (EUR)", format="%.2f"),
    "Live profit (EUR)": st.column_config.NumberColumn(
        "Live profit (EUR)", format="%.2f"),
    "Shares (derived)": st.column_config.NumberColumn(
        "Shares (derived)", format="%.6f"),
    "Check": st.column_config.TextColumn("Check"),
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

def _portfolio_signature(portfolio) -> str:
    """A stable signature of the saved table, to detect a DB update (T2)."""
    try:
        if portfolio is None or getattr(portfolio, "empty", True):
            return "empty"
        cols = [c for c in _EDIT_COLS if c in portfolio.columns]
        return "|".join(
            f"{r.get('Symbol','')}:{r.get('Current_Value_EUR','')}:"
            f"{r.get('Broker_PnL_EUR','')}:{r.get('Avg_Entry_Price','')}"
            for _, r in portfolio[cols].iterrows())
    except Exception:  # noqa: BLE001
        return "empty"


def _live_columns(base, live) -> None:
    """Append the read-only live Value / Profit / Shares and a mismatch check (T2).

    Broker-typed Value/Profit stay editable until the first confirm; the live
    columns come from the valuation store so an entry error is visible.
    """
    live = live or {}
    values, profits, shares, checks = [], [], [], []
    for sym in base["Symbol"].astype(str):
        h = live.get(sym)
        if not h:
            values.append(None)
            profits.append(None)
            shares.append(None)
            checks.append("")
            continue
        values.append(round(float(h.get("value_eur") or 0), 2))
        profits.append(round(float(h.get("profit_eur") or 0), 2))
        shares.append(round(float(h.get("shares") or 0), 6))
        checks.append("")
    # Mismatch >1% between the typed value and the live value (input sanity).
    for i, sym in enumerate(base["Symbol"].astype(str)):
        h = live.get(sym)
        if not h:
            continue
        try:
            typed = float(base["Current_Value_EUR"].iloc[i] or 0)
            live_v = float(h.get("value_eur") or 0)
            if typed > 0 and abs(typed - live_v) / typed > 0.01:
                checks[i] = f"differs by {abs(typed - live_v) / typed * 100:.0f}%"
        except Exception:  # noqa: BLE001
            pass
    base["Live value (EUR)"] = values
    base["Live profit (EUR)"] = profits
    base["Shares (derived)"] = shares
    base["Check"] = checks


def _render_broker_editor(portfolio, live=None) -> pd.DataFrame:
    """The editable broker statement table + the live read-only columns (T2).

    The autocomplete input says what it does; selecting a match appends an
    empty-value row and shows one helper line. No silent add. After a DB update
    (a refresh or a confirm) the widget key is invalidated so the editor
    re-syncs from the database instead of showing stale session state.
    """
    sig = _portfolio_signature(portfolio)
    if st.session_state.get("_editor_sig") != sig:
        st.session_state["_editor_sig"] = sig
        st.session_state.pop("holdings", None)
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
    # v10.8.4 (T2): append the live read-only Value/Profit/Shares + mismatch check.
    _live_columns(base, live)
    return st.data_editor(
        base, num_rows="dynamic", column_config=_COLUMN_CONFIG,
        width="stretch", key="holdings",
        disabled=["Name", "Live value (EUR)", "Live profit (EUR)",
                  "Shares (derived)", "Check"],
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


def _per_asset_series(since) -> dict:
    """{symbol: DataFrame(date, value_eur)} from the valuation store (T3)."""
    try:
        from quant.engine import mtm

        raw = mtm.read_symbol_series()
    except Exception:  # noqa: BLE001
        return {}
    out: dict = {}
    for sym, rows in (raw or {}).items():
        s = pd.DataFrame(rows)
        if s.empty:
            continue
        s["t"] = pd.to_datetime(s["date"], errors="coerce")
        s = s.dropna(subset=["t"]).sort_values("t")
        if since is not None:
            s = s[s["t"] >= since]
        if len(s) >= 2:
            out[str(sym)] = s
    return out


def _cash_frame(since) -> pd.DataFrame:
    try:
        from quant.engine import mtm

        rows = mtm.cash_series()
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    s = pd.DataFrame(rows)
    if s.empty:
        return s
    s["t"] = pd.to_datetime(s["date"], errors="coerce")
    s = s.dropna(subset=["t"]).sort_values("t")
    if since is not None:
        s = s[s["t"] >= since]
    return s


def _render_value_chart(history, holdings) -> None:
    """Portfolio value chart: Value | Growth (T3, v10.8.4).

    Value  = the absolute portfolio total from the valuation store, with cash and
             a per-asset hover.
    Growth = normalized-to-100: a bold portfolio line plus one line per held
             asset from its first held day. The benchmark stays an optional
             overlay. The x-axis shows at most six unique readable dates.
    """
    import plotly.graph_objects as go

    total = read_portfolio_total()
    using_store = total is not None and not total.empty and len(total) >= 2
    if using_store:
        df = total.copy()
        df["review_ts"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        series = read_value_series()
        if series is not None and len(series) >= 2:
            df = series.copy()
            df["review_ts"] = pd.to_datetime(df["date"], errors="coerce")
        elif history is not None and len(history) >= 3:
            df = history.copy()
            df["review_ts"] = pd.to_datetime(df["review_ts"], errors="coerce")
        else:
            st.info(C.CHART_BUILDING)
            return
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
    since = df["review_ts"].min()

    fig = go.Figure()
    if growth:
        port = _rebase(df["value_eur"])
        # v10.8.4 (T3): bold portfolio line + one line per held asset.
        for sym, s in _per_asset_series(since).items():
            fig.add_trace(go.Scatter(
                x=s["t"], y=_rebase(s["value_eur"]), mode="lines",
                name=_display_name(sym), line=dict(width=1), opacity=0.7))
        fig.add_trace(go.Scatter(
            x=df["review_ts"], y=port, mode="lines", name="Portfolio",
            line=dict(width=3, color=P.ACCENT), fill=None))
    else:
        port = df["value_eur"]
        # Per-asset hover helpers (thin, no legend) so hovering a date lists them.
        for sym, s in _per_asset_series(since).items():
            fig.add_trace(go.Scatter(
                x=s["t"], y=s["value_eur"], mode="lines",
                name=_display_name(sym), showlegend=False,
                line=dict(width=0.5), opacity=0.35))
        if using_store:
            cash = _cash_frame(since)
            if len(cash) >= 2:
                fig.add_trace(go.Scatter(
                    x=cash["t"], y=cash["cash_eur"], mode="lines", name=C.CLASS_CASH,
                    line=dict(width=1, dash="dot", color=P.BASELINE)))
        fig.add_trace(go.Scatter(
            x=df["review_ts"], y=port, mode="lines", name="Portfolio",
            line=dict(width=2, color=P.ACCENT), fill=None))

    if st.checkbox(C.LABEL_BENCHMARK, value=False, key="val_bench"):
        bench = q("SELECT Date AS d, Close FROM market_history WHERE Symbol = ? "
                  "ORDER BY Date ASC", [C.BENCHMARK_SYMBOL])
        if not bench.empty:
            bench["d"] = pd.to_datetime(bench["d"], errors="coerce")
            bench = bench.dropna(subset=["d"])
            bench = bench[bench["d"] >= since]
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
        fig.add_annotation(**_annotation_kwargs(ann))
    vals = [float(v) for v in port if v is not None]
    yaxis = {}
    if vals:
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.1 or max(1.0, abs(hi) * 0.01)
        yaxis = dict(range=[lo - pad, hi + pad])
    # v10.8.4 (T3): at most ~6 unique, readable date ticks (no duplicates).
    span_days = (df["review_ts"].max() - df["review_ts"].min()).days
    tickfmt = "%H:%M" if span_days < 3 else "%d %b"
    fig.update_layout(height=320, margin=dict(l=8, r=8, t=30, b=8),
                      hovermode="x unified",
                      xaxis=dict(tickformat=tickfmt, nticks=6), yaxis=yaxis,
                      showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                                  xanchor="left", x=0))
    st.plotly_chart(fig, width="stretch")
