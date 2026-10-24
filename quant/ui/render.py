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

import re
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import logging

import pandas as pd
import streamlit as st

from quant import paths
from quant.data.database import read_only_connection
from quant.reporting.artifacts import (
    chart_tick_dates,
    compute_growth_view,
    compute_value_view_hover,
    format_chart_ticks,
    read_portfolio_total,
    read_regime,
    read_value_series,
)
from quant.ui import columns as COLS
from quant.ui import copy as C
from quant.ui import palette as P
from quant.ui.search import discovery_candidates, label_for, load_index, search

# Re-exported for the five pages (quant/ui/pages.py reads them via this module).
__all__ = [
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
# v10.8.8 (R-P1): the editor grid and the read-only Entry-check table are defined
# by ONE column-spec contract in quant.ui.columns. The grid holds ONLY the two
# read-only identity columns (Name, Instrument) and the four typed inputs; the
# index is hidden and every header is non-empty. Derived and live data live
# OUTSIDE the grid, in the Entry-check table.


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


def render_delta_badge(eur, pct, label: str = "") -> None:
    """Render the ONE delta badge (v10.8.8, R-P3).

    Calls :func:`quant.ui.copy.delta_badge` for the icon/color/text; no inline
    badge formatting exists anywhere else in the UI.
    """
    icon, color, text = C.delta_badge(eur, pct)
    prefix = f"{label} " if label else ""
    st.markdown(
        f'<span style="color:{color}; font-weight:600">'
        f"{prefix}{icon} {text}</span>",
        unsafe_allow_html=True)


def _last_sync_date():
    """The date of the last Confirm and save (newest position snapshot), or None."""
    try:
        with read_only_connection() as conn:
            row = conn.execute("SELECT MAX(snapshot_date) FROM position_snapshots"
                               ).fetchone()
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


def _num(value) -> float | None:
    """A float or None (never raises)."""
    if value is None:
        return None
    f = pd.to_numeric(value, errors="coerce")
    if f is None or f != f:
        return None
    return float(f)


def _clean_name(symbol: str) -> str:
    """The clean display name for a symbol (no ticker in parentheses; A1).

    Strips a trailing parenthesized ticker ("MSCI World Xetra (EUNL)" ->
    "MSCI World Xetra") and returns "" when no real name is known, so the Name
    cell never repeats the Instrument ticker (addendum A1).
    """
    from quant.data.names import display_name

    sym = str(symbol)
    name = str(display_name(sym) or "").strip()
    if not name or name.upper() == sym.upper():
        return ""
    base = sym.split(".")[0]
    for tok in (sym, base):
        if tok:
            name = re.sub(rf"\s*\({re.escape(tok)}\)\s*$", "", name,
                          flags=re.IGNORECASE).strip()
    if sym and sym.upper() in name.upper():
        return ""
    for tok in (sym, base):
        if tok and re.search(rf"\({re.escape(tok)}\)", name, re.IGNORECASE):
            return ""
    return name


def _entry_check_rows(portfolio, live=None) -> list[dict]:
    """The read-only Entry-check rows (v10.8.8, R-P1; v10.8.10).

    Catches a mistyped row BEFORE it is saved: per typed row, shares = (value -
    profit) / average cost, live value = shares x the latest EUR price, and
    Delta % = (typed - live) / typed * 100. Independent of the ledger, so it
    checks exactly what is on screen. ``live`` is unused (kept for callers).
    """
    from quant.data.currency import price_in_eur

    if portfolio is None or getattr(portfolio, "empty", True):
        return []
    rows: list[dict] = []
    for _, r in portfolio.iterrows():
        sym = str(r.get("Symbol", ""))
        typed = _num(r.get("Current_Value_EUR"))
        entry = _num(r.get("Avg_Entry_Price"))
        invested = (typed or 0.0) - (_num(r.get("Broker_PnL_EUR")) or 0.0)
        shares = invested / entry if entry else None
        price = price_in_eur(sym) if sym and shares else None
        live_v = round(shares * price, 2) if price else None
        delta = (typed - live_v) / typed * 100.0 if typed and live_v is not None else None
        rows.append({"Symbol": sym, "Shares": shares, "Typed": typed,
                     "Live": live_v, "Delta": delta})
    return rows


def _is_amber(delta) -> bool:
    """True when a |Delta| > 1% disagreement flags the row amber (R-P1)."""
    d = _num(delta)
    return d is not None and abs(d) > 1.0


def _render_entry_check(portfolio, live=None, as_of=None) -> None:
    """The read-only Entry-check table (v10.8.8, R-P1).

    ``Instrument | Shares (derived) | Typed value (EUR) | Live value (EUR) |
    Delta %``. An amber row highlight flags a |Delta| > 1% disagreement. This is
    the ONLY place live and typed meet; the editor grid holds no live data.
    """
    rows = _entry_check_rows(portfolio, live)
    if not rows:
        st.caption(C.ENTRY_CHECK_EMPTY)
        return
    df = pd.DataFrame(rows)

    def _amber(row) -> list[str]:
        style = (f"background-color: {P.WARNING}; color: white"
                 if _is_amber(row.get("Delta")) else "")
        return [style for _ in row]

    styler = df.style.apply(_amber, axis=1)
    st.dataframe(styler, width="stretch", hide_index=True,
                 column_config=COLS.check_column_config(),
                 column_order=COLS.check_column_order())
    st.caption(C.ENTRY_CHECK_CAPTION.format(date=C.fmt_date(as_of)))


def _render_broker_editor(portfolio, live=None) -> pd.DataFrame:
    """The editable broker statement table (T2; v10.8.8, R-P1).

    The grid contains ONLY the two read-only identity columns (Name, Instrument)
    and the four typed inputs (Avg cost / share, Value, Profit, Plan). Derived and
    live data live OUTSIDE the grid, in the read-only Entry-check table. The index
    is hidden; every header is non-empty and money columns carry their currency.

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
    # v10.8.8 (R-P1/A1): the read-only identity columns. Name is the clean display
    # name (no ticker in parentheses); Instrument is the broker ticker.
    base.insert(0, "Name", [_clean_name(s) for s in base["Symbol"].astype(str)])
    return st.data_editor(
        base, num_rows="dynamic", column_config=COLS.editor_column_config(),
        width="stretch", key="holdings", hide_index=True,
        column_order=COLS.editor_column_order(),
        disabled=COLS.editor_disabled(),
    )


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


def _render_value_chart(holdings) -> None:
    """Portfolio value chart: Value | Growth (T3; v10.8.6 B3/B4/B5).

    Value  = the absolute portfolio total from the valuation store, with one
             composed per-asset hover (total + each held asset; no cash, R-C1).
    Growth = normalized to 100: a bold portfolio line plus one line per held asset.
    The chart reads the store through the pure helpers in
    quant.reporting.artifacts and never recomputes shares/values inline (R1).
    Fallback for a database without a store: the snapshot+flow value series.
    """
    import plotly.graph_objects as go

    from quant.engine import mtm

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

    # Per-asset series (store only) feed the pure helpers (R1, no joins).
    # v10.8.7 (R-C1): cash is not an entity; there is no cash line or hover key.
    per_symbol: dict = {}
    if using_store:
        try:
            per_symbol = mtm.read_symbol_series() or {}
        except Exception:  # noqa: BLE001
            per_symbol = {}
    held = [str(h.get("symbol")) for h in holdings or [] if h.get("symbol")]
    if not held:
        held = list(per_symbol.keys())
    port_rows = [{"date": r["review_ts"], "value_eur": r["value_eur"]}
                 for r in df.to_dict("records")]

    fig = go.Figure()
    if growth:
        lines = compute_growth_view(port_rows, per_symbol, held)
        for sym, pts in lines.items():
            if sym == "portfolio" or not pts:
                continue
            fig.add_trace(go.Scatter(
                x=[p.date for p in pts], y=[p.value for p in pts], mode="lines",
                name=_display_name(sym), line=dict(width=1), opacity=0.7))
        port = lines.get("portfolio", [])
        fig.add_trace(go.Scatter(
            x=[p.date for p in port], y=[p.value for p in port], mode="lines",
            name="Portfolio", line=dict(width=3, color=P.ACCENT)))
        y_vals = [p.value for p in port]
    else:
        # Thin per-asset context lines carry NO hover; the portfolio line carries
        # the full breakdown so one tooltip lists total + each held asset.
        for sym, rows in per_symbol.items():
            keep = [(pd.to_datetime(r["date"]), float(r["value_eur"]))
                    for r in rows if r.get("value_eur") is not None]
            keep = [(x, y) for x, y in keep if x >= since]
            if len(keep) >= 2:
                fig.add_trace(go.Scatter(
                    x=[k[0] for k in keep], y=[k[1] for k in keep], mode="lines",
                    name=_display_name(sym), showlegend=False, hoverinfo="skip",
                    line=dict(width=0.5), opacity=0.35))
        hover_texts = []
        for r in df.to_dict("records"):
            day = pd.Timestamp(r["review_ts"]).date()
            hd = compute_value_view_hover(day, port_rows, per_symbol, held)
            parts = [f"Total {hd['total']:.2f} EUR"]
            for sym, val in hd.items():
                if sym == "total":
                    continue
                parts.append(f"{_display_name(sym)} {val:.2f} EUR")
            hover_texts.append("<br>".join(parts))
        fig.add_trace(go.Scatter(
            x=df["review_ts"], y=df["value_eur"], mode="lines", name="Portfolio",
            text=hover_texts, hoverinfo="text",
            line=dict(width=2, color=P.ACCENT)))
        y_vals = [float(v) for v in df["value_eur"] if v is not None]

    # v10.8.9 (R-F2): the benchmark toggle exists ONLY in Growth view. Its state
    # is preserved across view switches: a widget that is not rendered loses its
    # own key, so its value is shadowed in ``val_bench_on`` and restored on return.
    bench_on = bool(st.session_state.get("val_bench_on", False))
    if growth:
        st.session_state.setdefault("val_bench", bench_on)
        bench_on = st.checkbox(C.LABEL_BENCHMARK, key="val_bench")
        st.session_state["val_bench_on"] = bench_on
    if growth and bench_on:
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
    # v10.8.5 (D6): the in-plot "+X% since <date>" annotation duplicated the
    # header's "since first purchase" line; it is removed.
    yaxis = {}
    if y_vals:
        lo, hi = min(y_vals), max(y_vals)
        pad = (hi - lo) * 0.1 or max(1.0, abs(hi) * 0.01)
        yaxis = dict(range=[lo - pad, hi + pad])
    # v10.8.6 (B5): unique, readable date ticks (no duplicate dates on the axis).
    span_days = (df["review_ts"].max() - df["review_ts"].min()).days
    dates = df["review_ts"].tolist()
    if span_days < 3:
        xaxis = dict(tickformat="%H:%M", nticks=6)
    else:
        xaxis = dict(tickmode="array", tickvals=chart_tick_dates(dates, max_ticks=6),
                     ticktext=format_chart_ticks(dates, max_ticks=6))
    fig.update_layout(height=320, margin=dict(l=8, r=8, t=30, b=8),
                      hovermode="x unified", xaxis=xaxis, yaxis=yaxis,
                      showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                                  xanchor="left", x=0))
    st.plotly_chart(fig, width="stretch")
