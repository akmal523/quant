"""
dashboard.py — Local daily portfolio manager (v10.5.1, spec 6).

Intent: Quant-AI is a daily portfolio manager, not a trading terminal. One
snapshot per day after market close; plain-language advice; the user acts in
the broker app. Four pages, each with one job (P12):
  Today     — decides
  Portfolio — edits
  Explore   — explains
  Settings  — maintains

Rules: single column (P9), no internal identifiers (P2), provenance only in the
glossary/methodology (P3), no silent defaults (P4), verb-first buttons (P5),
plain errors + View log (P6), diagnostics collapsed (P7), one accent color and
semantic status only (P8), units inline and human dates (P10), friendly names
first (P11). All strings come from quant.ui.copy (P14).

Reads use short-lived read-only connections (spec 4.1) so a refresh subprocess
can take the write lock. This app NEVER writes the database.

Run:  quant dash      (from the repo root)
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from quant import paths
import json
import os
import pandas as pd
import streamlit as st

from quant import __version__
from quant.config import STALE_DATA_DAYS, RISK_PROFILES
from quant.data.database import read_only_connection
from quant.execution.taxonomy import (
    resolve_broker, classify_instrument, get_structure,
    INVERSE_STRUCTURE, LEVERAGED_STRUCTURE,
)
from quant.portfolio.account import load_account, save_account, AccountState
from quant.portfolio.editor import validate_positions, save_portfolio
from quant.portfolio.history import load_history
from quant.portfolio.cash_rate import current_cash_apy
from quant.reporting.actions import build_actions
from quant.reporting.artifacts import latest_run
from quant.ui import copy as C
from quant.ui import runner
from quant.ui.search import load_index, search

st.set_page_config(page_title="Quant-AI", layout="centered")

_EDIT_COLS = ["Symbol", "Avg_Entry_Price", "Current_Value_EUR", "Broker_PnL_EUR"]
_COLUMN_CONFIG = {
    "Symbol": st.column_config.TextColumn(C.COLUMN_HEADERS["Symbol"]),
    "Avg_Entry_Price": st.column_config.NumberColumn(
        C.COLUMN_HEADERS["Avg_Entry_Price"], format="%.2f"),
    "Current_Value_EUR": st.column_config.NumberColumn(
        C.COLUMN_HEADERS["Current_Value_EUR"], format="%.2f"),
    "Broker_PnL_EUR": st.column_config.NumberColumn(
        C.COLUMN_HEADERS["Broker_PnL_EUR"], format="%.2f"),
}


# ── Read-only helpers (spec 4.1: short-lived, always closed) ──────────────────

def q(sql: str, params=None) -> pd.DataFrame:
    """Run a read-only query. Returns an empty frame on any failure."""
    try:
        with read_only_connection() as conn:
            return conn.execute(sql, params or []).df()
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def load_portfolio() -> pd.DataFrame:
    try:
        from quant.portfolio.portfolio import load_portfolio as _lp
        return _lp(paths.DATA_PORTFOLIO)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def load_audit() -> pd.DataFrame:
    path = os.path.join(str(paths.OUTPUTS_DIR), "portfolio_audit.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def load_metrics() -> dict:
    run_dir = latest_run()
    if not run_dir:
        return {}
    path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def latest_bar_date() -> str:
    df = q("SELECT MAX(Date) AS d FROM market_history")
    if df.empty or df["d"].iloc[0] is None:
        return ""
    return str(df["d"].iloc[0])


# ── Sidebar (P8b: name, tagline, version, nav only) ───────────────────────────

def render_sidebar() -> str:
    st.sidebar.title("Quant-AI")
    st.sidebar.caption("Daily portfolio management")
    st.sidebar.caption(f"Version {__version__}")
    return st.sidebar.radio(
        "Navigate", [C.PAGE_TODAY, C.PAGE_PORTFOLIO, C.PAGE_EXPLORE, C.PAGE_SETTINGS]
    )


# ── Shared renderers ──────────────────────────────────────────────────────────

def render_action_cards(actions: list[dict]) -> None:
    """Render action cards from the catalog (spec 3.2)."""
    if not actions:
        st.info(C.EMPTY_NOTHING_TO_DO)
        return
    for a in actions:
        if a["blocked"]:
            st.warning(C.ACTION_BLOCKED.format(symbol=a["symbol"]))
            continue
        target = a.get("target", "").rstrip("%") or "?"
        pct = abs(float(str(a.get("drift", "0")).rstrip("%") or 0))
        if a["action"] == "BUY MORE":
            st.write(C.ACTION_ADD.format(
                amount=f"{a['amount_eur']:.0f}", symbol=a["symbol"],
                name=a["symbol"], pct=f"{pct:.0f}", target=target))
        else:
            st.write(C.ACTION_SELL.format(
                amount=f"{a['amount_eur']:.0f}", symbol=a["symbol"],
                pct=f"{pct:.0f}", target=target))


def status_word_for(recommendation: str) -> str:
    rec = str(recommendation or "")
    if rec.startswith("BUY"):
        return "Add"
    if rec.startswith("SELL"):
        return "Trim"
    return "On track"


# ── Page: Today (P4) ──────────────────────────────────────────────────────────

def page_today() -> None:
    st.title(C.PAGE_TODAY)

    portfolio = load_portfolio()
    history = load_history()
    if portfolio.empty and history.empty:
        st.write("Start here:")
        for i, step in enumerate(C.FIRST_RUN_STEPS, 1):
            st.write(f"{i}. {step}")
        return

    # 1. Header line + market trend.
    if not history.empty:
        prepared = C.fmt_ts(history.iloc[-1]["review_ts"])
        bar = C.fmt_date(latest_bar_date())
        st.write(f"Review of {bar} close, prepared {prepared}.")
    metrics = load_metrics()
    regime = metrics.get("market_regime")
    st.write(f"Market trend: {C.regime_word(regime)}." if regime else C.EMPTY_REGIME)

    # 2. Portfolio value chart.
    st.subheader(C.SEC_PORTFOLIO_VALUE)
    if len(history) >= 2:
        import plotly.graph_objects as go
        fig = go.Figure(go.Scatter(
            x=history["review_ts"], y=history["value_eur"],
            mode="lines", fill="tozeroy", line=dict(width=2)))
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=0, b=0),
                          yaxis_title="EUR", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(C.EMPTY_VALUE_CHART)

    # 3. Where your money is (donut).
    st.subheader(C.SEC_WHERE_MONEY)
    account = load_account()
    if not portfolio.empty:
        import plotly.graph_objects as go
        values = list(portfolio["Amount_EUR"])
        labels = list(portfolio["Symbol"])
        if account.cash_is_set and account.cash_eur:
            values.append(account.cash_eur)
            labels.append("Cash")
        fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.55,
                               textinfo="label+percent"))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # 4. Holdings table (four columns).
    audit = load_audit()
    if not audit.empty:
        rows = []
        for _, r in audit.iterrows():
            rows.append({
                "Holding": r.get("Symbol", ""),
                "Value": C.fmt_eur(float(r.get("Value_EUR", 0) or 0)),
                "Share vs target": f"{str(r.get('Current_Weight', '')).rstrip('%')} / "
                                   f"{str(r.get('Target_Weight', '')).rstrip('%')}",
                "Status": status_word_for(r.get("Recommendation", "")),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # 5. What to do today.
    st.subheader(C.SEC_WHAT_TO_DO)
    actions = build_actions(audit)
    render_action_cards(actions)

    # 6. Needs attention first (blockers only, hidden when empty).
    blockers = [a for a in actions if a["blocked"]]
    if blockers:
        st.subheader(C.SEC_NEEDS_ATTENTION)
        for b in blockers:
            st.warning(C.ACTION_BLOCKED.format(symbol=b["symbol"]))
            st.button(C.BTN_FIX_IN_PORTFOLIO, key=f"fix_{b['symbol']}")


# ── Page: Portfolio (P5) ──────────────────────────────────────────────────────

def page_portfolio() -> None:
    st.title(C.PAGE_PORTFOLIO)
    st.write(C.HELP_BROKER_VALUES)

    # Autocomplete add-row.
    query = st.text_input("Type a name, symbol or ISIN", key="add_q")
    if query:
        results = search(load_index(), query, 10)
        if results:
            labels = [r["label"] for r in results]
            choice = st.selectbox("Matches", labels, key="add_choice")
            if st.button("Add to holdings", key="add_btn"):
                sym = next(r["symbol"] for r in results if r["label"] == choice)
                st.session_state.setdefault("_extra", []).append({
                    "Symbol": sym, "Avg_Entry_Price": 0.0,
                    "Current_Value_EUR": 0.0, "Broker_PnL_EUR": 0.0,
                })
        else:
            st.caption(C.EMPTY_NO_NEWS.format(name=query))

    # Holdings editor.
    portfolio = load_portfolio()
    base = portfolio[_EDIT_COLS] if not portfolio.empty and \
        set(_EDIT_COLS).issubset(portfolio.columns) else pd.DataFrame(columns=_EDIT_COLS)
    extra = st.session_state.get("_extra", [])
    if extra:
        base = pd.concat([base, pd.DataFrame(extra)], ignore_index=True)
    edited = st.data_editor(
        base, num_rows="dynamic", column_config=_COLUMN_CONFIG,
        width="stretch", key="holdings",
    )

    # Account block.
    st.subheader("Account")
    account = load_account()
    cash_val = account.cash_eur if account.cash_is_set else 0.0
    cash = st.number_input("Cash (EUR)", min_value=0.0, value=float(cash_val), step=10.0)
    profile = st.radio(
        "Risk profile", list(RISK_PROFILES.keys()),
        index=list(RISK_PROFILES.keys()).index(account.risk_profile),
        captions=[C.HELP_PROFILE_CONSERVATIVE, C.HELP_PROFILE_BALANCED,
                  C.HELP_PROFILE_AGGRESSIVE],
        format_func=lambda p: p.capitalize(),
    )
    st.caption(C.HELP_CASH_APY.format(apy=f"{current_cash_apy()*100:.2f}"))

    # Buttons.
    registry = q("SELECT symbol FROM asset_registry")
    universe = set(registry["symbol"].astype(str)) if not registry.empty else set()

    def _save_inputs() -> None:
        cleaned, warnings, errors = validate_positions(edited, universe)
        for w in warnings:
            st.warning(w)
        if errors:
            for e in errors:
                st.error(e)
            return
        save_portfolio(cleaned, paths.DATA_PORTFOLIO)
        save_account(AccountState(account.base_currency, float(cash), profile, True))
        st.session_state["_extra"] = []

    if st.button(C.BTN_SAVE_AND_REVIEW, type="primary", width="stretch"):
        _save_inputs()
        _run_with_progress(runner.SAVE_AND_REVIEW)
        st.button(C.BTN_OPEN_TODAY, key="open_today")
    if st.button(C.BTN_SAVE_ONLY, width="stretch"):
        _save_inputs()
        st.success("Saved. The next review will use these values.")

    # Broker registry (read-only, collapsed).
    with st.expander("Broker registry"):
        st.caption(C.HELP_ISIN_SCRIPTED)
        broker = pd.read_csv(paths.DATA_BROKER_REGISTRY) if os.path.exists(
            paths.DATA_BROKER_REGISTRY) else pd.DataFrame()
        st.dataframe(broker, width="stretch", hide_index=True)


def _run_with_progress(command: str) -> None:
    """Run a pipeline command with a progress bar and one status line."""
    bar = st.progress(0)
    with st.spinner("Working..."):
        res = runner.run(command)
    bar.progress(100)
    if res.status == "busy":
        st.warning(res.message)
    elif res.ok:
        st.success("Done. The advice below reflects this review.")
        audit = load_audit()
        render_action_cards(build_actions(audit))
    else:
        st.error(res.message if res.message else C.ERROR_REFRESH_FAILED)
        st.button(C.BTN_TRY_AGAIN, key="retry")
        if res.log_path:
            with st.expander(C.BTN_VIEW_LOG):
                st.code(_read_log(res.log_path) or "(no output)")


def _read_log(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()[-5000:]
    except Exception:  # noqa: BLE001
        return ""


# ── Page: Explore (P6) ────────────────────────────────────────────────────────

def page_explore() -> None:
    st.title(C.PAGE_EXPLORE)

    query = st.text_input("Search a name, symbol or ISIN", key="ex_q")
    options = [r["label"] for r in search(load_index(), query, 10)] if query else []
    if not options:
        st.caption("Type a name, symbol or ISIN to explore.")
        return
    choice = st.selectbox("Matches", options, key="ex_choice")
    symbol = next(r["symbol"] for r in search(load_index(), query, 10)
                  if r["label"] == choice)

    broker = resolve_broker(symbol)
    reg = q("SELECT name, instrument_class, currency, isin FROM asset_registry "
            "WHERE symbol = ?", [symbol])
    name = str(reg["name"].iloc[0]) if not reg.empty and reg["name"].iloc[0] else symbol
    cls = str(reg["instrument_class"].iloc[0]) if not reg.empty else \
        classify_instrument(symbol)

    st.subheader(f"{name}")
    st.caption(f"{C.class_word(cls)} · {broker.get('currency', '')}")
    structure = get_structure(symbol)
    if structure in (INVERSE_STRUCTURE, LEVERAGED_STRUCTURE):
        st.warning("This product is leveraged or inverse. It can lose value quickly.")

    # Price chart (full width).
    market = q("SELECT Date, Close, Volume FROM market_history WHERE Symbol = ? "
               "ORDER BY Date ASC", [symbol])
    if market.empty:
        st.info(C.EMPTY_NO_NEWS.format(name=name))
    else:
        import plotly.graph_objects as go
        close = market["Close"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=market["Date"], y=close, name="Price"))
        fig.add_trace(go.Scatter(x=market["Date"], y=close.rolling(200).mean(),
                                 name="200-day average"))
        band = 2.0 * close.pct_change().ewm(span=20).std() * close
        fig.add_trace(go.Scatter(x=market["Date"], y=close + band, name="Upper band",
                                 line=dict(color="rgba(0,0,0,0)")))
        fig.add_trace(go.Scatter(x=market["Date"], y=close - band, name="Lower band",
                                 fill="tonexty", line=dict(color="rgba(0,0,0,0)")))
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Why these scores.
    st.subheader(C.SEC_WHY_SCORES)
    metrics = load_metrics()
    review_date = C.fmt_ts(metrics.get("review_ts")) if metrics.get("review_ts") else ""
    if review_date:
        st.caption(f"From the review of {review_date}")
    scores = q("SELECT structural_grade, tactical_grade, active_score "
               "FROM asset_registry WHERE symbol = ?", [symbol])
    if not scores.empty:
        r = scores.iloc[0]
        for label, col in ((("Quality score", "structural_grade")),
                           ("Trend score", "tactical_grade"),
                           ("Overall score", "active_score")):
            val = float(r[col] or 0)
            st.write(f"{label}: {C.fmt_score(val)}")
            st.progress(min(max(val / 100.0, 0.0), 1.0))
    with st.expander(C.SEC_GLOSSARY):
        for term, text in C.GLOSSARY.items():
            st.write(f"{term}: {text}")

    # News and filings.
    st.subheader(C.SEC_NEWS)
    news = q("SELECT source, published_at, title, score FROM nlp_evidence "
             "WHERE symbol = ? ORDER BY published_at DESC LIMIT 20", [symbol])
    if news.empty:
        st.info(C.EMPTY_NO_NEWS.format(name=name))
    else:
        for _, row in news.iterrows():
            senti = "positive" if float(row["score"] or 0) > 0 else (
                "negative" if float(row["score"] or 0) < 0 else "neutral")
            st.write(f"{C.fmt_date(row['published_at'])} · {row['source']} · "
                     f"{row['title']} · {senti}")

    # How to buy.
    st.subheader(C.SEC_HOW_TO_BUY)
    if broker.get("isin"):
        st.write(f"ISIN {broker['isin']}")
        st.write(f"Route: {'savings plan' if cls in ('ETF', 'CASH') else 'one-off order'}")
    else:
        st.warning(C.ACTION_BLOCKED.format(symbol=symbol))
        st.button(C.BTN_FIX_IN_PORTFOLIO, key=f"ex_fix_{symbol}")


# ── Page: Settings (P7) ───────────────────────────────────────────────────────

def page_settings() -> None:
    st.title(C.PAGE_SETTINGS)

    # Data status.
    st.subheader(C.SEC_DATA_STATUS)
    instruments = q("SELECT COUNT(DISTINCT Symbol) AS n FROM market_history")
    m = int(instruments["n"].iloc[0]) if not instruments.empty else 0
    bar = C.fmt_date(latest_bar_date())
    history = load_history()
    refreshed = C.fmt_ts(history.iloc[-1]["review_ts"]) if not history.empty else "not yet"
    st.write(f"{m} instruments, prices through {bar or 'no data'}, refreshed {refreshed}.")
    if st.button(C.BTN_REFRESH, width="stretch"):
        _run_with_progress(runner.REFRESH)
    st.caption(C.HELP_REVIEW_CADENCE)

    # Reviews (last ten).
    st.subheader(C.SEC_REVIEWS)
    if history.empty:
        st.info(C.EMPTY_VALUE_CHART)
    else:
        for _, r in history.tail(10)[::-1].iterrows():
            st.write(f"{C.fmt_ts(r['review_ts'])} - {C.fmt_eur(r['value_eur'])} - "
                     f"{C.fmt_eur(r['pnl_eur'])}")

    # Health (problems only).
    problems = _health_problems()
    st.subheader("Health")
    if problems:
        for p in problems:
            st.warning(p)
    else:
        st.success(C.STATUS_ALL_CURRENT)

    # Diagnostics (collapsed).
    with st.expander(C.SEC_DIAGNOSTICS):
        from quant.portfolio.cash_rate import as_dicts
        st.write(f"Version {__version__}")
        st.write(f"Cash rate: {current_cash_apy()*100:.2f} percent")
        st.write(f"Stale threshold: {STALE_DATA_DAYS} days")
        for row in as_dicts():
            st.caption(f"{row['effective_date']} · {row['apy']*100:.2f}% · {row['source_url']}")


def _health_problems() -> list[str]:
    """Return plain-sentence problems only (empty when clean)."""
    problems: list[str] = []
    bar = latest_bar_date()
    if bar:
        try:
            from datetime import date
            days = (date.today() - date.fromisoformat(bar)).days
            if days > STALE_DATA_DAYS:
                problems.append(C.ERROR_STALE_PRICES.format(n=days))
        except Exception:  # noqa: BLE001
            pass
    else:
        problems.append(C.ERROR_STALE_PRICES.format(n=0))
    # Missing ISINs among holdings.
    portfolio = load_portfolio()
    if not portfolio.empty:
        for sym in portfolio["Symbol"].astype(str):
            if not resolve_broker(sym).get("isin"):
                problems.append(C.ACTION_BLOCKED.format(symbol=sym))
    return problems


def main() -> None:
    page = render_sidebar()
    if page == C.PAGE_TODAY:
        page_today()
    elif page == C.PAGE_PORTFOLIO:
        page_portfolio()
    elif page == C.PAGE_EXPLORE:
        page_explore()
    else:
        page_settings()


if __name__ == "__main__":
    main()
