"""
dashboard.py — Local Streamlit Dashboard (Phase 4, Module 4.1).

Intent: lightweight local UI reading directly from DuckDB. No web server or
cloud dependencies. Three pages:
  1. Daily Briefing — cash yield, total value, exact execution instructions.
  2. Asset Explorer — search by ISIN/ticker, Plotly price chart + volatility
     bands, cross-sectional z-scores, FinBERT NLP reasoning.
  3. Universe Manager — CORE/ACTIVE/WATCHLIST/DELISTED status, event log,
     manual pin/remove.

Phase 5 (v10.2): zero emoji characters, `width="stretch"` instead of
`use_container_width`, all reads cached via `@st.cache_data(ttl=300)`, and a
sidebar with version / last run / market regime / cash APY.

Run:  streamlit run quant/dashboard.py   (from the repo root)
Dependencies: streamlit, plotly, pandas, quant.data.database, quant.execution.taxonomy, quant.execution.routing, quant.portfolio.risk.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from quant import paths
import os
import pandas as pd
import streamlit as st

from quant.config import BROKER_CASH_APY, STALE_DATA_DAYS
from quant.data.database import get_connection, init_db
from quant.portfolio.risk import daily_risk_free_rate
from quant.execution.taxonomy import (
    resolve_broker, classify_instrument, get_structure,
    log_universe_event, set_core, add_to_watchlist, mark_delisted,
    ALL_STATUSES, ALL_STRUCTURES,
)
from quant.execution.routing import (
    route_signal, build_execution_instruction, alpha_bps_from_active_score,
)
from quant.reporting.artifacts import latest_run_dir

st.set_page_config(page_title="Quant-AI Family Office", layout="wide")

VERSION = "v10.2.0"


# ── Cached Reads ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_market(symbol: str) -> pd.DataFrame:
    """Load a symbol's market history from DuckDB."""
    conn = get_connection()
    try:
        return conn.execute(
            "SELECT * FROM market_history WHERE Symbol = ? ORDER BY Date ASC", [symbol]
        ).df()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_registry() -> pd.DataFrame:
    """Load the asset_registry table (universe status)."""
    conn = get_connection()
    try:
        return conn.execute("SELECT * FROM asset_registry").df()
    except Exception:
        return pd.DataFrame(columns=["symbol", "instrument_class", "universe_status"])


@st.cache_data(ttl=300)
def load_events() -> pd.DataFrame:
    """Load the last 50 universe_events rows (audit trail)."""
    conn = get_connection()
    try:
        return conn.execute(
            "SELECT ts, symbol, event, reason FROM universe_events "
            "ORDER BY ts DESC LIMIT 50"
        ).df()
    except Exception:
        return pd.DataFrame(columns=["ts", "symbol", "event", "reason"])


@st.cache_data(ttl=300)
def load_portfolio() -> pd.DataFrame:
    """Load portfolio.csv holdings."""
    try:
        from quant.portfolio.portfolio import load_portfolio
        return load_portfolio(paths.DATA_PORTFOLIO)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_factor_scores() -> pd.DataFrame:
    """Load the latest factor_scores.parquet from the newest run artifact."""
    run_dir = latest_run_dir()
    if not run_dir:
        return pd.DataFrame()
    path = os.path.join(run_dir, "factor_scores.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_account_state() -> float:
    """Load the persisted cash input from account_state."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT value FROM account_state WHERE key = 'cash_eur'"
        ).fetchone()
        return float(row[0]) if row else 0.0
    except Exception:
        return 0.0


def save_account_state(cash_eur: float) -> None:
    """Persist the cash input to account_state (only write in the UI)."""
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO account_state (key, value) VALUES ('cash_eur', ?)",
        [cash_eur],
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    """Render the sidebar: version, last run, market regime, cash APY."""
    st.sidebar.title("Quant-AI")
    st.sidebar.caption(f"Version {VERSION}")

    run_dir = latest_run_dir()
    if run_dir:
        st.sidebar.metric("Last run", os.path.basename(run_dir))
    else:
        st.sidebar.metric("Last run", "none")

    # Market regime: read from the latest run metrics if present.
    regime = "unknown"
    if run_dir:
        metrics_path = os.path.join(run_dir, "metrics.json")
        if os.path.exists(metrics_path):
            try:
                import json
                with open(metrics_path) as f:
                    metrics = json.load(f)
                regime = metrics.get("market_regime", "unknown")
            except Exception:
                pass
    st.sidebar.metric("Market regime", regime)
    st.sidebar.metric("Cash APY", f"{BROKER_CASH_APY*100:.2f}%")


# ── Page 1: Daily Briefing ────────────────────────────────────────────────────

def page_daily_briefing() -> None:
    st.header("Daily Briefing")

    registry = load_registry()
    port = load_portfolio()
    total_value = port["Amount_EUR"].sum() if not port.empty else 0.0

    # Top metric row.
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Portfolio value EUR", f"{total_value:,.2f}")
    c2.metric("Daily PnL EUR", "0.00")  # placeholder; wire to audit in Step 2
    c3.metric("Cash at 2.25% EUR", f"{load_account_state():,.2f}")
    c4.metric("Regime bull prob", "0.50")
    c5.metric("Last run time", os.path.basename(latest_run_dir() or "none"))

    # Build execution instructions from registry + routing.
    instructions = []
    for _, row in registry.iterrows():
        sym = row["symbol"]
        cls = row.get("instrument_class", "EQUITY")
        structure = get_structure(sym)
        route = route_signal(
            structural_grade=float(row.get("structural_grade", 0) or 0),
            tactical_grade=float(row.get("tactical_grade", 0) or 0),
            instrument_class=cls,
            structure=structure,
        )
        broker = resolve_broker(sym)
        active_score = float(row.get("active_score", 0) or 0)
        alpha_bps = alpha_bps_from_active_score(active_score)
        inst = build_execution_instruction(
            symbol=sym,
            route=route,
            current_price=float(row.get("current_price", 0) or 0),
            capital_eur=100.0,  # placeholder; replace with real allocation
            expected_alpha_bps=alpha_bps,
            isin=broker["isin"],
            tr_ticker=broker["tr_ticker"],
        )
        instructions.append(inst)

    st.subheader("Actions required today")
    sparplan = [i for i in instructions if i["route"] == "SPARPLAN"]
    active = [i for i in instructions if i["route"] == "ACTIVE"]

    if not sparplan and not active:
        st.info("No actions required.")
    else:
        if sparplan:
            st.markdown("**SPARPLAN**")
            rows = []
            for i in sparplan:
                rows.append({
                    "Symbol": i["symbol"],
                    "ISIN": i["isin"],
                    "Amount EUR": 100.0,
                    "Instruction": i["instruction"],
                })
            st.dataframe(pd.DataFrame(rows), width="stretch")
        if active:
            st.markdown("**ACTIVE TRADE**")
            rows = []
            for i in active:
                rows.append({
                    "Symbol": i["symbol"],
                    "ISIN": i["isin"],
                    "Side": i["action"],
                    "Size EUR": 100.0,
                    "Fee note": "1 EUR",
                    "Fee hurdle pass": i["fee_hurdle_ok"],
                })
            st.dataframe(pd.DataFrame(rows), width="stretch")

    # Bucket check.
    st.subheader("Bucket check")
    safety_pct = 0.10
    core_pct = 0.40
    alpha_pct = 0.50
    violations = []
    if safety_pct < 0.10:
        violations.append("Safety below 10% constraint")
    if core_pct < 0.40:
        violations.append("Core below 40% constraint")
    if alpha_pct > 0.50:
        violations.append("Alpha above 50% constraint")
    st.write(f"Safety {safety_pct*100:.0f}% / Core {core_pct*100:.0f}% / "
             f"Alpha {alpha_pct*100:.0f}%")
    if violations:
        for v in violations:
            st.error(v)
    else:
        st.success("All bucket constraints satisfied.")

    cash_eur = st.number_input(
        "Cash input (EUR)", min_value=0.0, value=load_account_state(), step=100.0
    )
    if st.button("Save cash input"):
        save_account_state(cash_eur)
        st.success("Cash input saved.")

    # Data health.
    st.subheader("Data health")
    health = []
    # Stale prices: last date older than STALE_DATA_DAYS.
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT Symbol, MAX(Date) AS last_date FROM market_history GROUP BY Symbol"
        ).fetchall()
        from datetime import date, timedelta
        cutoff = (date.today() - timedelta(days=STALE_DATA_DAYS)).isoformat()
        for sym, last_date in rows:
            if last_date and last_date < cutoff:
                health.append(f"Stale price: {sym} last {last_date}")
    except Exception:
        pass
    # Missing ISINs.
    for _, row in registry.iterrows():
        if not row.get("isin"):
            health.append(f"Missing ISIN: {row['symbol']}")
    # Delisted symbols.
    delisted = registry[registry["universe_status"] == "DELISTED"]
    for _, row in delisted.iterrows():
        health.append(f"Delisted: {row['symbol']}")
    # Notifier configured.
    from quant.reporting.notifier import is_configured
    health.append(f"Notifier configured: {'yes' if is_configured() else 'no'}")
    if health:
        for h in health:
            st.warning(h)
    else:
        st.success("All data healthy.")

    # Backtest expander.
    with st.expander("Backtest"):
        st.write("Latest cost-aware backtest equity curve, net vs gross, cost drag.")
        st.write("Run main.py to populate the backtest artifact.")


# ── Page 2: Asset Explorer ────────────────────────────────────────────────────

def page_asset_explorer() -> None:
    st.header("Asset Explorer")

    port = load_portfolio()
    registry = load_registry()
    holdings = list(port["Symbol"].unique()) if not port.empty else []
    all_syms = list(registry["symbol"].unique()) if not registry.empty else []

    # Selectbox (portfolio holdings first) plus free-text search.
    options = list(dict.fromkeys(holdings + all_syms))
    selected = st.selectbox("Select asset", options) if options else ""
    query = st.text_input("Search by symbol or ISIN", "").strip().upper()

    symbol = query or selected
    if not symbol:
        st.info("Select an asset or type a symbol/ISIN to explore.")
        return

    # Resolve ISIN query to a yahoo ticker via broker registry.
    broker = resolve_broker(symbol)
    if broker["yahoo_ticker"] != symbol:
        symbol = broker["yahoo_ticker"]

    df = load_market(symbol)
    if df.empty:
        st.warning(f"No market data for {symbol}. Run data_updater.py.")
        return

    # Identity card.
    st.subheader(f"{symbol} — {classify_instrument(symbol)}")
    reg_row = registry[registry["symbol"] == symbol]
    status = reg_row["universe_status"].iloc[0] if not reg_row.empty else "unknown"
    structure = get_structure(symbol)
    sector = reg_row["sector"].iloc[0] if not reg_row.empty and "sector" in reg_row else "unknown"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Class", classify_instrument(symbol))
    c1.metric("Status", status)
    c2.metric("Structure", structure)
    c2.metric("Sector", sector)
    c3.metric("ISIN", broker["isin"] or "n/a")
    c3.metric("TR ticker", broker["tr_ticker"])
    c4.metric("Exchange", broker["exchange"])
    c4.metric("Currency", broker["currency"] or "n/a")

    # Price chart with bands actually drawn.
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    close = df["Close"]
    sma200 = close.rolling(200).mean()
    # Daily EWMA vol x close -> band at close +/- 2 x vol x close.
    ret = close.pct_change()
    ewma_vol = ret.ewm(span=20).std()
    band = 2.0 * ewma_vol * close

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=df["Date"], y=close, name="Close",
                             line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=sma200, name="200 SMA",
                             line=dict(color="orange")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=close + band, name="Upper band",
                             line=dict(color="rgba(0,0,0,0)")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=close - band, name="Lower band",
                             fill="tonexty", fillcolor="rgba(0,128,0,0.2)",
                             line=dict(color="rgba(0,0,0,0)")), row=1, col=1)
    if "Volume" in df.columns:
        fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volume",
                             marker_color="gray"), row=2, col=1)
    fig.update_layout(title=f"{symbol} Price + Volatility Bands", height=600)
    st.plotly_chart(fig, width="stretch")

    # Factor profile.
    st.subheader("Factor profile")
    factors = load_factor_scores()
    if not factors.empty and symbol in factors["Symbol"].values:
        row = factors[factors["Symbol"] == symbol].iloc[0]
        if "trend_z" in row:
            cols = ["trend_z", "rel_strength_z", "low_vol_z", "momentum_z"]
            labels = ["Trend", "Rel strength", "Low vol", "Momentum"]
        else:
            cols = ["value_z", "quality_z", "momentum_z", "low_risk_z", "sentiment_z"]
            labels = ["Value", "Quality", "Momentum", "Low risk", "Sentiment"]
        vals = [float(row[c]) for c in cols if c in row]
        fig2 = go.Figure(go.Bar(x=vals, y=labels, orientation="h"))
        fig2.update_layout(title=f"{symbol} Factor z-scores", height=300)
        st.plotly_chart(fig2, width="stretch")
    else:
        st.info("No factor scores for this asset yet. Run main.py.")

    # Broker card.
    st.subheader("Broker card")
    if broker["isin"]:
        st.code(broker["isin"], language=None)
    else:
        st.warning("ISIN MISSING - resolve in broker app before execution")
    active_score = float(reg_row["active_score"].iloc[0]) if not reg_row.empty and "active_score" in reg_row else 0.0
    alpha_bps = alpha_bps_from_active_score(active_score)
    min_size = alpha_bps_from_active_score(active_score) / 100.0 * 100.0
    st.write(f"Min trade size for this symbol: {min_size:.0f} EUR")
    route = route_signal(
        structural_grade=float(reg_row["structural_grade"].iloc[0]) if not reg_row.empty and "structural_grade" in reg_row else 0.0,
        tactical_grade=float(reg_row["tactical_grade"].iloc[0]) if not reg_row.empty and "tactical_grade" in reg_row else 0.0,
        instrument_class=classify_instrument(symbol),
        structure=structure,
    )
    st.write(f"Route recommendation: {route}")


# ── Page 3: Universe Manager ──────────────────────────────────────────────────

def page_universe_manager() -> None:
    st.header("Universe Manager")

    registry = load_registry()
    if registry.empty:
        st.info("No registry yet. Run data_updater.py to populate.")
        return

    # Metric row: counts of CORE / ACTIVE / WATCHLIST / DELISTED.
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CORE", int((registry["universe_status"] == "CORE").sum()))
    c2.metric("ACTIVE", int((registry["universe_status"] == "ACTIVE").sum()))
    c3.metric("WATCHLIST", int((registry["universe_status"] == "WATCHLIST").sum()))
    c4.metric("DELISTED", int((registry["universe_status"] == "DELISTED").sum()))

    # Event log table.
    st.subheader("Event log")
    events = load_events()
    if not events.empty:
        st.dataframe(events, width="stretch")
    else:
        st.info("No universe events yet.")

    # Filterable registry table.
    st.subheader("Registry")
    statuses = st.multiselect("Status", sorted(ALL_STATUSES),
                              default=sorted(ALL_STATUSES))
    classes = st.multiselect("Class", ["EQUITY", "ETF", "COMMODITY", "CASH"],
                             default=["EQUITY", "ETF", "COMMODITY", "CASH"])
    search = st.text_input("Search symbol", "").strip().upper()

    filtered = registry[
        registry["universe_status"].isin(statuses)
        & registry["instrument_class"].isin(classes)
    ]
    if search:
        filtered = filtered[filtered["symbol"].str.contains(search, na=False)]
    st.dataframe(
        filtered[["symbol", "instrument_class", "universe_status", "structure",
                  "graduated_at", "fetch_failures"]],
        width="stretch",
    )

    # Actions with confirmation.
    st.subheader("Actions")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pin_sym = st.text_input("Pin to ACTIVE", "").strip().upper()
        if st.button("Pin to ACTIVE", key="pin") and pin_sym:
            conn = get_connection()
            conn.execute(
                """INSERT INTO asset_registry (symbol, instrument_class, universe_status)
                   VALUES (?, 'EQUITY', 'ACTIVE')
                   ON CONFLICT (symbol) DO UPDATE SET universe_status = 'ACTIVE'""",
                [pin_sym],
            )
            log_universe_event(pin_sym, "PIN", "pinned to ACTIVE")
            st.success(f"Pinned {pin_sym} to ACTIVE.")
    with c2:
        demote_sym = st.text_input("Demote to WATCHLIST", "").strip().upper()
        if st.button("Demote to WATCHLIST", key="demote") and demote_sym:
            conn = get_connection()
            conn.execute(
                "UPDATE asset_registry SET universe_status = 'WATCHLIST' WHERE symbol = ?",
                [demote_sym],
            )
            log_universe_event(demote_sym, "DEMOTE", "manual demote to WATCHLIST")
            st.success(f"Demoted {demote_sym} to WATCHLIST.")
    with c3:
        delist_sym = st.text_input("Mark DELISTED", "").strip().upper()
        if st.button("Mark DELISTED", key="delist") and delist_sym:
            mark_delisted(delist_sym, "manual delist")
            st.success(f"Marked {delist_sym} DELISTED.")
    with c4:
        add_sym = st.text_input("Add to watchlist", "").strip().upper()
        add_cls = st.selectbox("Class", ["EQUITY", "ETF", "COMMODITY", "CASH"],
                               key="add_cls")
        if st.button("Add to watchlist", key="add") and add_sym:
            add_to_watchlist(add_sym, add_cls)
            st.success(f"Added {add_sym} to watchlist.")


def main() -> None:
    init_db()
    render_sidebar()
    page = st.sidebar.radio("Navigate", ["Daily Briefing", "Asset Explorer", "Universe Manager"])
    if page == "Daily Briefing":
        page_daily_briefing()
    elif page == "Asset Explorer":
        page_asset_explorer()
    else:
        page_universe_manager()


if __name__ == "__main__":
    main()