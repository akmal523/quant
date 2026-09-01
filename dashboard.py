"""
dashboard.py — Local Streamlit Dashboard (Phase 4, Module 4.1).

Intent: lightweight local UI reading directly from DuckDB. No web server or
cloud dependencies. Three pages:
  1. Daily Briefing — cash yield, total value, exact execution instructions.
  2. Asset Explorer — search by ISIN/ticker, Plotly price chart + GARCH bands,
     cross-sectional z-scores, FinBERT NLP reasoning.
  3. Universe Manager — ACTIVE vs WATCHLIST status, manual pin/remove.

Run:  streamlit run dashboard.py
Dependencies: streamlit, plotly, pandas, database, taxonomy, routing, risk.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import BROKER_CASH_APY
from database import get_connection, init_db
from risk import daily_risk_free_rate
from taxonomy import resolve_broker, classify_instrument
from routing import route_signal, build_execution_instruction

st.set_page_config(page_title="Family Office Terminal", layout="wide")


def load_market(symbol: str) -> pd.DataFrame:
    """Load a symbol's market history from DuckDB."""
    conn = get_connection()
    try:
        return conn.execute(
            "SELECT * FROM market_history WHERE Symbol = ? ORDER BY Date ASC", [symbol]
        ).df()
    except Exception:
        return pd.DataFrame()


def load_registry() -> pd.DataFrame:
    """Load the asset_registry table (universe status)."""
    conn = get_connection()
    try:
        return conn.execute("SELECT * FROM asset_registry").df()
    except Exception:
        return pd.DataFrame(columns=["symbol", "instrument_class", "universe_status"])


# ── Page 1: Daily Briefing ────────────────────────────────────────────────────
def page_daily_briefing() -> None:
    st.title("📊 Daily Briefing")
    st.metric("Cash Yield (APY)", f"{BROKER_CASH_APY*100:.2f}%",
              f"daily {daily_risk_free_rate()*100:.4f}%")

    registry = load_registry()
    if registry.empty:
        st.info("No asset registry yet. Run data_updater.py + main.py first.")
        return

    # Total portfolio value from portfolio.csv.
    try:
        from portfolio import load_portfolio
        port = load_portfolio("portfolio.csv")
        total_value = port["Amount_EUR"].sum() if not port.empty else 0.0
        st.metric("Total Portfolio Value", f"€{total_value:,.2f}")
    except Exception:
        total_value = 0.0

    # Build execution instructions from registry + routing.
    instructions = []
    for _, row in registry.iterrows():
        sym = row["symbol"]
        cls = row.get("instrument_class", "EQUITY")
        route = route_signal(
            structural_grade=float(row.get("structural_grade", 0) or 0),
            tactical_grade=float(row.get("tactical_grade", 0) or 0),
            instrument_class=cls,
        )
        broker = resolve_broker(sym)
        inst = build_execution_instruction(
            symbol=sym,
            route=route,
            current_price=float(row.get("current_price", 0) or 0),
            capital_eur=100.0,  # placeholder; replace with real allocation
            expected_alpha_bps=200.0,
            isin=broker["isin"],
            tr_ticker=broker["tr_ticker"],
        )
        instructions.append(inst)

    st.subheader("📋 Today's Execution Instructions")
    if instructions:
        st.dataframe(pd.DataFrame(instructions)[
            ["symbol", "route", "action", "instruction", "min_trade_size_eur", "fee_hurdle_ok"]
        ])
    else:
        st.info("No execution instructions.")


# ── Page 2: Asset Explorer ────────────────────────────────────────────────────
def page_asset_explorer() -> None:
    st.title("🔍 Asset Explorer")
    query = st.text_input("Search by ISIN or Ticker", "").strip().upper()

    if not query:
        st.info("Type an ISIN or ticker to explore.")
        return

    # Resolve query to a yahoo ticker via broker registry.
    broker = resolve_broker(query)
    symbol = query if broker["yahoo_ticker"] == query else broker["yahoo_ticker"]

    df = load_market(symbol)
    if df.empty:
        st.warning(f"No market data for {symbol}. Run data_updater.py.")
        return

    st.subheader(f"{symbol} — {classify_instrument(symbol)}")
    st.write(f"ISIN: {broker['isin'] or 'n/a'} | TR ticker: {broker['tr_ticker']} | "
             f"Exchange: {broker['exchange']}")

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close"))
    # GARCH/EWMA volatility band overlay.
    if "GARCH_Vol" in df.columns:
        upper = df["Close"] * (1 + df["GARCH_Vol"])
        lower = df["Close"] * (1 - df["GARCH_Vol"])
        fig.add_trace(go.Scatter(x=df["Date"], y=upper, name="Upper band",
                                 line=dict(dash="dot", color="green")))
        fig.add_trace(go.Scatter(x=df["Date"], y=lower, name="Lower band",
                                 line=dict(dash="dot", color="red")))
    fig.update_layout(title=f"{symbol} Price + Volatility Bands", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Cross-sectional z-scores (placeholder from registry if present).
    st.subheader("Cross-Sectional Z-Scores")
    st.write("Value / Quality / Momentum z-scores appear here after factor scoring.")


# ── Page 3: Universe Manager ──────────────────────────────────────────────────
def page_universe_manager() -> None:
    st.title("🗂️ Universe Manager")
    registry = load_registry()
    if registry.empty:
        st.info("No registry yet. Run data_updater.py to populate.")
        return

    st.subheader("Graduated (ACTIVE) vs Watchlist")
    st.dataframe(registry[["symbol", "instrument_class", "universe_status", "graduated_at"]])

    st.subheader("Manual Pin / Remove")
    col1, col2 = st.columns(2)
    with col1:
        pin_sym = st.text_input("Pin symbol to ACTIVE", "").strip().upper()
        if st.button("Pin to ACTIVE") and pin_sym:
            conn = get_connection()
            conn.execute(
                """INSERT INTO asset_registry (symbol, instrument_class, universe_status)
                   VALUES (?, 'EQUITY', 'ACTIVE')
                   ON CONFLICT (symbol) DO UPDATE SET universe_status = 'ACTIVE'""",
                [pin_sym],
            )
            st.success(f"Pinned {pin_sym} to ACTIVE.")
    with col2:
        remove_sym = st.text_input("Remove symbol", "").strip().upper()
        if st.button("Remove from registry") and remove_sym:
            conn = get_connection()
            conn.execute("DELETE FROM asset_registry WHERE symbol = ?", [remove_sym])
            st.success(f"Removed {remove_sym}.")


def main() -> None:
    init_db()
    page = st.sidebar.radio("Navigate", ["Daily Briefing", "Asset Explorer", "Universe Manager"])
    if page == "Daily Briefing":
        page_daily_briefing()
    elif page == "Asset Explorer":
        page_asset_explorer()
    else:
        page_universe_manager()


if __name__ == "__main__":
    main()