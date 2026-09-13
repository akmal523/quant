"""
dashboard.py — Local Streamlit workspace (v10.5.0, spec 4).

Intent: the interactive workspace runs locally (DuckDB, caches, input files
live here). It is the ONLY place with write access, and it writes ONLY input
files (portfolio.csv, account.yaml), never derived artifacts. Four pages:
  1. Briefing   — actions, portfolio, data health, evidence footnote.
  2. Portfolio  — positions editor, account form, broker registry (read-only).
  3. Explorer   — chart, metrics, evidence list, execution card.
  4. Data and Runs — run buttons, run history, universe registry, data health.

Doctrine (spec 1): every visible element passes the action test or the trust
test. No silent defaults (R1), one source of truth (R2), every number carries
metadata (R3), terse (R4), no emoji (R5), recommendations cite their rule (R6).

Run:  streamlit run quant/dashboard.py   (from the repo root)
Dependencies: streamlit, plotly, pandas, quant.data.database,
quant.execution.taxonomy, quant.execution.routing, quant.portfolio.account,
quant.portfolio.editor, quant.reporting.actions, quant.reporting.artifacts.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from quant import paths
import os
import subprocess
import pandas as pd
import streamlit as st

from quant import __version__
from quant.config import BROKER_CASH_APY, STALE_DATA_DAYS, RISK_PROFILES
from quant.data.database import get_connection, init_db
from quant.execution.taxonomy import resolve_broker, classify_instrument, get_structure
from quant.execution.routing import route_signal, alpha_bps_from_active_score
from quant.portfolio.account import load_account, save_account, AccountState
from quant.portfolio.editor import validate_positions, save_portfolio
from quant.reporting.actions import build_actions
from quant.reporting.artifacts import latest_run

st.set_page_config(page_title="Quant-AI Family Office", layout="wide")

# ── Empty-state catalog (spec 4.6, exact strings) ─────────────────────────────
EMPTY_NO_RUN = "No run yet. Run quant run, or use Data and Runs."
EMPTY_REGIME = "Regime: not estimated (needs 250 bars of IWDA.AS)."
EMPTY_NEWS = "No news evidence for {symbol}. Sentiment scored neutral, confidence low."
EMPTY_BLOCKED = ("Blocked: ISIN missing for {symbol}. Add it in Portfolio, "
                 "or resolve in the broker app.")
EMPTY_ACTIONS = "No actions required today."
EMPTY_CASH = "Cash not set. Set it in Portfolio to enable cash-aware recommendations."

RISK_PROFILE_SENTENCES = {
    "conservative": "conservative keeps at least 20 percent in safety assets and at least 15 percent in cash",
    "balanced": "balanced keeps at least 10 percent in safety assets and at least 10 percent in cash",
    "aggressive": "aggressive keeps at least 5 percent in safety assets and at least 5 percent in cash",
}


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
def load_portfolio() -> pd.DataFrame:
    """Load portfolio.csv holdings."""
    try:
        from quant.portfolio.portfolio import load_portfolio as _lp
        return _lp(paths.DATA_PORTFOLIO)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_audit() -> pd.DataFrame:
    """Load the latest portfolio audit CSV (single source of actions)."""
    path = os.path.join(str(paths.OUTPUTS_DIR), "portfolio_audit.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_factor_scores() -> pd.DataFrame:
    """Load factor_scores.parquet via the single latest_run() accessor (T4a)."""
    run_dir = latest_run()
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
def load_evidence(symbol: str) -> pd.DataFrame:
    """Load NLP evidence rows for a symbol (spec 6)."""
    conn = get_connection()
    try:
        return conn.execute(
            "SELECT source, title, published_at, score, confidence "
            "FROM nlp_evidence WHERE symbol = ? ORDER BY published_at DESC",
            [symbol],
        ).df()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_broker_registry() -> pd.DataFrame:
    """Load data/broker_registry.csv (read-only)."""
    try:
        return pd.read_csv(paths.DATA_BROKER_REGISTRY)
    except Exception:
        return pd.DataFrame()


def _run_timestamp() -> str:
    run_dir = latest_run()
    return os.path.basename(run_dir) if run_dir else "none"


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> str:
    """Render the sidebar: app name, version, run timestamp, page nav."""
    st.sidebar.title("Quant-AI")
    st.sidebar.caption(f"Version {__version__}")
    st.sidebar.caption(f"Run {_run_timestamp()}")
    return st.sidebar.radio(
        "Navigate", ["Briefing", "Portfolio", "Explorer", "Data and Runs"]
    )


# ── Page 1: Briefing ──────────────────────────────────────────────────────────

def page_briefing() -> None:
    st.header("Briefing")

    run_dir = latest_run()
    if not run_dir:
        st.info(EMPTY_NO_RUN)
        return

    audit = load_audit()
    account = load_account()
    actions = build_actions(audit)

    # 1. Actions.
    st.subheader("Actions")
    if actions:
        rows = []
        for a in actions:
            rows.append({
                "Symbol": a["symbol"],
                "Action": a["action"],
                "Amount EUR": "" if a["blocked"] else f"{a['amount_eur']:.0f}",
                "Reason": a["reason"],
            })
        st.dataframe(pd.DataFrame(rows), width="stretch")
        for a in actions:
            if a["blocked"]:
                st.warning(a["remedy"])
    else:
        st.info(EMPTY_ACTIONS)

    # 2. Portfolio (one line per R3).
    st.subheader("Portfolio")
    port = load_portfolio()
    total_value = port["Amount_EUR"].sum() if not port.empty else 0.0
    pnl = float(audit["Real_PnL_EUR"].sum()) if (
        not audit.empty and "Real_PnL_EUR" in audit) else 0.0
    cash = (f"{account.cash_eur:.2f} EUR (cash, manual input)"
            if account.cash_is_set else EMPTY_CASH)
    st.write(f"Value {total_value:.2f} EUR (source: portfolio.csv)")
    st.write(f"Broker PnL {pnl:+.2f} EUR (source: broker)")
    st.write(f"Cash {cash}")
    st.write(f"Risk profile {account.risk_profile} (source: account.yaml)")
    st.write(f"Regime {EMPTY_REGIME}")

    # 3. Data health (blockers only).
    st.subheader("Data health")
    blockers = [a for a in actions if a["blocked"]]
    if blockers:
        for a in blockers:
            st.warning(a["remedy"])
    else:
        st.write("No blockers.")

    # 4. Evidence footnote.
    st.caption("Evidence: see Explorer for per-symbol news detail.")


# ── Page 2: Portfolio ─────────────────────────────────────────────────────────

def page_portfolio() -> None:
    st.header("Portfolio")

    registry = load_registry()
    universe = set(registry["symbol"].astype(str)) if not registry.empty else set()

    # Positions editor.
    st.subheader("Positions")
    port = load_portfolio()
    edit_cols = ["Symbol", "Avg_Entry_Price", "Current_Value_EUR", "Broker_PnL_EUR"]
    base = port[edit_cols] if not port.empty and set(edit_cols).issubset(port.columns) \
        else pd.DataFrame(columns=edit_cols)
    edited = st.data_editor(base, num_rows="dynamic", width="stretch", key="positions")

    if st.button("Save positions"):
        cleaned, warnings, errors = validate_positions(edited, universe)
        for w in warnings:
            st.warning(w)
        if errors:
            for e in errors:
                st.error(e)
        else:
            save_portfolio(cleaned, paths.DATA_PORTFOLIO)
            st.success("saved; takes effect on next quant run")

    # Account form.
    st.subheader("Account")
    account = load_account()
    cash_val = account.cash_eur if account.cash_is_set else 0.0
    cash = st.number_input("Cash EUR", min_value=0.0, value=float(cash_val), step=10.0)
    profile = st.radio(
        "Risk profile",
        list(RISK_PROFILES.keys()),
        index=list(RISK_PROFILES.keys()).index(account.risk_profile),
        captions=[RISK_PROFILE_SENTENCES[p] for p in RISK_PROFILES],
    )
    st.text_input("Base currency", value=account.base_currency, disabled=True)
    if st.button("Save account"):
        save_account(AccountState(account.base_currency, float(cash), profile, True))
        st.success("saved; takes effect on next quant run")

    # Broker registry (read-only).
    with st.expander("Broker registry (read-only)"):
        st.caption("Fix ISINs with scripts/repair_registry.py; editing stays scripted.")
        st.dataframe(load_broker_registry(), width="stretch")


# ── Page 3: Explorer ──────────────────────────────────────────────────────────

def page_explorer() -> None:
    st.header("Explorer")

    port = load_portfolio()
    registry = load_registry()
    holdings = list(port["Symbol"].unique()) if not port.empty else []
    all_syms = list(registry["symbol"].unique()) if not registry.empty else []
    options = list(dict.fromkeys(holdings + all_syms))
    selected = st.selectbox("Select asset", options) if options else ""
    query = st.text_input("Search by symbol or ISIN", "").strip().upper()
    symbol = query or selected
    if not symbol:
        st.info("Select an asset or type a symbol/ISIN to explore.")
        return

    broker = resolve_broker(symbol)
    if broker["yahoo_ticker"] != symbol:
        symbol = broker["yahoo_ticker"]

    df = load_market(symbol)
    if df.empty:
        st.warning(f"No market data for {symbol}. Run quant update.")
        return

    st.subheader(f"{symbol} — {classify_instrument(symbol)}")

    # Chart (kept: the one widget already earning its place).
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    close = df["Close"]
    sma200 = close.rolling(200).mean()
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

    # Metrics row: each labelled with the run timestamp it comes from (T4b).
    st.subheader("Metrics")
    run_ts = _run_timestamp()
    factors = load_factor_scores()
    reg_row = registry[registry["symbol"] == symbol] if not registry.empty else pd.DataFrame()
    if not reg_row.empty:
        r = reg_row.iloc[0]
        st.write(f"Structural {r.get('structural_grade', '')} (run {run_ts})")
        st.write(f"Tactical {r.get('tactical_grade', '')} (run {run_ts})")
        st.write(f"Active score {r.get('active_score', '')} (run {run_ts})")
    elif not factors.empty and "Symbol" in factors.columns and symbol in factors["Symbol"].values:
        st.write(f"Factor scores available (run {run_ts})")
    else:
        st.info(EMPTY_NO_RUN)

    # Evidence list (T4c).
    st.subheader("Evidence")
    evidence = load_evidence(symbol)
    if evidence.empty:
        st.info(EMPTY_NEWS.format(symbol=symbol))
    else:
        st.dataframe(evidence, width="stretch")

    # Execution card (T4d).
    st.subheader("Execution")
    isin = broker.get("isin", "")
    if not isin:
        st.warning(EMPTY_BLOCKED.format(symbol=symbol))
    else:
        st.write(f"ISIN {isin}")
        st.write(f"Route {route_signal(0.0, 0.0, classify_instrument(symbol), get_structure(symbol))}")
        st.write(f"Fee {BROKER_CASH_APY*0:.0f} EUR placeholder")
        active_score = float(reg_row["active_score"].iloc[0]) if (
            not reg_row.empty and "active_score" in reg_row) else 0.0
        st.write(f"Min trade size {alpha_bps_from_active_score(active_score)/100.0*100.0:.0f} EUR")


# ── Page 4: Data and Runs ─────────────────────────────────────────────────────

def page_data_and_runs() -> None:
    st.header("Data and Runs")

    # Run buttons.
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Run update"):
            _run_cli("update")
    with c2:
        if st.button("Run run"):
            _run_cli("run")

    # Run history.
    st.subheader("Run history")
    st.dataframe(_run_history(), width="stretch")

    # Universe registry (read-only; the only surviving piece of Universe Manager).
    st.subheader("Universe registry")
    registry = load_registry()
    if registry.empty:
        st.info("No registry yet. Run quant update to populate.")
    else:
        statuses = st.multiselect(
            "Status", sorted(registry["universe_status"].dropna().unique()),
            default=sorted(registry["universe_status"].dropna().unique()),
        )
        classes = st.multiselect(
            "Class", sorted(registry["instrument_class"].dropna().unique()),
            default=sorted(registry["instrument_class"].dropna().unique()),
        )
        filtered = registry[
            registry["universe_status"].isin(statuses)
            & registry["instrument_class"].isin(classes)
        ]
        cols = [c for c in ["symbol", "instrument_class", "universe_status", "structure"]
                if c in filtered.columns]
        st.dataframe(filtered[cols], width="stretch")

    # Data health (full list) + cache sizes + DB path.
    st.subheader("Data health")
    st.write(f"DB path {paths.DB_FILE}")
    st.write(f"Stale threshold {STALE_DATA_DAYS} days")
    st.write(f"Cash APY {BROKER_CASH_APY*100:.2f}%")


def _run_cli(command: str) -> None:
    """Execute the CLI as a subprocess and stream the log into an expander."""
    with st.expander(f"quant {command} log", expanded=True):
        proc = subprocess.run(
            [_sys.executable, "-m", "quant.cli", command],
            capture_output=True, text=True,
        )
        st.code(proc.stdout or "(no output)")
        if proc.returncode != 0:
            st.error(f"quant {command} exited {proc.returncode}")
            st.code(proc.stderr or "")


def _run_history() -> pd.DataFrame:
    """List run directories with their log path."""
    out = str(paths.OUTPUTS_DIR)
    if not os.path.isdir(out):
        return pd.DataFrame(columns=["run", "log"])
    runs = sorted([d for d in os.listdir(out) if d.startswith("run_")], reverse=True)
    rows = [{"run": r, "log": os.path.join(out, r, "pipeline.log")} for r in runs]
    return pd.DataFrame(rows)


def main() -> None:
    init_db()
    page = render_sidebar()
    if page == "Briefing":
        page_briefing()
    elif page == "Portfolio":
        page_portfolio()
    elif page == "Explorer":
        page_explorer()
    else:
        page_data_and_runs()


if __name__ == "__main__":
    main()
