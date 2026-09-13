"""
reconciliation.py — Automated Broker Reconciliation (v10.4.0, Phase 2).

Intent: trust but verify. Compare the DuckDB theoretical portfolio state against
a Trade Republic CSV export and alert on any divergence (missing dividend,
unexecuted limit order, stale CSV). A silent divergence is how a family office
loses money without noticing.

Invariants:
  - reconcile is pure: (theoretical, broker) -> per-symbol divergence DataFrame.
  - build_theoretical_snapshot derives shares from broker value / entry price.
  - save_snapshot / load_snapshot are the only I/O (portfolio_snapshot table).

Dependencies: pandas, database.get_connection, currency.get_fx_to_eur.
"""
from __future__ import annotations

import pandas as pd

# Divergence above this EUR amount is flagged as a reconciliation break.
RECON_TOLERANCE_EUR = 1.00


def build_theoretical_snapshot(
    portfolio_df: pd.DataFrame,
    scan_df: pd.DataFrame,
    snapshot_date: str,
) -> pd.DataFrame:
    """Build the theoretical portfolio state from broker cost basis + live prices.

    Intent: shares are backed out from the broker's recorded value and entry price
    (never price-guessed), then revalued at the live EUR price. This is the
    system's independent estimate of what the account should be worth.
    Invariants: returns columns snapshot_date, symbol, shares, price_eur, value_eur.
    Pure computation (no I/O).
    """
    from quant.data.currency import get_fx_to_eur

    if portfolio_df is None or portfolio_df.empty:
        return pd.DataFrame(
            columns=["snapshot_date", "symbol", "shares", "price_eur", "value_eur"]
        )

    scan_map = scan_df.set_index("Symbol").to_dict("index") if scan_df is not None and not scan_df.empty else {}
    rows = []
    for _, p in portfolio_df.iterrows():
        sym = p["Symbol"]
        entry = p.get("Avg_Entry_Price", p.get("Buy_Price", 0)) or 0
        value_eur = p.get("Current_Value_EUR", p.get("Amount_EUR", 0)) or 0
        shares = (value_eur / entry) if entry and entry > 0 else 0.0

        price_eur = 0.0
        if sym in scan_map:
            native = scan_map[sym].get("Current_Price", 0) or 0
            price_eur = native * get_fx_to_eur(sym)

        rows.append({
            "snapshot_date": snapshot_date,
            "symbol": sym,
            "shares": round(shares, 6),
            "price_eur": round(price_eur, 4),
            "value_eur": round(shares * price_eur, 2),
        })
    return pd.DataFrame(rows)


def reconcile(theoretical_df: pd.DataFrame, broker_df: pd.DataFrame) -> pd.DataFrame:
    """Diff theoretical vs broker values per symbol.

    Intent: surface any position where the system's estimate diverges from the
    broker's reported value beyond RECON_TOLERANCE_EUR.
    Invariants: returns columns symbol, theoretical_eur, broker_eur, deviation_eur,
    flag; pure function (no I/O).
    """
    if theoretical_df is None or theoretical_df.empty:
        return pd.DataFrame(
            columns=["symbol", "theoretical_eur", "broker_eur", "deviation_eur", "flag"]
        )

    broker_map: dict[str, float] = {}
    if broker_df is not None and not broker_df.empty:
        for _, r in broker_df.iterrows():
            val = r.get("Value_EUR", r.get("Current_Value_EUR", r.get("value_eur")))
            if val is not None and pd.notna(val):
                broker_map[str(r["Symbol"])] = float(val)

    rows = []
    for _, t in theoretical_df.iterrows():
        sym = t["symbol"]
        theo = float(t["value_eur"])
        broker = broker_map.get(sym)
        if broker is None:
            rows.append({"symbol": sym, "theoretical_eur": round(theo, 2),
                         "broker_eur": None, "deviation_eur": None,
                         "flag": "[!] MISSING AT BROKER"})
            continue
        dev = theo - broker
        flag = "[!]" if abs(dev) > RECON_TOLERANCE_EUR else ""
        rows.append({"symbol": sym, "theoretical_eur": round(theo, 2),
                     "broker_eur": round(broker, 2), "deviation_eur": round(dev, 2),
                     "flag": flag})
    return pd.DataFrame(rows)


def save_snapshot(snapshot_df: pd.DataFrame) -> int:
    """Persist a theoretical snapshot to portfolio_snapshot. Returns row count.

    Intent: keep a daily theoretical state so divergences can be diffed over time.
    Invariants: best-effort (never raises); INSERT OR REPLACE on (date, symbol).
    """
    if snapshot_df is None or snapshot_df.empty:
        return 0
    try:
        from quant.data.database import get_connection
        conn = get_connection()
        n = 0
        for _, r in snapshot_df.iterrows():
            conn.execute(
                """INSERT OR REPLACE INTO portfolio_snapshot
                   (snapshot_date, symbol, shares, price_eur, value_eur)
                   VALUES (?, ?, ?, ?, ?)""",
                [r["snapshot_date"], r["symbol"], r["shares"], r["price_eur"], r["value_eur"]],
            )
            n += 1
        return n
    except Exception:
        return 0


def load_snapshot(snapshot_date: str) -> pd.DataFrame:
    """Load a persisted theoretical snapshot for a date. Empty-safe."""
    try:
        from quant.data.database import get_connection
        conn = get_connection()
        return conn.execute(
            "SELECT snapshot_date, symbol, shares, price_eur, value_eur "
            "FROM portfolio_snapshot WHERE snapshot_date = ?",
            [snapshot_date],
        ).df()
    except Exception:
        return pd.DataFrame(
            columns=["snapshot_date", "symbol", "shares", "price_eur", "value_eur"]
        )
