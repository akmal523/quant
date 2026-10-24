"""
portfolio.py — Portfolio audit engine v9.
Schema: Symbol,Buy_Price,Amount_EUR,Original_Amount.
Original_Amount tracks original cost basis for PnL tracking.
"""
from __future__ import annotations

import os

import pandas as pd

from quant import paths
from quant.errors import DataError


def load_portfolio(filepath: str = paths.DATA_PORTFOLIO) -> pd.DataFrame:
    """Load portfolio.csv (Plan 3 broker-synced schema).

    New schema: Symbol, Avg_Entry_Price, Current_Value_EUR, Broker_PnL_EUR.
    The user copies exactly what Trade Republic shows. Invested_EUR is derived
    (Current_Value_EUR - Broker_PnL_EUR), never guessed from price history.

    Backward-compat aliases are emitted for the review audit: Buy_Price=
    Avg_Entry_Price, Amount_EUR=Current_Value_EUR, Original_Amount=Invested_EUR.
    Invariants: returns a DataFrame with Symbol + all new/legacy columns.
    """
    required_cols = ["Symbol", "Avg_Entry_Price", "Current_Value_EUR", "Broker_PnL_EUR"]
    if not os.path.exists(filepath):
        return pd.DataFrame(columns=required_cols)

    try:
        # comment='#' strips inline comments (e.g. "#Global Aerospace, Defence")
        # BEFORE CSV parsing, preventing stray commas from creating extra fields.
        df = pd.read_csv(filepath, comment="#").dropna(how="all")
        df.columns = df.columns.str.strip()

        # Accept either the new schema or the legacy (Buy_Price/Amount_EUR).
        if "Avg_Entry_Price" not in df.columns and "Buy_Price" in df.columns:
            df["Avg_Entry_Price"] = df["Buy_Price"]
        if "Current_Value_EUR" not in df.columns and "Amount_EUR" in df.columns:
            df["Current_Value_EUR"] = df["Amount_EUR"]
        if "Broker_PnL_EUR" not in df.columns:
            # Legacy fallback: derive from Original_Amount cost basis.
            if "Original_Amount" in df.columns:
                df["Broker_PnL_EUR"] = df["Current_Value_EUR"] - df["Original_Amount"]
            else:
                df["Broker_PnL_EUR"] = 0.0

        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0 if col != "Symbol" else "UNKNOWN"

        df["Symbol"] = df["Symbol"].astype(str).str.strip()
        df["Avg_Entry_Price"] = pd.to_numeric(
            df["Avg_Entry_Price"].astype(str).str.strip(), errors="coerce")
        df["Current_Value_EUR"] = pd.to_numeric(
            df["Current_Value_EUR"].astype(str).str.strip(), errors="coerce")
        df["Broker_PnL_EUR"] = pd.to_numeric(
            df["Broker_PnL_EUR"].astype(str).str.strip(), errors="coerce")

        # v10.8.2 (B3): optional savings-plan rate per holding (may be absent).
        if "Plan_EUR_month" in df.columns:
            df["Plan_EUR_month"] = pd.to_numeric(
                df["Plan_EUR_month"].astype(str).str.strip(), errors="coerce")
        else:
            df["Plan_EUR_month"] = pd.NA

        # Derived invested amount (broker truth, not price-guessed).
        df["Invested_EUR"] = df["Current_Value_EUR"] - df["Broker_PnL_EUR"]

        # Backward-compat aliases for legacy callers.
        df["Buy_Price"] = df["Avg_Entry_Price"]
        df["Amount_EUR"] = df["Current_Value_EUR"]
        df["Original_Amount"] = df["Invested_EUR"]

        df = df.dropna(subset=["Symbol"]).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=required_cols)

    # v10.7.4 (Part 5.2): a negative position value is corrupt data.
    if (df["Current_Value_EUR"] < 0).any():
        raise DataError("portfolio.csv contains a negative position value")
    # v10.7.4 (Part 5.2): duplicate rows are deduplicated by symbol (keep last).
    df = df.drop_duplicates(subset=["Symbol"], keep="last").reset_index(drop=True)
    return df


def portfolio_issues(filepath: str = paths.DATA_PORTFOLIO) -> list[str]:
    """Plain warnings about portfolio.csv (Part 5.2). Never raises."""
    if not os.path.exists(filepath):
        return []
    try:
        raw = pd.read_csv(filepath, comment="#").dropna(how="all")
        raw.columns = raw.columns.str.strip()
    except Exception:  # noqa: BLE001
        return []
    issues: list[str] = []
    if "Symbol" in raw.columns:
        dups = raw[raw.duplicated(subset=["Symbol"], keep=False)]
        if not dups.empty:
            issues.append(
                f"portfolio.csv: {len(dups)} duplicate rows "
                f"({sorted(str(s) for s in dups['Symbol'].unique())})")
    if "Current_Value_EUR" in raw.columns:
        vals = pd.to_numeric(raw["Current_Value_EUR"], errors="coerce")
        neg = int((vals < 0).sum())
        if neg:
            issues.append(f"portfolio.csv: {neg} negative position value(s)")
    return issues


# ── v10.6.2: Three-tier routing (R-TIER-1) ───────────────────────────────────
# The legacy 4-tier classify_asset() above is retained for backward
# compatibility. The functions below add the 3-tier system on top: tier
# assignments come from data/tiers.csv, and each holding is routed to its tier
# scorer. portfolio.csv is never modified.
