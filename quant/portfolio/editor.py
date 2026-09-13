"""
editor.py — Portfolio CSV editor logic (v10.5.0, spec 4.2 / T6).

Intent: pure validation + atomic write for data/portfolio.csv so the Streamlit
editor and the tests share one implementation. The UI only dispatches intent;
this module owns the rules (state/view separation).

Invariants:
  - validate_positions returns (cleaned_df, warnings, errors).
  - save_portfolio writes atomically (temp file + rename); no temp left behind.
  - Invested_EUR is recomputed as Current_Value_EUR - Broker_PnL_EUR.
  - Unknown symbols warn (not error); negative prices/values error.

Dependencies: pandas, os, tempfile.
"""
from __future__ import annotations

import os
import tempfile

import pandas as pd

REQUIRED_COLS = ["Symbol", "Avg_Entry_Price", "Current_Value_EUR", "Broker_PnL_EUR"]


def validate_positions(
    df: pd.DataFrame,
    universe_symbols: set[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Validate edited positions.

    Returns (cleaned_df, warnings, errors). Errors block the save; warnings do
    not. Recomputes Invested_EUR from broker truth.
    """
    warnings: list[str] = []
    errors: list[str] = []
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_COLS), warnings, errors

    out = df.copy()
    for col in REQUIRED_COLS:
        if col not in out.columns:
            out[col] = "" if col == "Symbol" else 0.0

    out["Symbol"] = out["Symbol"].astype(str).str.strip()
    for col in ["Avg_Entry_Price", "Current_Value_EUR", "Broker_PnL_EUR"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Drop fully empty rows.
    out = out[out["Symbol"].ne("") | out["Current_Value_EUR"].notna()]

    for _, r in out.iterrows():
        sym = str(r["Symbol"])
        if not sym:
            errors.append("A row has an empty symbol.")
            continue
        if universe_symbols is not None and sym not in universe_symbols:
            warnings.append(
                f"{sym} not in universe; it will enter the watchlist on next update."
            )
        for col in ["Avg_Entry_Price", "Current_Value_EUR"]:
            val = r[col]
            if pd.isna(val) or val < 0:
                errors.append(f"{sym}: {col} must be a positive number.")

    out["Invested_EUR"] = out["Current_Value_EUR"] - out["Broker_PnL_EUR"]
    return out.reset_index(drop=True), warnings, errors


def save_portfolio(df: pd.DataFrame, path: str) -> None:
    """Atomically write portfolio.csv (temp file + rename). No temp left behind."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    cols = REQUIRED_COLS
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            df[cols].to_csv(f, index=False)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
