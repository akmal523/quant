"""
assertions.py — Hard Data Quality Assertions (v10.4.0, Phase 1).

Intent: institutional systems do not trust data, they verify it. Unlike the soft
repair in data_quality.py, these assertions are a HARD gate: a violation raises
DataAssertionError and aborts the pipeline. They are pure and offline so CI can
run them without network access.

Invariants:
  - Each assert_* raises DataAssertionError on violation, returns None on pass.
  - run_assertions returns the list of passed check names; raises on first fail.
  - No I/O. pandas/numpy only.

Dependencies: pandas, numpy, config.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.config import ASSERT_MAX_DAILY_DROP


class DataAssertionError(Exception):
    """Raised when a hard data-quality assertion fails. Aborts the pipeline."""


def assert_no_duplicate_timestamps(df: pd.DataFrame, symbol: str) -> None:
    """Fail if (Symbol, Date) or Date has duplicates.

    Intent: duplicate timestamps corrupt rolling windows and double-count volume.
    Invariants: raises DataAssertionError on any duplicate; pure function.
    """
    if df is None or df.empty:
        return
    if "Date" in df.columns:
        dups = int(df["Date"].duplicated().sum())
        if dups > 0:
            raise DataAssertionError(
                f"[{symbol}] {dups} duplicate timestamps in market data"
            )
    elif df.index.has_duplicates:
        raise DataAssertionError(f"[{symbol}] duplicate index timestamps")


def assert_no_unexplained_drop(
    df: pd.DataFrame,
    symbol: str,
    max_drop: float = ASSERT_MAX_DAILY_DROP,
    split_dates: set | None = None,
) -> None:
    """Fail if Close drops more than max_drop in one day without a split flag.

    Intent: a >50% one-day drop is either a split (legitimate, must be flagged) or
    data corruption. If it is not explained by a known split date, abort.
    Invariants: raises DataAssertionError on an unexplained drop; pure function.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return
    close = df["Close"].astype(float)
    ret = close.pct_change()
    split_dates = split_dates or set()
    dates = df["Date"] if "Date" in df.columns else df.index

    for i in range(1, len(close)):
        r = ret.iloc[i]
        if not np.isfinite(r) or r >= -max_drop:
            continue
        d = pd.Timestamp(dates.iloc[i] if hasattr(dates, "iloc") else dates[i])
        if d in split_dates:
            continue
        raise DataAssertionError(
            f"[{symbol}] {r:.1%} one-day drop on {d.date()} without a split flag"
        )


def assert_fundamentals_sane(f_data: dict, symbol: str) -> None:
    """Fail if a profitable company has NaN or negative Debt/Equity.

    Intent: profitable firms with missing/negative leverage are a data error, not
    a signal. ROE > 0 defines "profitable".
    Invariants: raises DataAssertionError on violation; pure function.
    """
    if not f_data:
        return
    roe = f_data.get("ROE")
    de = f_data.get("DebtToEquity")
    profitable = roe is not None and np.isfinite(roe) and roe > 0
    if not profitable:
        return
    if de is None or (isinstance(de, float) and not np.isfinite(de)):
        raise DataAssertionError(
            f"[{symbol}] profitable company (ROE={roe}) has NaN Debt/Equity"
        )
    if de < 0:
        raise DataAssertionError(
            f"[{symbol}] profitable company (ROE={roe}) has negative Debt/Equity={de}"
        )


def run_assertions(
    df: pd.DataFrame,
    symbol: str,
    f_data: dict | None = None,
    split_dates: set | None = None,
) -> list[str]:
    """Run the full hard-assertion suite. Returns passed check names.

    Intent: single entry point for the pipeline and for CI. Raises
    DataAssertionError on the first violation.
    Invariants: returns list of passed check names; pure function (no I/O).
    """
    passed: list[str] = []
    assert_no_duplicate_timestamps(df, symbol)
    passed.append("no_duplicate_timestamps")
    assert_no_unexplained_drop(df, symbol, split_dates=split_dates)
    passed.append("no_unexplained_drop")
    if f_data is not None:
        assert_fundamentals_sane(f_data, symbol)
        passed.append("fundamentals_sane")
    return passed
