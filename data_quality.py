"""
data_quality.py — Data Quality Gate (Part 3, Gap #1).

Intent: validate incoming market data BEFORE it enters DuckDB. Yahoo Finance
can return garbage (NaN, duplicate dates, split-adjustment errors, stale
prices). Without a gate, bad data silently corrupts the scoring pipeline.
Runs in data_updater.py before appending.

Invariants:
  - validate_batch returns (is_valid, issues); is_valid True iff no issues.
  - auto_repair removes duplicates/NaN and interpolates small gaps.
  - Pure computation; no I/O.

Dependencies: pandas.
"""
from __future__ import annotations

import pandas as pd


class DataQualityValidator:
    """Validates incoming market data before it enters DuckDB."""

    THRESHOLDS = {
        "max_daily_return": 0.25,   # 25% daily move without news = suspicious
        "min_price": 0.01,          # below 1 cent = error
        "max_price": 100000.0,      # above EUR 100k for ETFs = error
        "max_stale_days": 5,        # older than 5 days = stale
        "min_history_days": 60,     # need at least 60 days for scoring
    }

    def validate_batch(self, df: pd.DataFrame, symbol: str) -> tuple[bool, list[str]]:
        """Validate a batch of market data. Returns (is_valid, issues)."""
        issues = []
        if df is None or df.empty:
            return False, ["Empty dataframe"]

        if "Close" not in df.columns:
            return False, ["Missing Close column"]

        close = df["Close"]

        nan_count = int(close.isna().sum())
        if nan_count > 0:
            issues.append(f"{nan_count} NaN values in Close price")

        if (close < 0).any():
            issues.append("Negative price detected")

        if (close > self.THRESHOLDS["max_price"]).any():
            issues.append("Price above max threshold")

        daily_return = close.pct_change()
        extreme = daily_return[daily_return.abs() > self.THRESHOLDS["max_daily_return"]]
        if len(extreme) > 0:
            issues.append(
                f"{len(extreme)} days with >{self.THRESHOLDS['max_daily_return']*100:.0f}% moves"
            )

        if "Date" in df.columns:
            dup_count = int(df["Date"].duplicated().sum())
            if dup_count > 0:
                issues.append(f"{dup_count} duplicate dates")

            latest_date = pd.to_datetime(df["Date"]).max()
            days_stale = (pd.Timestamp.now() - latest_date).days
            if days_stale > self.THRESHOLDS["max_stale_days"]:
                issues.append(f"Data is {days_stale} days stale")

        if len(close) < self.THRESHOLDS["min_history_days"]:
            issues.append(
                f"Only {len(close)} days history (< {self.THRESHOLDS['min_history_days']})"
            )

        if close.iloc[-1] < self.THRESHOLDS["min_price"]:
            issues.append("Price below minimum threshold")

        return len(issues) == 0, issues

    def auto_repair(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Attempt automatic repairs for common issues."""
        if df is None or df.empty:
            return df

        if "Date" in df.columns:
            df = df.drop_duplicates(subset="Date", keep="last")
            df = df.sort_values("Date")

        df = df.dropna(subset=["Close"])

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            # time-weighted interpolation requires a DatetimeIndex.
            df = df.set_index("Date")
            df["Close"] = df["Close"].interpolate(method="time", limit=2)
            df = df.reset_index()

        df = df.dropna(subset=["Close"])
        return df