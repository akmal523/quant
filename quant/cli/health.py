"""
health.py — System health checks and diagnostics (v10.6.4).

Intent: one read-only check that reports the common failure modes: stale market
data, an invalid tiers file, tier allocation violations, a broken database, a
stale signal cache, and a missing data source. No writes, no secrets.

Invariants:
  - ``run_health_check`` never raises; returns
    ``{status, checks}`` with status in {HEALTHY, WARNING, CRITICAL}.
  - Each check returns ``{name, status, message}`` with status in
    {OK, WARNING, CRITICAL}.

Dependencies: pandas, quant.paths, quant.data.database.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd


def run_health_check() -> dict:
    """Run the full system health check. Never raises."""
    checks = [
        _check_data_freshness(),
        _check_tiers_validation(),
        _check_tier_allocations(),
        _check_database_integrity(),
        _check_signal_cache(),
        _check_external_apis(),
    ]
    statuses = [c["status"] for c in checks]
    if "CRITICAL" in statuses:
        overall = "CRITICAL"
    elif "WARNING" in statuses:
        overall = "WARNING"
    else:
        overall = "HEALTHY"
    return {"status": overall, "checks": checks}


def _check_data_freshness() -> dict:
    """Check that the latest market bar is recent."""
    try:
        from quant.data.database import read_only_connection
        with read_only_connection() as conn:
            row = conn.execute("SELECT MAX(Date) AS d FROM market_history").fetchone()
        if row and row[0]:
            latest = pd.to_datetime(row[0])
            age = datetime.now() - latest
            if age > timedelta(days=3):
                return {"name": "Data freshness", "status": "WARNING",
                        "message": f"Latest bar is {age.days} days old. Run 'quant update'."}
            return {"name": "Data freshness", "status": "OK",
                    "message": f"Latest bar is fresh ({latest.date()})."}
        return {"name": "Data freshness", "status": "WARNING",
                "message": "No price data found. Run 'quant update'."}
    except Exception as e:  # noqa: BLE001
        return {"name": "Data freshness", "status": "WARNING",
                "message": f"Could not check data freshness: {e}"}


def _check_tiers_validation() -> dict:
    """Check that tiers.csv is valid."""
    try:
        from quant.portfolio.portfolio import load_portfolio
        from quant.portfolio.tier_manager import load_tiers, validate_tiers_csv

        portfolio_df = load_portfolio()
        tiers_df = load_tiers()
        is_valid, errors = validate_tiers_csv(tiers_df, portfolio_df)
        if is_valid:
            return {"name": "Tiers validation", "status": "OK",
                    "message": "tiers.csv is valid."}
        return {"name": "Tiers validation", "status": "WARNING",
                "message": f"tiers.csv has issues: {'; '.join(errors)}. "
                           f"Run 'quant repair-tiers'."}
    except Exception as e:  # noqa: BLE001
        return {"name": "Tiers validation", "status": "CRITICAL",
                "message": f"Failed to validate tiers.csv: {e}"}


def _check_tier_allocations() -> dict:
    """Check that tier allocations respect their limits."""
    try:
        from quant.portfolio.autobalance import analyze_tier_allocations
        from quant.portfolio.portfolio import load_portfolio
        from quant.portfolio.tier_manager import load_tiers

        portfolio_df = load_portfolio()
        tiers_df = load_tiers()
        analysis = analyze_tier_allocations(portfolio_df, tiers_df)
        if analysis["violations"]:
            return {"name": "Tier allocations", "status": "WARNING",
                    "message": f"Tier allocation violations: "
                               f"{', '.join(analysis['violations'])}. "
                               f"Run 'quant suggest-rebalance'."}
        return {"name": "Tier allocations", "status": "OK",
                "message": "All tier allocations within limits."}
    except Exception as e:  # noqa: BLE001
        return {"name": "Tier allocations", "status": "CRITICAL",
                "message": f"Failed to check tier allocations: {e}"}


def _check_database_integrity() -> dict:
    """Check that the DuckDB store is readable and populated."""
    try:
        from quant.data.database import read_only_connection
        with read_only_connection() as conn:
            n = conn.execute("SELECT COUNT(*) FROM market_history").fetchone()[0]
        if n and n > 0:
            return {"name": "Database integrity", "status": "OK",
                    "message": f"Database healthy ({n} price records)."}
        return {"name": "Database integrity", "status": "WARNING",
                "message": "Database exists but has no price data."}
    except Exception as e:  # noqa: BLE001
        return {"name": "Database integrity", "status": "CRITICAL",
                "message": f"Database check failed: {e}"}


def _check_signal_cache() -> dict:
    """Check that the signal cache is fresh."""
    try:
        from quant.portfolio.signal_cache import load_signal_cache
        cache = load_signal_cache(max_age_days=7)
        if cache is None:
            return {"name": "Signal cache", "status": "WARNING",
                    "message": "Signal cache is stale or missing. Run 'quant run'."}
        return {"name": "Signal cache", "status": "OK",
                "message": "Signal cache is fresh."}
    except Exception as e:  # noqa: BLE001
        return {"name": "Signal cache", "status": "WARNING",
                "message": f"Failed to check signal cache: {e}"}


def _check_external_apis() -> dict:
    """Check that the data source library is importable (no network call)."""
    try:
        import yfinance  # noqa: F401
        return {"name": "External APIs", "status": "OK",
                "message": "yfinance is importable."}
    except Exception as e:  # noqa: BLE001
        return {"name": "External APIs", "status": "WARNING",
                "message": f"yfinance not available: {e}"}
