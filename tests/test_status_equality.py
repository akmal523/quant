"""
test_status_equality.py — Status equality contract (v10.5.2, A3).

The holdings table and the action cards must render the same status word, derived
from one audit object. A position over target during cooldown can never read
"On track" in one widget and "Waiting" in another.
"""
from __future__ import annotations

import pandas as pd

from quant.reporting.actions import build_actions
from quant.ui import copy as C


def _audit() -> pd.DataFrame:
    # Symbols with registry ISINs (AMZN, AAPL, MSFT, NVDA) + one without (ZZZZ).
    return pd.DataFrame([
        {"Symbol": "AMZN", "Tier": "ACTIVE", "Target_Weight": "20.0%", "Drift": "-16.1%",
         "Recommendation": "BUY 150 EUR (ACTIVE drift -16.1% exceeds 5.0% threshold)"},
        {"Symbol": "AAPL", "Tier": "ACTIVE", "Target_Weight": "20.0%", "Drift": "8.0%",
         "Recommendation": "SELL 50 EUR (ACTIVE drift 8.0% exceeds 5.0% threshold)"},
        {"Symbol": "MSFT", "Tier": "ACTIVE", "Target_Weight": "20.0%", "Drift": "1.0%",
         "Recommendation": "HOLD: within threshold"},
        {"Symbol": "ZZZZ", "Tier": "ACTIVE", "Target_Weight": "10.0%", "Drift": "-9.0%",
         "Recommendation": "BUY 100 EUR (ACTIVE drift -9.0% exceeds 5.0% threshold)"},
        {"Symbol": "NVDA", "Tier": "ACTIVE", "Target_Weight": "20.0%", "Drift": "-7.0%",
         "Cooldown_Until": "2026-09-20",
         "Recommendation": "BUY 60 EUR (ACTIVE drift -7.0% exceeds 5.0% threshold)"},
    ])


def test_status_vocabulary_is_canonical():
    by_sym = {a["symbol"]: a for a in build_actions(_audit())}
    assert by_sym["AMZN"]["status"] == C.STATUS_ADD
    assert by_sym["AAPL"]["status"] == C.STATUS_TRIM
    assert by_sym["ZZZZ"]["status"] == C.STATUS_BLOCKED
    assert by_sym["NVDA"]["status"] == C.STATUS_WAITING.format(date=C.fmt_date("2026-09-20"))
    # Within threshold -> no action; the table default is On track.
    assert "MSFT" not in by_sym
    assert C.status_for("HOLD: within threshold") == C.STATUS_ON_TRACK


def _cooldown(row) -> str | None:
    """Mirror production: a missing Cooldown_Until is NaN in a DataFrame."""
    value = row.get("Cooldown_Until")
    if isinstance(value, float) and value != value:
        return None
    return value


def test_table_status_equals_action_status_for_every_holding():
    audit = _audit()
    by_sym = {a["symbol"]: a for a in build_actions(audit)}
    for _, r in audit.iterrows():
        sym = str(r["Symbol"])
        act = by_sym.get(sym)
        # The dashboard table renders the action status, or On track by default.
        rendered = act["status"] if act else C.STATUS_ON_TRACK
        expected = C.status_for(
            r["Recommendation"],
            blocked=bool(act and act["blocked"]),
            cooldown_until=_cooldown(r),
        )
        assert rendered == expected, f"status mismatch for {sym}"
