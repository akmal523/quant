"""
test_integration.py — Integration tests for full workflows (v10.6.5).

Hermetic: temporary input files only, no network, no live store.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF\U0000FE0F]"
)


def _write_inputs(tmp_path):
    portfolio_csv = tmp_path / "portfolio.csv"
    portfolio_csv.write_text(
        "Symbol,Avg_Entry_Price,Current_Value_EUR,Broker_PnL_EUR\n"
        "URTH,100.0,6000.0,500.0\n"
        "AAPL,150.0,4000.0,300.0\n"
        "NVDA,450.0,3000.0,200.0\n",
        encoding="utf-8",
    )
    tiers_csv = tmp_path / "tiers.csv"
    tiers_csv.write_text(
        "symbol,tier,last_updated,notes\n"
        "URTH,FORTRESS,2026-10-01,\n"
        "AAPL,ALPHA,2026-10-01,\n"
        "NVDA,ALPHA,2026-10-01,\n",
        encoding="utf-8",
    )
    return str(portfolio_csv), str(tiers_csv)



def test_emergency_liquidity_workflow(tmp_path):
    """Emergency liquidity returns a sensible, tier-aware plan."""
    from quant.portfolio.portfolio import load_portfolio
    from quant.portfolio.risk import emergency_sell_plan
    from quant.portfolio.tier_manager import load_tiers

    portfolio_path, tiers_path = _write_inputs(tmp_path)
    portfolio_df = load_portfolio(portfolio_path)
    tiers_df = load_tiers(tiers_path)

    plan = emergency_sell_plan(5000.0, portfolio_df, tiers_df)
    assert plan["total_available"] >= 5000.0 or plan["fortress_warning"] is not None
    assert len(plan["recommendations"]) > 0
    alpha_recs = [r for r in plan["recommendations"] if r["tier"] == "ALPHA"]
    assert len(alpha_recs) > 0

