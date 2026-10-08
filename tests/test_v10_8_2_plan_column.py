"""test_v10_8_2_plan_column.py — the optional Plan_EUR_month column (B3).

The table stays five columns and Review is unaffected; the savings-plan rate is
an optional extra saved by the same Confirm.
"""
from __future__ import annotations

import pandas as pd

from quant.portfolio.editor import save_portfolio, validate_positions
from quant.portfolio.portfolio import load_portfolio


def _df(plan=None):
    row = {"Symbol": "EUNL.DE", "Avg_Entry_Price": 95.5,
           "Current_Value_EUR": 287.58, "Broker_PnL_EUR": 9.58}
    if plan is not None:
        row["Plan_EUR_month"] = plan
    return pd.DataFrame([row])


def test_plan_column_round_trips(tmp_path):
    path = str(tmp_path / "portfolio.csv")
    save_portfolio(_df(140.0), path)
    loaded = load_portfolio(path)
    assert "Plan_EUR_month" in loaded.columns
    assert float(loaded.iloc[0]["Plan_EUR_month"]) == 140.0


def test_missing_plan_column_is_null_not_an_error(tmp_path):
    path = str(tmp_path / "portfolio.csv")
    save_portfolio(_df(), path)
    loaded = load_portfolio(path)
    assert pd.isna(loaded.iloc[0]["Plan_EUR_month"])


def test_validation_ignores_plan_and_still_passes():
    cleaned, warnings, errors = validate_positions(_df(140.0))
    assert errors == []
    assert float(cleaned.iloc[0]["Plan_EUR_month"]) == 140.0
