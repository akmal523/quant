"""test_v10_8_0_rates.py — one cash rate source and honest as-of dates (v10.8.0, 2.5)."""
from __future__ import annotations

from datetime import date

from quant.portfolio.cash_rate import current_cash_apy
from quant.ui import copy as ui_copy


def test_cash_apy_is_one_source():
    apy = f"{current_cash_apy()*100:g}"
    assert ui_copy.OPERATIONAL_CASH_LINE.format(
        amount="100", date="1 Jan 2026", apy=apy)
    assert ui_copy.MONTHLY_CASH_LEG_LINE.format(amount="60", apy=apy)
    assert ui_copy.INCOME_LINE.format(dividends="0", cash_yield="0", apy=apy)


def test_savings_plan_weekend_moves_to_monday():
    # 3 Oct 2026 is a Saturday -> the next trading day is Monday 5 Oct.
    line = ui_copy.savings_plan_line(date(2026, 9, 28), 3)
    assert "Monday 5 Oct 2026" in line


def test_account_cash_updated_roundtrip(tmp_path):
    from quant.portfolio.account import AccountState, load_account, save_account

    p = str(tmp_path / "account.yaml")
    save_account(AccountState("EUR", 100.0, "balanced", True, 1,
                              cash_updated="2026-10-01"), p)
    st = load_account(p)
    assert st.cash_updated == "2026-10-01"
