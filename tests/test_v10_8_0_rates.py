"""test_v10_8_0_rates.py — one cash rate source and honest as-of dates (v10.8.0, 2.5)."""
from __future__ import annotations


def test_account_cash_updated_roundtrip(tmp_path):
    from quant.portfolio.account import AccountState, load_account, save_account

    p = str(tmp_path / "account.yaml")
    save_account(AccountState("EUR", 100.0, "balanced", True, 1,
                              cash_updated="2026-10-01"), p)
    st = load_account(p)
    assert st.cash_updated == "2026-10-01"
