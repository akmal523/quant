"""
test_account_config.py — account.yaml contract (v10.5.0, spec 7 / T2f).

Asserts: account.yaml loads; each risk profile maps to the documented limits;
a missing file yields the cash empty state, not zero.
"""
from __future__ import annotations

from quant.config import RISK_PROFILES
from quant.portfolio.account import AccountState, load_account, save_account


def test_load_account_reads_yaml(tmp_path):
    p = tmp_path / "account.yaml"
    p.write_text("base_currency: EUR\ncash_eur: 83.04\nrisk_profile: balanced\n")
    a = load_account(str(p))
    assert a.cash_eur == 83.04
    assert a.risk_profile == "balanced"
    assert a.base_currency == "EUR"
    assert a.loaded


def test_missing_file_yields_empty_cash(tmp_path):
    a = load_account(str(tmp_path / "nope.yaml"))
    assert a.cash_eur is None
    assert not a.cash_is_set
    assert not a.loaded


def test_each_profile_maps_to_documented_limits():
    for name, limits in RISK_PROFILES.items():
        a = AccountState("EUR", 0.0, name, True)
        assert a.risk_limits() == limits


def test_unknown_profile_falls_back(tmp_path):
    p = tmp_path / "account.yaml"
    p.write_text("risk_profile: yolo\n")
    a = load_account(str(p))
    assert a.risk_profile in RISK_PROFILES


def test_save_account_roundtrip(tmp_path):
    p = tmp_path / "account.yaml"
    save_account(AccountState("EUR", 100.0, "aggressive", True), str(p))
    a = load_account(str(p))
    assert a.cash_eur == 100.0
    assert a.risk_profile == "aggressive"
