"""test_v10_8_0_privacy.py — user state is never tracked; the app starts clean.

v10.8.0 (1.1): the owner's real positions, cash, tiers and backups must not be
committed, and the app must start against an empty data directory (the first-run
state of the redesign).
"""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = str(ROOT / "quant" / "dashboard.py")


def _load_guard():
    spec = importlib.util.spec_from_file_location(
        "check_no_user_state", ROOT / "scripts" / "check_no_user_state.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_load_portfolio_missing_file_is_empty(tmp_path):
    from quant.portfolio.portfolio import load_portfolio

    df = load_portfolio(str(tmp_path / "portfolio.csv"))
    assert df.empty


def test_load_account_missing_file_is_empty(tmp_path):
    from quant.portfolio.account import load_account

    state = load_account(str(tmp_path / "account.yaml"))
    assert state.cash_eur is None
    assert state.loaded is False


def test_app_starts_with_empty_data_dir(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from quant import paths

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(paths, "DATA_PORTFOLIO", str(empty / "portfolio.csv"))
    monkeypatch.setattr(paths, "DATA_ACCOUNT", str(empty / "account.yaml"))

    at = AppTest.from_file(DASHBOARD, default_timeout=90)
    at.run()
    assert not at.exception, f"dashboard raised: {at.exception}"
    for page in ("pages/today.py", "pages/portfolio.py", "pages/monthly.py",
                 "pages/explore.py", "pages/tax.py", "pages/settings.py"):
        at.switch_page(page).run()
        assert not at.exception, f"{page} raised: {at.exception}"


def test_no_user_state_tracked():
    try:
        out = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                             cwd=ROOT, check=True).stdout
    except Exception:  # noqa: BLE001
        pytest.skip("git unavailable")
    guard = _load_guard()
    assert guard.tracked_user_state(out.splitlines()) == []


def test_gitignore_excludes_user_state():
    text = (ROOT / ".gitignore").read_text()
    for pat in ("data/portfolio.csv", "data/account.yaml", "data/tiers.csv",
                "data/backups/"):
        assert pat in text
    assert "!data/portfolio.csv" not in text
