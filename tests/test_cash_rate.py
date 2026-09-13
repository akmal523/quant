"""
test_cash_rate.py — Trade Republic cash rate contract (v10.5.1, CASH4).

Asserts: schedule lookup by date; live-fetch failure falls back to the schedule;
the copy text shows 2.5 percent.
"""
from __future__ import annotations

from datetime import date

from quant.portfolio import cash_rate
from quant.ui import copy as C


def test_schedule_apy_before_and_after_change():
    assert cash_rate.schedule_apy(date(2026, 9, 15)) == 0.0225
    assert cash_rate.schedule_apy(date(2026, 9, 16)) == 0.025
    assert cash_rate.schedule_apy(date(2027, 1, 1)) == 0.025


def test_current_cash_apy_defaults_to_schedule():
    assert cash_rate.current_cash_apy(date(2026, 9, 16)) == 0.025


def test_live_fetch_failure_falls_back():
    def _boom(_url):
        raise RuntimeError("network down")

    rate = cash_rate.fetch_live_cash_apy(fetcher=_boom)
    assert rate is None
    # The schedule remains the deterministic fallback when the fetch fails.
    assert cash_rate.schedule_apy(date(2026, 9, 16)) == 0.025


def test_live_fetch_success_uses_parsed_rate():
    rate = cash_rate.fetch_live_cash_apy(
        fetcher=lambda _u: "interest rate 3,0 % p.a.",
        parser=cash_rate._default_parser,
    )
    assert rate == 0.03


def test_update_cash_rate_appends():
    before = len(cash_rate.CASH_RATE_SCHEDULE)
    cash_rate.update_cash_rate(0.028, date(2027, 1, 1))
    try:
        assert cash_rate.schedule_apy(date(2027, 6, 1)) == 0.028
        assert len(cash_rate.CASH_RATE_SCHEDULE) == before + 1
    finally:
        cash_rate.CASH_RATE_SCHEDULE[:] = [
            r for r in cash_rate.CASH_RATE_SCHEDULE
            if r.effective_date != date(2027, 1, 1)
        ]


def test_copy_text_shows_current_rate():
    text = C.HELP_CASH_APY.format(apy="2.5", date="16 Sep 2026")
    assert "2.5 percent" in text
    assert "16 Sep 2026" in text


def test_current_rate_exposes_effective_date():
    row = cash_rate.current_rate(date(2026, 9, 16))
    assert row.apy == 0.025
    assert row.effective_date == date(2026, 9, 16)
