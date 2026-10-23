"""
test_v10_7_3_advice.py — advice truth + one target source (v10.7.3, Part 9.3/9.4/9.5).

Intent: prove the FORTRESS verdict law, that the allocator and build_advice name
the SAME top-up symbol on the live-shaped fixture, and that the top-up sentence
is the ONE shared string.

Invariants: pure unit tests; no network, no I/O.
"""
from __future__ import annotations

from quant.engine.advice import build_advice
from quant.ui import copy as C

# The live-shaped fixture: EUNL.DE under target, 5J50.DE over target.
_LIVE = [
    {"symbol": "EUNL.DE", "name": "iShares Core MSCI World", "tier": "FORTRESS",
     "value_eur": 3400.0, "current_weight": 0.34, "target_weight": 0.50,
     "conviction": 0.0},
    {"symbol": "5J50.DE", "name": "Global Aero & Defense", "tier": "FORTRESS",
     "value_eur": 1700.0, "current_weight": 0.17, "target_weight": 0.10,
     "conviction": 0.0},
    {"symbol": "SXRV.DE", "name": "iShares Nasdaq 100", "tier": "ALPHA",
     "value_eur": 2000.0, "current_weight": 0.20, "target_weight": 0.20,
     "conviction": 0.0},
    {"symbol": "AMZN", "name": "Amazon.com", "tier": "ALPHA",
     "value_eur": 2900.0, "current_weight": 0.29, "target_weight": 0.20,
     "conviction": 0.0},
]
_TIERS = {"EUNL.DE": "FORTRESS", "5J50.DE": "FORTRESS",
          "SXRV.DE": "ALPHA", "AMZN": "ALPHA"}


def test_fortress_verdict_never_sell():
    advice, _rejected = build_advice(holdings=_LIVE, tiers=_TIERS)
    by_sym = {a["symbol"]: a for a in advice if a.get("symbol")}
    # 5J50.DE is FORTRESS and over target: it can never be sold.
    assert by_sym["5J50.DE"]["kind"] in ("keep", "change_savings_plan")
    assert by_sym["5J50.DE"]["kind"] != "sell_part"


def test_topup_sentence_is_the_one_shared_string():
    advice, _rejected = build_advice(holdings=_LIVE, tiers=_TIERS)
    topup = next(a for a in advice if a["kind"] == "change_savings_plan")
    expected = C.STEP_TOP_UP.format(
        label="iShares Core MSCI World (EUNL.DE)", pct="34", target="50")
    assert topup["why"] == expected
    assert topup["why"] == (
        "Top up the savings plan for iShares Core MSCI World (EUNL.DE): it is 34 "
        "percent of invested vs 50 percent target.")
