"""test_properties.py — property-based invariants (v10.7.4, Part 6).

Intent: invariants that must hold for ALL valid inputs, not just the examples in
the grid. Hypothesis generates the inputs; each property runs at least 200
examples.

Invariants:
  1. Advice kinds are a subset of the allowed kinds per tier.
  2. The allocator conserves the budget exactly.
  3. The sizing law is monotone across the 100 EUR untouchable threshold.
  4. Modified Dietz is bounded by the best/worst flow interpretations.
  5. Whether a buy is allowed is independent of the symbol's cooldown.
"""
from __future__ import annotations

from datetime import date

from hypothesis import given, settings
from hypothesis import strategies as st

from quant.engine import allocator, flows
from quant.engine.advice import build_advice
from tests.fixtures.live_portfolio import alpha_case, holding

AS_OF = date(2026, 10, 2)
FAR_FUTURE = date(2099, 1, 1)

ALLOWED_KINDS = {
    "FORTRESS": {"keep", "change_savings_plan"},
    "ALPHA": {"buy", "top_up", "sell_part", "keep", "to_cash"},
    "SPECULATIVE": {"keep"},
}


# ── Property 1: advice is a subset of allowed kinds per tier ──────────────────

@given(
    tier=st.sampled_from(["FORTRESS", "ALPHA", "SPECULATIVE"]),
    drift=st.floats(-0.5, 0.5),
    regime=st.sampled_from(["bull", "bear", "chop"]),
    cooldown=st.booleans(),
    conviction=st.floats(0, 100),
)
@settings(max_examples=200)
def test_advice_kinds_subset_of_tier(tier, drift, regime, cooldown, conviction):
    record = holding(
        "SYN", "Synthetic", tier, 1000.0, total=1000.0, target=0.20,
        current_weight=0.20 + drift, conviction=conviction,
        cooldown_until=(FAR_FUTURE if cooldown else None))
    advice, _ = build_advice([record], tiers={"SYN": tier}, regime=regime, as_of=AS_OF)
    kinds = {a["kind"] for a in advice if a.get("symbol") == "SYN"}
    assert kinds <= ALLOWED_KINDS[tier]


# ── Property 2: budget conservation ───────────────────────────────────────────

def _alloc_fixture(n_fortress, n_high, bets):
    holdings = []
    for i in range(n_fortress):
        holdings.append(holding(
            f"F{i}", f"Fortress {i}", "FORTRESS", 1000.0, total=10_000.0,
            target=0.30, current_weight=0.10 + 0.05 * i))
    for i in range(n_high):
        holdings.append(holding(
            f"A{i}", f"Alpha {i}", "ALPHA", 1000.0, total=10_000.0,
            target=0.20, current_weight=0.10, conviction=90.0))
    candidates = []
    if bets:
        holdings.append(holding(
            "BET", "Small Bet", "SPECULATIVE", 50.0, total=10_000.0,
            target=0.02, conviction=90.0))
        candidates.append({"symbol": "BET", "name": "Small Bet",
                           "tier": "SPECULATIVE", "conviction": 90.0})
    return holdings, candidates


@given(
    budget=st.floats(5, 10_000),
    n_fortress=st.integers(0, 3),
    n_high=st.integers(0, 3),
    regime=st.sampled_from(["bull", "bear"]),
    bets=st.booleans(),
)
@settings(max_examples=200)
def test_budget_conservation(budget, n_fortress, n_high, regime, bets):
    holdings, candidates = _alloc_fixture(n_fortress, n_high, bets)
    legs = allocator.allocate(
        budget, holdings, regime=regime, candidates=candidates, bets_enabled=bets)
    assert abs(sum(leg["amount_eur"] for leg in legs) - budget) < 1e-6


# ── Property 3: sizing law monotonicity ───────────────────────────────────────

@given(value=st.floats(99, 101))
@settings(max_examples=200)
def test_sizing_law_monotonicity(value):
    record = alpha_case(0.30, value=value)
    advice, rejected = build_advice([record], tiers={"AMZN": "ALPHA"}, as_of=AS_OF)
    has_sell = any(a["kind"] == "sell_part" for a in advice)
    if value < 100:
        assert not has_sell
        assert rejected
    else:
        assert has_sell


# ── Property 4: Modified Dietz bounds ─────────────────────────────────────────

@given(
    v_start=st.floats(100, 10_000),
    v_end=st.floats(100, 10_000),
    flow=st.floats(-1000, 1000),
)
@settings(max_examples=200)
def test_modified_dietz_bounds(v_start, v_end, flow):
    flow_type = "buy" if flow >= 0 else "sell"
    f = [{"date": date(2026, 1, 16), "type": flow_type, "amount_eur": abs(flow)}]
    ret = flows.modified_dietz(v_start, v_end, f, date(2026, 1, 1), date(2026, 1, 31))
    if ret is None:
        return
    net = flow
    # Best case: the flow lands at the start (weight 1); worst: at the end (0).
    denom_start = v_start + net
    denom_end = v_start
    if denom_start <= 0 or denom_end <= 0:
        return
    r_start = (v_end - v_start - net) / denom_start
    r_end = (v_end - v_start - net) / denom_end
    lo, hi = min(r_start, r_end), max(r_start, r_end)
    assert lo - 1e-9 <= ret <= hi + 1e-9


# ── Property 5: cooldown independence ─────────────────────────────────────────

@given(
    value=st.floats(100, 5000),
    drift=st.floats(-0.5, -0.06),
    conviction=st.floats(75, 100),
)
@settings(max_examples=200)
def test_cooldown_independence(value, drift, conviction):
    base = alpha_case(drift, value=value, conviction=conviction)
    cooled = alpha_case(drift, value=value, conviction=conviction, cooldown=FAR_FUTURE)
    advice_base, _ = build_advice([base], tiers={"AMZN": "ALPHA"}, as_of=AS_OF)
    advice_cooled, _ = build_advice([cooled], tiers={"AMZN": "ALPHA"}, as_of=AS_OF)
    buys_base = [a for a in advice_base if a["kind"] == "buy"]
    buys_cooled = [a for a in advice_cooled if a["kind"] == "buy"]
    assert buys_base == buys_cooled
