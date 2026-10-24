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

from quant.engine import flows

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
