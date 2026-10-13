"""test_v10_8_7_phase3.py — v10.8.7 Phase 3: T3 (one weight math + plan).

R-C3: a single pair of functions produces instrument and class weights
(invested-only). Every consumer calls them; a lint test fails on any other weight
computation.

T3: the invest plan takes the open-advice set as input (no buy into an open
trim/sell), separates one-off EUR from savings-plan EUR/m, and can suggest a
non-held idea. A1: the canonical class words are Long-term / Active / Speculative;
a lint test fails on synonyms.

Synthetic fixtures only; the user's real DB and data dir are never touched.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from quant.engine.invest_plan import propose_by_class
from quant.engine.weights import class_weights, instrument_weights

_QUANT = Path(__file__).resolve().parents[1] / "quant"


# ── R-C3: the ONE weight math ─────────────────────────────────────────────────

def test_instrument_weights_sum_to_one():
    holdings = [
        {"symbol": "AAA", "value_eur": 500.0},
        {"symbol": "BBB", "value_eur": 300.0},
        {"symbol": "CCC", "value_eur": 200.0},
    ]
    w = instrument_weights(holdings)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert abs(w["AAA"] - 0.5) < 1e-9


def test_class_weights_sum_to_one():
    holdings = [
        {"symbol": "AAA", "tier": "FORTRESS", "value_eur": 500.0},
        {"symbol": "BBB", "tier": "ALPHA", "value_eur": 300.0},
        {"symbol": "CCC", "tier": "SPECULATIVE", "value_eur": 200.0},
    ]
    w = class_weights(holdings)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert abs(w["FORTRESS"] - 0.5) < 1e-9


def test_no_local_weight_computation_outside_weights():
    """R-C3: no UI or engine module computes a weight locally.

    The product path is quant/ui + quant/engine; the legacy reporting/portfolio
    audit modules are not in the five-page path and are out of scope.
    """
    allow = {"weights.py", "holdings_view.py"}
    patterns = [
        re.compile(r"[\"']current_weight[\"']\s*:"),          # dict assignment
        re.compile(r"value_eur[^\n]*/[^\n]*(total|pool)"),    # value/total
    ]
    offenders: list[str] = []
    for sub in ("ui", "engine"):
        for path in (_QUANT / sub).rglob("*.py"):
            if path.name in allow:
                continue
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                for pat in patterns:
                    if pat.search(line):
                        offenders.append(
                            f"{path.relative_to(_QUANT)}:{i}: {line.strip()}")
    assert not offenders, f"local weight computation outside weights.py: {offenders}"


# ── T3: the plan vs open advice ───────────────────────────────────────────────

def test_open_trim_blocks_buy_and_notes_exclusion():
    """An open trim on SXRV => no SXRV buy; the exclusion is noted in one line."""
    holdings = [
        {"symbol": "EUNL.DE", "name": "World", "tier": "FORTRESS", "value_eur": 500.0},
        {"symbol": "SXRV.DE", "name": "Nasdaq", "tier": "ALPHA", "value_eur": 100.0},
    ]
    plan = propose_by_class(
        1000.0, holdings, "balanced",
        open_advice=[{"verb": "sell_part", "symbol": "SXRV.DE"}])
    syms = {o["symbol"] for o in plan["orders"]}
    assert "SXRV.DE" not in syms, "must not buy into an open trim"
    assert any(e["symbol"] == "SXRV.DE" for e in plan["excluded"])
    assert all(e["reason"] for e in plan["excluded"])


def test_plan_can_suggest_a_non_held_symbol():
    """T2/T3: a non-held idea can fill an active class (How = 'New position')."""
    holdings = [
        {"symbol": "EUNL.DE", "name": "World", "tier": "FORTRESS", "value_eur": 1000.0},
    ]
    plan = propose_by_class(
        500.0, holdings, "balanced",
        candidates=[{"symbol": "AMZN", "name": "Amazon", "score": 80.0}])
    syms = {o["symbol"] for o in plan["orders"]}
    assert "AMZN" in syms
    assert any(o["how"] == "New position" for o in plan["orders"])


def test_one_off_sum_never_exceeds_entered_amount():
    holdings = [
        {"symbol": "EUNL.DE", "name": "World", "tier": "FORTRESS", "value_eur": 500.0},
        {"symbol": "SXRV.DE", "name": "Nasdaq", "tier": "ALPHA", "value_eur": 100.0},
    ]
    plan = propose_by_class(1000.0, holdings, "balanced")
    assert plan["total"] <= 1000.0 + 1e-9
    assert sum(o["amount_eur"] for o in plan["orders"]) == plan["total"]


def test_savings_plan_eur_month_is_separate():
    """The Long-term savings-plan increase is a separate EUR/m dimension."""
    holdings = [
        {"symbol": "SXRV.DE", "name": "Nasdaq", "tier": "ALPHA", "value_eur": 1000.0},
    ]
    plan = propose_by_class(500.0, holdings, "balanced")
    assert plan["savings_plan_eur_month"] > 0
    # The one-off total never includes the savings-plan EUR/m.
    assert plan["total"] == sum(o["amount_eur"] for o in plan["orders"])
    assert all(o["how"] == "Savings plan" for o in plan["savings_plan"])


def test_class_now_sums_to_100_percent():
    holdings = [
        {"symbol": "EUNL.DE", "tier": "FORTRESS", "value_eur": 500.0},
        {"symbol": "SXRV.DE", "tier": "ALPHA", "value_eur": 300.0},
        {"symbol": "5J50.DE", "tier": "SPECULATIVE", "value_eur": 200.0},
    ]
    plan = propose_by_class(100.0, holdings, "balanced")
    assert abs(sum(r["now_pct"] for r in plan["classes"]) - 100.0) < 1e-6


# ── A1: the canonical class words ─────────────────────────────────────────────

def test_class_labels_are_canonical():
    from quant.ui.terminology import CLASS_LABELS

    assert CLASS_LABELS == {"FORTRESS": "Long-term", "ALPHA": "Active",
                            "SPECULATIVE": "Speculative"}


def test_no_banned_class_synonyms_in_copy():
    """A1: no 'Small bets' or 'FORTRESS' in a copy string value."""
    copy_py = _QUANT / "ui" / "copy.py"
    tree = ast.parse(copy_py.read_text(encoding="utf-8"))
    key_ids = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant):
                    key_ids.add(id(k))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in key_ids:
                continue
            v = node.value
            if "Small bets" in v or re.search(r"\bFORTRESS\b", v):
                offenders.append(v)
    assert not offenders, f"banned class synonym in copy: {offenders}"


@pytest.mark.parametrize("page", ["pages/portfolio.py", "pages/analysis.py"])
def test_rendered_pages_have_no_banned_class_words(page):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    dash = str(Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")
    at = AppTest.from_file(dash, default_timeout=60)
    at.run()
    at.switch_page(page).run()
    assert not at.exception, f"{page} raised: {at.exception}"
    chunks = []
    for attr in ("markdown", "info", "warning", "error", "caption", "text",
                 "title", "header", "subheader"):
        for el in getattr(at, attr, []):
            chunks.append(str(getattr(el, "value", "")))
    for el in getattr(at, "dataframe", []):
        v = getattr(el, "value", None)
        if v is not None:
            try:
                chunks.append(str(v.to_string()))
            except Exception:  # noqa: BLE001
                chunks.append(str(v))
    text = " ".join(chunks)
    assert "Small bets" not in text, f"'Small bets' rendered on {page}"
    assert not re.search(r"\bFORTRESS\b", text), f"'FORTRESS' rendered on {page}"
