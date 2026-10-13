"""test_v10_8_7_phase2.py — v10.8.7 Phase 2: T2 (scoring + manifest).

R-C2: scoring degrades gracefully and NEVER renders a dash. Any subset of
components yields a numeric composite + a confidence tier; a missing component is
a documented neutral penalty.

A3: the analysis chain (prices -> fundamentals -> nlp -> regime -> scoring ->
optimization) writes one run_manifest row per stage; fetch stages reuse data
fresher than their TTL; compute stages always recompute.

Synthetic fixtures only; the user's real DB and data dir are never touched.
"""
from __future__ import annotations

from datetime import date

import pytest

from quant.analytics.scoring import composite_score, confidence_tier

# ── R-C2: the pure composite ──────────────────────────────────────────────────

def test_composite_is_numeric_with_only_technical():
    """Prices-only: a numeric composite + confidence 'low', never a dash."""
    result = composite_score({"technical": 60.0, "fundamental": None,
                              "sentiment": None, "stewardship": None})
    assert isinstance(result["composite"], float)
    assert 0.0 <= result["composite"] <= 100.0
    assert result["confidence"] == "low"
    assert result["present"] == 1 and result["total"] == 4


def test_composite_high_confidence_when_all_present():
    result = composite_score({"technical": 70.0, "fundamental": 80.0,
                              "sentiment": 60.0, "stewardship": 50.0})
    assert result["confidence"] == "high"
    assert result["present"] == 4


def test_composite_never_aborts_on_empty():
    result = composite_score({})
    assert isinstance(result["composite"], float)
    assert result["confidence"] == "low"


def test_confidence_tier_mapping():
    assert confidence_tier(4, 4) == "high"
    assert confidence_tier(3, 4) == "medium"
    assert confidence_tier(1, 4) == "low"
    assert confidence_tier(0, 0) == "low"


# ── A3: the pipeline + manifest on a prices-only fixture ──────────────────────

_SYMS = ("AAA", "BBB", "CCC")
_TODAY = date.today()


@pytest.fixture
def prices_only():
    """A prices-only universe: no fundamentals, no NLP."""
    from quant.data.database import write_connection

    with write_connection() as conn:
        conn.execute("DELETE FROM scores")
        conn.execute("DELETE FROM run_manifest")
        for sym in _SYMS:
            conn.execute("DELETE FROM asset_registry WHERE symbol = ?", [sym])
            conn.execute("DELETE FROM market_history WHERE Symbol = ?", [sym])
            conn.execute("DELETE FROM fundamentals WHERE symbol = ?", [sym])
            conn.execute("DELETE FROM nlp_evidence WHERE symbol = ?", [sym])
            conn.execute(
                "INSERT INTO asset_registry (symbol, name, instrument_class, "
                "universe_status) VALUES (?, ?, 'EQUITY', 'WATCHLIST')",
                [sym, sym])
            for i, d in enumerate((date(2026, 9, 1), date(2026, 9, 2), _TODAY)):
                conn.execute(
                    "INSERT OR REPLACE INTO market_history (Symbol, Date, Close) "
                    "VALUES (?, ?, ?)", [sym, str(d), 100.0 + i])
    yield
    with write_connection() as conn:
        for sym in _SYMS:
            conn.execute("DELETE FROM asset_registry WHERE symbol = ?", [sym])
            conn.execute("DELETE FROM market_history WHERE Symbol = ?", [sym])


def test_run_analysis_scores_every_watched_symbol(prices_only):
    """Every watched symbol gets a numeric composite with confidence 'low'."""
    from quant.data.database import write_connection
    from quant.engine import analysis

    with write_connection() as conn:
        res = analysis.run_analysis(conn, symbols=list(_SYMS))
        rows = conn.execute(
            "SELECT symbol, composite, confidence FROM scores ORDER BY symbol"
        ).fetchall()
    assert res["scored"] == len(_SYMS)
    assert len(rows) == len(_SYMS)
    for _sym, comp, conf in rows:
        assert comp is not None and 0.0 <= float(comp) <= 100.0
        assert conf == "low"


def test_run_analysis_writes_one_manifest_row_per_stage(prices_only):
    """Every stage is recorded; scoring is ok with coverage N/N."""
    from quant.data.database import write_connection
    from quant.engine import analysis, manifest

    with write_connection() as conn:
        analysis.run_analysis(conn, symbols=list(_SYMS))
        rows = {r["stage"]: r for r in manifest.read_manifest(conn)}
    for stage in ("prices", "fundamentals", "nlp", "regime", "scoring",
                  "optimization"):
        assert stage in rows, f"missing manifest row for {stage}"
    assert rows["scoring"]["status"] == manifest.STATUS_OK
    assert rows["scoring"]["coverage"] == f"{len(_SYMS)}/{len(_SYMS)} symbols"
    # No fundamentals / NLP in the fixture: flagged missing, not silently skipped.
    assert rows["fundamentals"]["status"] == manifest.STATUS_MISSING
    assert rows["nlp"]["status"] == manifest.STATUS_MISSING


def test_regime_empty_state_is_reasoned(prices_only):
    """A short history yields a manifest-reasoned regime empty state, not a crash."""
    from quant.data.database import write_connection
    from quant.engine import analysis, manifest

    with write_connection() as conn:
        analysis.run_analysis(conn, symbols=list(_SYMS))
        rows = {r["stage"]: r for r in manifest.read_manifest(conn)}
    assert rows["regime"]["status"] == manifest.STATUS_MISSING
    assert rows["regime"]["error"], "the regime empty state must carry a reason"


def test_fetch_stage_reuses_fresh_data(prices_only):
    """A fresh price store is reused (ok-reused), not re-fetched (A3)."""
    from quant.data.database import write_connection
    from quant.engine import analysis, manifest

    with write_connection() as conn:
        analysis.run_analysis(conn, symbols=list(_SYMS))
        rows = {r["stage"]: r for r in manifest.read_manifest(conn)}
    assert rows["prices"]["status"] == analysis.STATUS_OK_REUSED


def test_fetch_stage_calls_fetcher_when_stale(prices_only):
    """A stale price store triggers the injected fetcher (ok-fresh)."""
    from quant.data.database import write_connection
    from quant.engine import analysis, manifest

    calls = {"n": 0}

    def _fetch():
        calls["n"] += 1

    with write_connection() as conn:
        # Force staleness by asking for a future "now".
        analysis.run_analysis(conn, symbols=list(_SYMS),
                              now=date(2030, 1, 1), fetchers={"prices": _fetch})
        rows = {r["stage"]: r for r in manifest.read_manifest(conn)}
    assert calls["n"] == 1
    assert rows["prices"]["status"] == analysis.STATUS_OK_FRESH
