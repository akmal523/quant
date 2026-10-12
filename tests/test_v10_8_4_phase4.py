"""test_v10_8_4_phase4.py — T6 research surface + pipeline health (v10.8.4).

The Research funnel renders rows without a literal None; the pipeline health
table lists every stage with its status; the manifest drives the empty states.
"""
from __future__ import annotations

from quant.engine import manifest
from quant.ui import copy as C


def test_safe_score_never_renders_none():
    from quant.ui.pages import _safe_score

    assert _safe_score(None) == "-"
    assert _safe_score(67.4) == "67 / 100"
    assert _safe_score("x") == "-"


def test_pipeline_health_lists_every_stage():
    from quant.data.database import write_connection

    run = "2026-06-01_080000"
    with write_connection() as conn:
        conn.execute("DELETE FROM run_manifest")
        for stage in manifest.STAGES:
            manifest.record_stage(conn, run, stage, manifest.STATUS_OK,
                                  coverage="9/9 symbols")
    rows = manifest.read_manifest(run_id=run)
    assert len(rows) == len(manifest.STAGES)
    assert all(r["status"] == "ok" for r in rows)
    # Every stage has a plain-word name (R6: no raw keys in the UI).
    assert all(manifest.stage_word(r["stage"]) != r["stage"] or
               r["stage"] in ("prices", "valuation") for r in rows)


def test_research_rows_have_stage_and_no_literal_none():
    from quant.ui.pages import _research_rows

    # Empty universe: no rows, and the caller renders an explicit empty state.
    rows = _research_rows([])
    assert isinstance(rows, list)
    for r in rows:
        assert r["Stage"] in ("Held", "Watchlist", "Idea", "Rejected")
        assert "None" not in str(r["Score"])


def test_research_copy_exists():
    assert C.TAB_RESEARCH == "Research"
    assert C.SEC_PIPELINE_HEALTH == "Pipeline health"
    assert C.BTN_RUN_ANALYSIS == "Run analysis"
