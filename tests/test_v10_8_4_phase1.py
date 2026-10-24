"""test_v10_8_4_phase1.py — T1 valuation store + T7 manifest (v10.8.4, Phase 1).

T1: the daily mark-to-market store is the single source; shares are the saved
holdings tables (position_snapshots) restated by splits; the store is rebuilt.
T7: every stage is recorded in run_manifest; an empty state names the failed
stage; a missing NLP input yields a low-confidence score, never a silent None.
"""
from __future__ import annotations

from datetime import date

from quant.engine import manifest
from quant.engine.mtm import shares_on

# ── T1: shares from the saved holdings tables (v10.8.10) ──────────────────────

_SNAPS = [(date(2026, 1, 1), {"AAA": 10.0, "BBB": 3.0}),
          (date(2026, 3, 1), {"AAA": 11.0})]           # BBB sold on the 2nd save


def test_shares_are_the_latest_saved_table_on_or_before_the_day():
    assert shares_on(_SNAPS, [], date(2025, 12, 31)) == {}          # before any save
    assert shares_on(_SNAPS, [], date(2026, 2, 1)) == {"AAA": 10.0, "BBB": 3.0}
    assert shares_on(_SNAPS, [], date(2026, 6, 1)) == {"AAA": 11.0}  # BBB gone


def test_shares_are_restated_by_later_splits_only():
    splits = [(date(2026, 4, 1), "AAA", 2.0)]                       # 2:1
    assert shares_on(_SNAPS, splits, date(2026, 2, 1))["AAA"] == 20.0
    # a table saved after the split already counts the new shares
    after = [(date(2026, 5, 1), {"AAA": 22.0})]
    assert shares_on(after, splits, date(2026, 6, 1)) == {"AAA": 22.0}


# ── T1: the store (idempotent writes) ─────────────────────────────────────────


def test_portfolio_total_and_day_delta_from_store():
    from quant.data.database import write_connection
    from quant.engine import mtm

    with write_connection() as conn:
        conn.execute("DELETE FROM portfolio_total")
        conn.execute("INSERT INTO portfolio_total (date, value_eur, cash_eur) "
                     "VALUES (?, ?, ?), (?, ?, ?)",
                     [date(2026, 5, 1), 100.0, 10.0, date(2026, 5, 2), 110.0, 10.0])
    eur, pct, when = mtm.day_delta()
    # v10.8.7 (R-C1): Total = invested only: 110 - 100 = 10 EUR, 10/100 = 10%.
    assert round(eur, 2) == 10.0
    assert round(pct, 4) == 10.0
    assert when == date(2026, 5, 2)


def test_read_value_series_prefers_the_store():
    from quant.data.database import write_connection
    from quant.engine import mtm  # noqa: F401
    from quant.reporting.artifacts import read_value_series

    with write_connection() as conn:
        conn.execute("DELETE FROM portfolio_total")
        conn.execute("INSERT INTO portfolio_total (date, value_eur, cash_eur) "
                     "VALUES (?, ?, ?)", [date(2026, 5, 3), 123.0, 0.0])
    series = read_value_series()
    assert not series.empty
    assert float(series.iloc[-1]["value_eur"]) == 123.0


# ── T7: the manifest contract ─────────────────────────────────────────────────

def test_manifest_records_and_reads_in_stage_order():
    from quant.data.database import write_connection

    run = "2026-05-01_120000"
    with write_connection() as conn:
        conn.execute("DELETE FROM run_manifest")
        manifest.record_stage(conn, run, "prices", manifest.STATUS_OK,
                              coverage=manifest.coverage_text((10, 10)))
        manifest.record_stage(conn, run, "valuation", manifest.STATUS_OK)
    rows = manifest.read_manifest(run_id=run)
    stages = [r["stage"] for r in rows]
    assert stages == list(manifest.STAGES)
    by = {r["stage"]: r for r in rows}
    assert by["prices"]["status"] == "ok"
    assert by["prices"]["coverage"] == "10/10 symbols"
    # A stage never recorded reads as missing (never a guess).
    assert by["nlp"]["status"] == "missing"


def test_manifest_empty_state_names_the_failed_stage():
    from quant.data.database import write_connection

    run = "2026-05-02_090000"
    with write_connection() as conn:
        conn.execute("DELETE FROM run_manifest")
        manifest.record_stage(conn, run, "nlp", manifest.STATUS_FAILED,
                              error="news source unreachable")
    reason = manifest.empty_state_reason("nlp")
    assert "News sentiment" in reason
    assert "news source unreachable" in reason


def test_manifest_missing_stage_reason_is_not_a_guess():
    from quant.data.database import write_connection

    with write_connection() as conn:
        conn.execute("DELETE FROM run_manifest")
    reason = manifest.empty_state_reason("regime")
    assert "not available yet" in reason
