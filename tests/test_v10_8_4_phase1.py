"""test_v10_8_4_phase1.py — T1 valuation store + T7 manifest (v10.8.4, Phase 1).

T1: the daily mark-to-market store is the single source; shares are reconstructed
from the ledger (buys/sells/splits); writes are idempotent per (date, symbol).
T7: every stage is recorded in run_manifest; an empty state names the failed
stage; a missing NLP input yields a low-confidence score, never a silent None.
"""
from __future__ import annotations

from datetime import date

from quant.engine import manifest
from quant.engine.mtm import (
    reconstruct_from_rows,
    reconstruct_shares,
)

# ── T1: pure reconstruction ───────────────────────────────────────────────────

def test_reconstruct_buys_and_sells_from_ledger():
    opening = {"AAA": (date(2026, 1, 1), 10.0)}
    trades = [
        (date(2026, 2, 1), "AAA", "buy", 5.0),
        (date(2026, 3, 1), "AAA", "sell", 4.0),
    ]
    shares = reconstruct_from_rows(opening, trades, [], date(2026, 6, 1))
    assert shares["AAA"] == 11.0  # 10 + 5 - 4


def test_reconstruct_applies_split_factor():
    opening = {"AAA": (date(2026, 1, 1), 10.0)}
    splits = [(date(2026, 4, 1), "AAA", 2.0)]  # 2:1 split
    shares = reconstruct_from_rows(opening, [], splits, date(2026, 6, 1))
    assert shares["AAA"] == 20.0


def test_reconstruct_ignores_trades_on_or_before_opening():
    # A trade dated on the opening lot's date is already reflected in it.
    opening = {"AAA": (date(2026, 3, 1), 10.0)}
    trades = [(date(2026, 3, 1), "AAA", "buy", 5.0)]
    shares = reconstruct_from_rows(opening, trades, [], date(2026, 6, 1))
    assert shares["AAA"] == 10.0


def test_reconstruct_excludes_future_trades():
    opening = {"AAA": (date(2026, 1, 1), 10.0)}
    trades = [(date(2026, 9, 1), "AAA", "buy", 100.0)]
    shares = reconstruct_from_rows(opening, trades, [], date(2026, 6, 1))
    assert shares["AAA"] == 10.0


# ── T1: the store (idempotent writes) ─────────────────────────────────────────

def test_write_valuation_is_idempotent_per_date_symbol():
    from quant.data.database import write_connection
    from quant.engine import mtm

    d = date(2026, 5, 1)
    price = {"AAA": 10.0, "BBB": 20.0}
    fx = {"AAA": 1.0, "BBB": 0.5}
    with write_connection() as conn:
        conn.execute("DELETE FROM valuation")
        conn.execute("DELETE FROM portfolio_total")
        conn.execute("DELETE FROM holdings_meta")
        conn.execute("DELETE FROM trades")
        conn.execute("DELETE FROM flows")
        conn.execute(
            "INSERT INTO holdings_meta (symbol, shares, sync_date, invested_at_sync) "
            "VALUES ('AAA', 10.0, ?, 100.0), ('BBB', 4.0, ?, 80.0)", [d, d])
        n1 = mtm.write_valuation(conn, d, price_lookup=price.get,
                                 fx_lookup=fx.get)
        n2 = mtm.write_valuation(conn, d, price_lookup=price.get,
                                 fx_lookup=fx.get)
        rows = conn.execute("SELECT symbol, value_eur FROM valuation WHERE date = ? "
                            "ORDER BY symbol", [d]).fetchall()
    assert n1 == 2 and n2 == 2
    # Idempotent: still exactly one row per symbol.
    assert len(rows) == 2
    assert rows[0] == ("AAA", 100.0)   # 10 * 10 * 1.0
    assert rows[1] == ("BBB", 40.0)    # 4 * 20 * 0.5


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


def test_reconstruct_shares_uses_connection_ledger():
    from quant.data.database import write_connection

    with write_connection() as conn:
        conn.execute("DELETE FROM holdings_meta")
        conn.execute("DELETE FROM trades")
        conn.execute("INSERT INTO holdings_meta (symbol, shares, sync_date, "
                     "invested_at_sync) VALUES ('CCC', 2.0, ?, 20.0)",
                     [date(2026, 1, 1)])
        conn.execute("INSERT INTO trades (date, symbol, action, shares) "
                     "VALUES (?, 'CCC', 'buy', 3.0)", [date(2026, 2, 1)])
        shares = reconstruct_shares(conn, date(2026, 6, 1))
    assert shares["CCC"] == 5.0


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


def test_confidence_flag_low_when_nlp_missing():
    assert manifest.confidence_flag(0, 10) == "low"
    assert manifest.confidence_flag(10, 10) == "high"
    assert manifest.confidence_flag(6, 10) == "medium"


def test_scoring_penalises_missing_nlp_but_still_scores():
    """T7: a missing NLP input lowers the grade; the score still exists."""
    from quant.analytics.scoring import evaluate_tactical_grade

    with_data = evaluate_tactical_grade(0.7, 20.0, 0.0, data_confidence=1.0)
    without = evaluate_tactical_grade(0.7, 20.0, 0.0, data_confidence=0.0)
    assert without < with_data
    assert without >= 0.0  # a real score, never None
