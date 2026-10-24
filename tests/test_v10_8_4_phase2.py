"""test_v10_8_4_phase2.py — T5 advice state/pairing + T4 reminder (v10.8.4).

T5: advice is a stateful object; a repeated item keeps open_since and shows it;
a finished item stops repeating and lands in History. A sell is paired with where
its proceeds go (underweight -> idea -> untracked hold; v10.8.7, R-C1).
T4: the savings-plan reminder triggers, auto-clears when the flow lands, and can
be dismissed persistently.
"""
from __future__ import annotations

from datetime import date

from quant.engine import advice_state
from quant.engine.decisions import open_since_line
from quant.engine.pairing import (
    ROUTE_HOLD,
    ROUTE_IDEA,
    ROUTE_UNDERWEIGHT,
    paired_line,
    route_sell,
)
from quant.engine.reminders import dismiss, savings_plan_reminder

# ── T5: pairing priority ──────────────────────────────────────────────────────

def _holdings():
    return [
        {"symbol": "EUNL.DE", "name": "MSCI World", "current_weight": 0.30,
         "target_weight": 0.40, "value_eur": 300.0},
        {"symbol": "SXRV.DE", "name": "Nasdaq 100", "current_weight": 0.30,
         "target_weight": 0.20, "value_eur": 300.0},
    ]


def test_route_prefers_largest_underweight():
    route = route_sell({"symbol": "SXRV.DE", "amount_eur": 65.0}, _holdings())
    assert route["kind"] == ROUTE_UNDERWEIGHT
    assert route["symbol"] == "EUNL.DE"
    assert "under target" in route["text"]


def test_route_falls_back_to_best_idea():
    # Only the overweight holding exists, but there is a candidate idea.
    held = [{"symbol": "SXRV.DE", "current_weight": 0.30, "target_weight": 0.20,
             "value_eur": 300.0}]
    route = route_sell({"symbol": "SXRV.DE", "amount_eur": 65.0}, held,
                       candidates=[{"symbol": "AMZN", "name": "Amazon",
                                    "score": 88.0}])
    assert route["kind"] == ROUTE_IDEA
    assert route["symbol"] == "AMZN"


def test_route_falls_back_to_untracked_hold():
    """v10.8.7 (R-C1): no tracked-cash target; the proceeds stay at the broker."""
    route = route_sell({"symbol": "SXRV.DE", "amount_eur": 65.0},
                       [{"symbol": "SXRV.DE", "current_weight": 0.3,
                         "target_weight": 0.2, "value_eur": 300.0}])
    assert route["kind"] == ROUTE_HOLD
    assert "untracked" in route["text"]
    assert "cash target" not in route["text"]


def test_paired_line_is_one_item_with_a_route():
    line = paired_line({"symbol": "SXRV.DE", "label": "Nasdaq 100 (SXRV.DE)",
                        "amount_eur": 65.0}, _holdings())
    assert line.startswith("Sell ~65 EUR of Nasdaq 100 (SXRV.DE) ->")


def test_route_never_buys_the_sold_symbol():
    route = route_sell({"symbol": "EUNL.DE", "amount_eur": 10.0}, _holdings())
    assert route.get("symbol") != "EUNL.DE"


# ── T5: the stateful advice log ───────────────────────────────────────────────

def _item(symbol="EUNL.DE", verb="buy", horizon="This month"):
    return {"verb": verb, "symbol": symbol, "amount_eur": 100.0,
            "reason": "underweight", "horizon": horizon, "class": "FORTRESS"}


def test_repeated_advice_keeps_open_since():
    from quant.data.database import write_connection

    with write_connection() as conn:
        conn.execute("DELETE FROM advice_log")
        day1 = advice_state.sync(conn, [_item()], as_of=date(2026, 5, 1))
        day5 = advice_state.sync(conn, [_item()], as_of=date(2026, 5, 5))
    assert day1[0]["open_since"] == date(2026, 5, 1)
    # The repeated item keeps its ORIGINAL open date (not today).
    assert day5[0]["open_since"] == date(2026, 5, 1)
    assert open_since_line(day5[0]) == "open since 1 May 2026"


def test_finished_advice_is_closed_and_logged():
    from quant.data.database import write_connection

    with write_connection() as conn:
        conn.execute("DELETE FROM advice_log")
        advice_state.sync(conn, [_item()], as_of=date(2026, 5, 1))
        # Next day the item is gone -> it closes.
        advice_state.sync(conn, [], as_of=date(2026, 5, 2))
        still_open = advice_state.open_items(conn)
        log = advice_state.history(conn)
    assert [i for i in still_open if i["symbol"] == "EUNL.DE"] == []
    assert any(r["symbol"] == "EUNL.DE" and r["status"] == advice_state.STATUS_CLOSED for r in log)


def test_mark_done_stops_repetition():
    from quant.data.database import write_connection

    with write_connection() as conn:
        conn.execute("DELETE FROM advice_log")
        items = advice_state.sync(conn, [_item()], as_of=date(2026, 5, 1))
        rid = items[0]["id"]
        assert advice_state.mark_done(conn, rid) is True
        conn.execute("UPDATE advice_log SET done_at = ? WHERE id = ?",
                     [date(2026, 5, 2), rid])
        # v10.8.10: within a week the done item stays hidden, even if produced...
        assert advice_state.sync(conn, [_item()], as_of=date(2026, 5, 3)) == []
        assert advice_state.annotate([_item()], as_of=date(2026, 5, 3), conn=conn) == []
        # ...after that it starts a NEW open object (the done one stays done).
        again = advice_state.sync(conn, [_item()], as_of=date(2026, 5, 12))
        assert again[0]["id"] != rid
        assert again[0]["open_since"] == date(2026, 5, 12)


# ── T4: the savings-plan reminder lifecycle ───────────────────────────────────

def test_reminder_triggers_after_plan_date_without_a_flow():
    from quant.data.database import write_connection

    with write_connection() as conn:
        conn.execute("DELETE FROM flows")
        conn.execute("DELETE FROM meta WHERE key LIKE 'savings_reminder%'")
        rem = savings_plan_reminder(conn, today=date(2026, 10, 12), day=1,
                                    amount_eur=100.0)
    assert rem is not None
    assert "100 EUR" in rem["message"]
    assert "1 Oct 2026" in rem["message"]


def test_reminder_clears_when_the_flow_lands():
    from quant.data.database import write_connection

    with write_connection() as conn:
        conn.execute("DELETE FROM flows")
        conn.execute("DELETE FROM meta WHERE key LIKE 'savings_reminder%'")
        conn.execute("INSERT INTO flows (date, type, amount_eur, symbol, note) "
                     "VALUES (?, 'buy', 100.0, 'EUNL.DE', 'confirmed')",
                     [date(2026, 10, 5)])
        rem = savings_plan_reminder(conn, today=date(2026, 10, 12), day=1,
                                    amount_eur=100.0)
    assert rem is None


def test_reminder_dismiss_persists():
    from quant.data.database import write_connection

    with write_connection() as conn:
        conn.execute("DELETE FROM flows")
        conn.execute("DELETE FROM meta WHERE key LIKE 'savings_reminder%'")
        first = savings_plan_reminder(conn, today=date(2026, 10, 12), day=1,
                                      amount_eur=100.0)
        assert first is not None
        assert dismiss(conn, today=date(2026, 10, 12), day=1) is True
        # Dismissed: gone across reruns and later days this month.
        assert savings_plan_reminder(conn, today=date(2026, 10, 12), day=1) is None
        assert savings_plan_reminder(conn, today=date(2026, 10, 20), day=1) is None


def test_reminder_not_before_plan_date():
    from quant.data.database import write_connection

    with write_connection() as conn:
        conn.execute("DELETE FROM flows")
        conn.execute("DELETE FROM meta WHERE key LIKE 'savings_reminder%'")
        rem = savings_plan_reminder(conn, today=date(2026, 10, 1), day=5,
                                    amount_eur=100.0)
    assert rem is None
