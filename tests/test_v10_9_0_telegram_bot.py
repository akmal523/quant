"""test_v10_9_0_telegram_bot.py — holdings and actions from Telegram (v10.9.0; v10.9.1).

The bot proposes a table change with the Review diff, saves it only on /confirm
(through Confirm and save), marks ONE-list items done through Mark done, and
answers only the configured chat. The Bot API is stubbed at the network
boundary; the table, ledger and advice log are the real ones (isolated session
DB, tmp data files).
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from quant.engine import telegram_bot as B


def _setup(tmp_path, monkeypatch, rows=()):
    from quant import paths

    monkeypatch.setattr(paths, "DATA_PORTFOLIO", str(tmp_path / "portfolio.csv"))
    monkeypatch.setattr(paths, "DATA_TIERS", str(tmp_path / "tiers.csv"))
    monkeypatch.setattr(paths, "DATA_ACCOUNT", str(tmp_path / "account.yaml"))
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path / "outputs")
    if rows:
        pd.DataFrame(list(rows), columns=[
            "Symbol", "Avg_Entry_Price", "Current_Value_EUR", "Broker_PnL_EUR"]
        ).to_csv(paths.DATA_PORTFOLIO, index=False)
    return paths


def _row(sym, entry, value, profit):
    return {"Symbol": sym, "Avg_Entry_Price": entry, "Current_Value_EUR": value,
            "Broker_PnL_EUR": profit}


def _cleanup(*symbols):
    from quant.data import database

    conn = database.get_connection()
    for table in ("position_snapshots", "flows", "trades"):
        for s in symbols:
            conn.execute(f"DELETE FROM {table} WHERE symbol = ?", [s])
    return conn


# ── Parsing and the table math (pure) ─────────────────────────────────────────

def test_decimal_comma_is_accepted():
    assert B._num("12,5") == 12.5
    assert B._num("100 EUR") == 100.0
    with pytest.raises(ValueError, match="Not a number: abc"):
        B._num("abc")


def test_buy_keeps_profit_and_moves_the_average_cost():
    rows = [_row("AAA", 10.0, 120.0, 20.0)]           # 100 invested, 10 shares
    out = B._apply_buy(rows, ["aaa", "50", "12,5"])   # 4 shares at 12.5
    r = out[0]
    assert r["Current_Value_EUR"] == pytest.approx(170.0)
    assert r["Broker_PnL_EUR"] == pytest.approx(20.0)
    assert r["Avg_Entry_Price"] == pytest.approx(150.0 / 14.0)
    assert rows[0]["Current_Value_EUR"] == 120.0      # the input is not mutated


def test_buy_of_a_new_symbol_adds_a_row():
    out = B._apply_buy([], ["NEW", "100", "20"])
    assert out == [{"Symbol": "NEW", "Avg_Entry_Price": 20.0,
                    "Current_Value_EUR": 100.0, "Broker_PnL_EUR": 0.0,
                    "Plan_EUR_month": None}]


def test_sell_keeps_the_average_cost_and_reduces_invested():
    rows = [_row("AAA", 10.0, 120.0, 20.0)]            # 10 shares
    r = B._apply_sell(rows, ["AAA", "60", "12"])[0]    # 5 shares sold
    assert r["Avg_Entry_Price"] == 10.0
    assert r["Current_Value_EUR"] == pytest.approx(60.0)
    assert r["Broker_PnL_EUR"] == pytest.approx(60.0 - 50.0)


def test_sell_all_or_more_than_held_removes_the_row():
    rows = [_row("AAA", 10.0, 120.0, 20.0), _row("BBB", 5.0, 50.0, 0.0)]
    assert [r["Symbol"] for r in B._apply_sell(rows, ["AAA", "all"])] == ["BBB"]
    assert [r["Symbol"] for r in B._apply_sell(rows, ["AAA", "500", "12"])] == ["BBB"]


def test_sell_of_an_unknown_symbol_says_so():
    with pytest.raises(ValueError, match="ZZZ is not in your holdings"):
        B._apply_sell([], ["zzz", "all"])


def test_set_needs_a_cost_for_a_new_holding():
    with pytest.raises(ValueError, match="COST is needed"):
        B._apply_set([], ["NEW", "100", "5"])
    out = B._apply_set([_row("AAA", 10.0, 120.0, 20.0)], ["AAA", "130", "30"])
    assert out[0]["Avg_Entry_Price"] == 10.0 and out[0]["Current_Value_EUR"] == 130.0


def test_buy_without_a_price_asks_for_one():
    with pytest.raises(ValueError, match="No price for NOPRICE"):
        B._apply_buy([], ["NOPRICE", "100"])


# ── Review -> /confirm -> Confirm and save ────────────────────────────────────

def test_set_then_confirm_saves_the_table_and_the_ledger(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    conn = _cleanup("TGB1")
    s = B.Session()
    reply = B.handle("/set TGB1 180 30 150", s)
    assert "Bought TGB1" in reply and B.PENDING_HINT in reply
    assert not (tmp_path / "portfolio.csv").exists()     # nothing written yet
    assert B.handle("/confirm", s) == B.SAVED.format(n=1)
    assert "TGB1" in (tmp_path / "portfolio.csv").read_text(encoding="utf-8")
    assert conn.execute("SELECT action, amount_eur FROM trades WHERE symbol = 'TGB1'"
                        ).fetchall() == [("buy", 150.0)]
    assert B.handle("/confirm", s) == B.NO_PENDING
    _cleanup("TGB1")


def test_changes_stack_until_confirm_and_cancel_drops_them(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, [_row("TGB2", 10.0, 100.0, 0.0)])
    s = B.Session()
    B.handle("/buy TGB2 50 10", s)
    reply = B.handle("/buy TGB2 50 10", s)
    assert "Bought TGB2: 100.00 EUR" in reply
    assert B.handle("/cancel", s) == B.CANCELLED
    assert s.pending is None
    assert B.handle("/confirm", s) == B.NO_PENDING


def test_invalid_numbers_never_create_a_pending_change(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    s = B.Session()
    assert B.handle("/set X 1 2 three", s) == "Not a number: three"
    assert s.pending is None


# ── /todo and /done -> Mark done ──────────────────────────────────────────────

def test_done_marks_the_numbered_item_done(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from quant.engine import advice_state

    s = B.Session()
    assert B.handle("/done 1", s) == B.TODO_FIRST
    s.todo = [{"verb": "sell_part", "symbol": "TGB3", "horizon": "Today",
               "amount_eur": 50.0, "reason": "over target", "class": "ALPHA",
               "label": "TGB3", "id": None}]
    assert B.handle("/done 2", s) == B.TODO_RANGE.format(n=2, last=1)
    assert B.handle("/done 1", s).startswith("Marked done:")
    rows = advice_state._query(None, "SELECT status FROM advice_log WHERE symbol = ?",
                               ["TGB3"])
    assert rows == [(advice_state.STATUS_DONE,)]
    assert s.todo == []
    assert "TGB3" in B.handle("/history", s)


# ── Polling: the configured chat only, offset persisted first ─────────────────

def test_poll_answers_only_the_configured_chat(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    sent = []

    def api(token, method, params, timeout):
        if method == "getUpdates":
            return {"ok": True, "result": [
                {"update_id": 7, "message": {"chat": {"id": 999}, "text": "/help"}},
                {"update_id": 8, "message": {"chat": {"id": 42}, "text": "/help"}},
            ]}
        sent.append(params)
        return {"ok": True}

    cfg = {"telegram_bot_token": "T", "telegram_chat_id": "42"}
    assert B.poll_once(cfg, B.Session(), 0, api=api) == 9
    assert sent == [{"chat_id": "42", "text": B.HELP}]
    assert B._read_state()["offset"] == 9


def test_poll_failure_raises_for_the_backoff():
    with pytest.raises(ConnectionError):
        B.poll_once({}, B.Session(), 0, api=lambda *a: None)


def test_unknown_command_and_help():
    s = B.Session()
    assert B.handle("hello", s) == B.UNKNOWN
    assert B.handle("/help@QuantBot", s) == B.HELP
    assert B.handle("/commands", s) == B.HELP


# ── v10.9.1: the "/" menu, restarts, messages that waited ─────────────────────

def test_menu_lists_every_command_in_the_help():
    calls = []

    def api(token, method, params, timeout):
        calls.append((method, json.loads(params["commands"])))
        return {"ok": True}

    assert B.set_menu({"telegram_bot_token": "T"}, api=api) is True
    method, commands = calls[0]
    assert method == "setMyCommands"
    names = [c["command"] for c in commands]
    assert "commands" in names and "todo" in names
    for name in names:
        assert f"/{name}" in B.HELP
        assert B.handle(f"/{name}", B.Session()) != B.UNKNOWN
    assert all(len(c["description"]) <= 256 for c in commands)


def test_pending_change_survives_a_restart(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, [_row("TGB4", 10.0, 100.0, 0.0)])

    def api_with(text, update_id):
        def api(token, method, params, timeout):
            if method == "getUpdates":
                return {"ok": True, "result": [{"update_id": update_id, "message": {
                    "chat": {"id": 42}, "date": 1000, "text": text}}]}
            return {"ok": True}
        return api

    cfg = {"telegram_bot_token": "T", "telegram_chat_id": "42"}
    B.poll_once(cfg, B.Session(), 0, api=api_with("/buy TGB4 50 10", 1), now=1000)
    restarted = B.Session(B._read_state())          # a new process
    assert restarted.pending is not None
    assert restarted.pending["diff"]["changes"][0]["symbol"] == "TGB4"
    B.handle("/cancel", restarted)


def test_a_message_that_waited_says_so():
    msg = {"date": 1_000_000}
    assert B._late_note(msg, 1_000_000 + B.LATE_SECONDS - 1) == ""
    note = B._late_note(msg, 1_000_000 + 3600)
    assert note.startswith("Sent ") and "while the computer was off" in note
    assert B._late_note({}, 5.0) == ""


def test_bot_status_without_systemd(tmp_path, monkeypatch):
    from quant.engine import scheduler

    monkeypatch.setattr(scheduler, "systemd_user_dir", lambda: str(tmp_path))
    assert scheduler.bot_status_line() == "not running in the background"
    assert scheduler.restart_bot() == "not installed; run quant bot --install"
    (tmp_path / scheduler.BOT_SERVICE_NAME).write_text("x", encoding="utf-8")
    monkeypatch.setattr(scheduler, "bot_running", lambda: False)
    assert scheduler.bot_status_line() == "installed but stopped"


def test_run_refuses_without_telegram_setup(capsys):
    assert B.run({"channel": "none"}) == 2


def test_bot_unit_runs_the_bot_command():
    from quant.engine.scheduler import build_bot_unit

    unit = build_bot_unit("/venv/bin/python", "/code")
    assert "ExecStart=/venv/bin/python -m quant.cli bot" in unit
    assert "Restart=on-failure" in unit
