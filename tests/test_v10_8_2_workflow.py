"""test_v10_8_2_workflow.py — the end-to-end one workflow (v10.8.2, section 12).

Empty database -> enter holdings -> Review -> Confirm -> History -> the list
updates -> Telegram sends once, then nothing, then a second message, then none.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from quant.engine import notify
from quant.engine.confirm import confirm_save
from quant.engine.diff import diff_tables


def _row(symbol, entry, value, profit):
    return {"Symbol": symbol, "Avg_Entry_Price": entry,
            "Current_Value_EUR": value, "Broker_PnL_EUR": profit}


def _df(rows):
    return pd.DataFrame(rows, columns=["Symbol", "Avg_Entry_Price",
                                       "Current_Value_EUR", "Broker_PnL_EUR"])


def _setup(tmp_path, monkeypatch):
    from quant import paths

    monkeypatch.setattr(paths, "DATA_PORTFOLIO", str(tmp_path / "portfolio.csv"))
    monkeypatch.setattr(paths, "DATA_TIERS", str(tmp_path / "tiers.csv"))
    monkeypatch.setattr(paths, "DATA_ACCOUNT", str(tmp_path / "account.yaml"))
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path / "outputs")
    return paths


def test_empty_database_review_shows_nothing_to_compare(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    out = diff_tables([], [])
    assert out["summary"]["changed"] is False


def test_enter_holdings_confirm_then_history(tmp_path, monkeypatch):
    paths = _setup(tmp_path, monkeypatch)
    from quant.data import database

    conn = database.get_connection()
    for sym in ("EUNL.DE", "SXRV.DE", "AMZN", "5J50.DE"):
        conn.execute("DELETE FROM position_snapshots WHERE symbol = ?", [sym])
        conn.execute("DELETE FROM flows WHERE symbol = ?", [sym])
        conn.execute("DELETE FROM trades WHERE symbol = ?", [sym])

    rows = [_row("EUNL.DE", 95.5, 287.58, 9.58),
            _row("SXRV.DE", 85.2, 271.08, 17.18),
            _row("AMZN", 150.0, 150.2, 0.2),
            _row("5J50.DE", 151.0, 143.37, -7.63)]
    diff = diff_tables([], rows, prices={"EUNL.DE": 100.0, "SXRV.DE": 100.0,
                                         "AMZN": 150.0, "5J50.DE": 140.0})
    assert sum(1 for c in diff["changes"] if c["kind"] == "Bought") == 4
    res = confirm_save(_df(rows), diff, when=date(2026, 10, 8),
                       path=paths.DATA_PORTFOLIO)
    assert res["ok"] is True
    assert (tmp_path / "portfolio.csv").exists()
    assert conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] >= 4

    # Edit one row (add shares) and delete another -> one Bought, one Sold.
    new = [_row("EUNL.DE", 95.5, 383.0, 9.58),   # more shares
           _row("SXRV.DE", 85.2, 271.08, 17.18),
           _row("AMZN", 150.0, 150.2, 0.2)]        # 5J50.DE removed
    diff2 = diff_tables(rows, new, prices={"5J50.DE": 140.0})
    kinds = {c["kind"] for c in diff2["changes"]}
    assert "Bought" in kinds and "Sold" in kinds
    sold = next(c for c in diff2["changes"] if c["kind"] == "Sold")
    assert sold["estimate"] is True
    assert diff2["summary"]["realized_eur"] != 0.0

    for sym in ("EUNL.DE", "SXRV.DE", "AMZN", "5J50.DE"):
        conn.execute("DELETE FROM position_snapshots WHERE symbol = ?", [sym])
        conn.execute("DELETE FROM flows WHERE symbol = ?", [sym])
        conn.execute("DELETE FROM trades WHERE symbol = ?", [sym])


def test_telegram_sends_once_then_caps(tmp_path, monkeypatch):
    monkeypatch.setattr(notify.paths, "OUTPUTS_DIR", tmp_path)
    calls = []

    def _sender(text, config=None):
        calls.append(text)
        return True

    d = date(2026, 10, 8)
    a = [{"group": "Recommended", "verb": "buy", "label": "Name (AAA)",
          "amount_eur": 100.0, "symbol": "AAA"}]
    b = [{"group": "Recommended", "verb": "buy", "label": "Name (AAA)",
          "amount_eur": 120.0, "symbol": "AAA"}]
    c = [{"group": "Recommended", "verb": "buy", "label": "Name (AAA)",
          "amount_eur": 140.0, "symbol": "AAA"}]
    assert notify.notify_decisions(a, today=d, sender=_sender) is True
    assert notify.notify_decisions(a, today=d, sender=_sender) is False  # unchanged
    assert notify.notify_decisions(b, today=d, sender=_sender) is True   # changed
    assert notify.notify_decisions(c, today=d, sender=_sender) is False  # cap
    assert len(calls) == 2
