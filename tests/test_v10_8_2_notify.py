"""test_v10_8_2_notify.py — the Telegram-only notify rule (v10.8.2, section 7).

After a confirmed save or the daily check, send one message when there is
something to do AND the list changed since the last message; never more than two
per local calendar day; never a message when there is nothing to do.
"""
from __future__ import annotations

from datetime import date

from quant.engine import notify


def _item(symbol, amount, verb="buy"):
    return {"group": "Recommended", "verb": verb, "label": f"Name ({symbol})",
            "amount_eur": amount, "symbol": symbol}


class _Sender:
    def __init__(self):
        self.calls = 0
        self.last = ""
    def __call__(self, text, config=None):
        self.calls += 1
        self.last = text
        return True


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(notify.paths, "OUTPUTS_DIR", tmp_path)


def test_first_message_sends_once(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    s = _Sender()
    items = [_item("AAA", 100.0), _item("BBB", 50.0, "sell_part")]
    assert notify.notify_decisions(items, today=date(2026, 10, 8), sender=s) is True
    assert s.calls == 1
    assert s.last.splitlines()[0] == "Quant-AI: 2 things to do"
    assert len(s.last.splitlines()) <= 8


def test_unchanged_list_does_not_resend(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    s = _Sender()
    items = [_item("AAA", 100.0)]
    d = date(2026, 10, 8)
    assert notify.notify_decisions(items, today=d, sender=s) is True
    assert notify.notify_decisions(items, today=d, sender=s) is False
    assert s.calls == 1


def test_changed_list_sends_a_second_message_then_caps(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    s = _Sender()
    d = date(2026, 10, 8)
    a = [_item("AAA", 100.0)]
    b = [_item("AAA", 120.0)]           # changed
    c = [_item("AAA", 140.0)]           # changed again
    assert notify.notify_decisions(a, today=d, sender=s) is True   # 1st
    assert notify.notify_decisions(b, today=d, sender=s) is True   # 2nd
    assert notify.notify_decisions(c, today=d, sender=s) is False  # cap reached
    assert s.calls == 2


def test_nothing_to_do_sends_nothing(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    s = _Sender()
    keep = [{"group": "Optional", "verb": "keep", "label": "Name (AAA)",
             "amount_eur": None, "symbol": "AAA"}]
    assert notify.notify_decisions(keep, today=date(2026, 10, 8), sender=s) is False
    assert s.calls == 0


def test_fingerprint_rounds_amounts_to_ten_euro(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    assert (notify.message_fingerprint([_item("AAA", 101.0)])
            == notify.message_fingerprint([_item("AAA", 104.0)]))
    assert (notify.message_fingerprint([_item("AAA", 101.0)])
            != notify.message_fingerprint([_item("AAA", 200.0)]))


def test_no_email_channel_and_send_is_telegram_only():
    assert notify.send("hi", {"channel": "none"}) is False
    assert notify.send("hi", {"channel": "email"}) is False  # email removed
