"""test_v10_8_2_notify.py — the Telegram-only notify rule (v10.8.2, section 7).

After a confirmed save or the daily check, send one message when there is
something to do AND the list changed since the last message; never more than two
per local calendar day; never a message when there is nothing to do.
"""
from __future__ import annotations

from quant.engine import notify


def test_no_email_channel_and_send_is_telegram_only():
    assert notify.send("hi", {"channel": "none"}) is False
    assert notify.send("hi", {"channel": "email"}) is False  # email removed
