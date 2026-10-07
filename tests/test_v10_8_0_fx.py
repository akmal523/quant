"""test_v10_8_0_fx.py — price_in_eur converts native prices to EUR (v10.8.0, 2.2).

A USD stock and a GBX line must convert; the as-of lookup must pick the close on
or before the date; a missing symbol returns None.
"""
from __future__ import annotations

import pytest


def _seed(rows):
    from quant.data import database

    database.init_db()
    with database.write_connection() as conn:
        for sym, d, close in rows:
            conn.execute(
                "INSERT OR REPLACE INTO market_history (Date, Close, Symbol) "
                "VALUES (?, ?, ?)",
                [d, close, sym],
            )


def test_price_in_eur_eur_is_identity(monkeypatch):
    from quant.data import currency

    monkeypatch.setattr(currency, "get_fx_to_eur", lambda s: 1.0)
    _seed([("EUNL.DE", "2026-10-01", 100.0)])
    assert currency.price_in_eur("EUNL.DE") == 100.0


def test_price_in_eur_usd_converts(monkeypatch):
    from quant.data import currency

    monkeypatch.setattr(currency, "get_fx_to_eur", lambda s: 0.9)
    _seed([("AMZN", "2026-10-01", 200.0)])
    assert currency.price_in_eur("AMZN") == pytest.approx(180.0)


def test_price_in_eur_gbx_converts(monkeypatch):
    from quant.data import currency

    monkeypatch.setattr(currency, "get_fx_to_eur", lambda s: 0.0117)
    _seed([("VOD.L", "2026-10-01", 100.0)])
    assert currency.price_in_eur("VOD.L") == pytest.approx(1.17)


def test_price_in_eur_as_of(monkeypatch):
    from quant.data import currency

    monkeypatch.setattr(currency, "get_fx_to_eur", lambda s: 1.0)
    _seed([("EUNL.DE", "2026-10-01", 100.0),
           ("EUNL.DE", "2026-10-05", 110.0)])
    assert currency.price_in_eur("EUNL.DE", as_of="2026-10-03") == 100.0
    assert currency.price_in_eur("EUNL.DE") == 110.0


def test_price_in_eur_missing_returns_none(monkeypatch):
    from quant.data import currency

    monkeypatch.setattr(currency, "get_fx_to_eur", lambda s: 1.0)
    assert currency.price_in_eur("NOPE") is None
