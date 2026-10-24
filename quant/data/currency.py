"""
currency.py — EUR conversion for every price the app shows.

A symbol's currency is the one Yahoo reports for it (stored in
``asset_registry.currency`` by refresh); the ticker-suffix guess is only the
fallback for a symbol never fetched. Rates are "units of X per 1 EUR" from Yahoo
(``EURX=X``), then the ECB (frankfurter.dev), then the last live rate persisted
in the data folder. Pence quotes (``GBp``/``GBX``) are divided by 100.

v10.8.10: before, every offline run used EUR/USD = 1.0 (the valuation backfill
then rewrote the whole history of USD holdings at that rate), every ``.L``
listing was treated as pence, and non-USD rates were proxied through USD.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Any

import yfinance as yf

from quant.cli.output import reporter

_LAST_RATES_FILE = "fx_rates_last.json"
_LEGACY_LAST_RATE_FILE = "fx_eurusd_last.json"
_rate_cache: dict[str, float] = {}
_ccy_cache: dict[str, str] = {}
# Plausible "X per EUR" bands; a quote outside is a bad tick, not a rate.
_BANDS = {"USD": (0.8, 1.5), "GBP": (0.6, 1.1), "CHF": (0.7, 1.4)}


def _outputs_file(name: str) -> str:
    import os

    from quant import paths

    return os.path.join(str(paths.OUTPUTS_DIR), name)


def _load_last_rates() -> dict:
    rates: dict = {}
    for name in (_LEGACY_LAST_RATE_FILE, _LAST_RATES_FILE):
        try:
            with open(_outputs_file(name), encoding="utf-8") as f:
                data = json.load(f)
            if "eurusd" in data:
                data = {"USD": data["eurusd"]}
            rates.update({str(k): float(v) for k, v in data.items()})
        except Exception:  # noqa: BLE001
            continue
    return rates


def _save_last_rate(ccy: str, rate: float) -> None:
    """Persist a live rate for offline runs. Never raises."""
    import os

    try:
        rates = _load_last_rates()
        rates[ccy] = rate
        os.makedirs(os.path.dirname(_outputs_file(_LAST_RATES_FILE)), exist_ok=True)
        with open(_outputs_file(_LAST_RATES_FILE), "w", encoding="utf-8") as f:
            json.dump(rates, f)
    except Exception:  # noqa: BLE001
        pass


def _plausible(ccy: str, rate: float) -> bool:
    lo, hi = _BANDS.get(ccy, (1e-6, 1e6))
    return lo < rate < hi


def _yahoo_rate(ccy: str) -> float | None:
    try:
        df = yf.Ticker(f"EUR{ccy}=X").history(period="5d")
        if not df.empty and "Close" in df.columns:
            return float(df["Close"].dropna().iloc[-1])
    except Exception:  # noqa: BLE001
        return None
    return None


def _ecb_rate(ccy: str) -> float | None:
    # Cloudflare answers 403 to urllib's default User-Agent.
    try:
        req = urllib.request.Request(
            f"https://api.frankfurter.dev/v1/latest?from=EUR&to={ccy}",
            headers={"User-Agent": "quant-ai"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return float(json.loads(resp.read())["rates"][ccy])
    except Exception:  # noqa: BLE001
        return None


def eur_rate(ccy: str) -> float | None:
    """Units of ``ccy`` per 1 EUR (live, else the last live rate), or None."""
    ccy = ccy.upper()
    if ccy == "EUR":
        return 1.0
    if ccy in _rate_cache:
        return _rate_cache[ccy]
    for source, fetch in (("Yahoo", _yahoo_rate), ("ECB", _ecb_rate)):
        rate = fetch(ccy)
        if rate is not None and _plausible(ccy, rate):
            _rate_cache[ccy] = rate
            _save_last_rate(ccy, rate)
            reporter.detail(f"  [FX] EUR/{ccy} = {rate:.4f} ({source})")
            return rate
    last = _load_last_rates().get(ccy)
    if last is not None and _plausible(ccy, last):
        reporter.detail(f"  [FX] WARNING: live EUR/{ccy} unavailable; using the "
                        f"last live rate {last:.4f}.")
        _rate_cache[ccy] = last
        return last
    return None


def get_eur_rate() -> float:
    """EUR/USD (USD per EUR). 1.0 only on an install that has never been online."""
    rate = eur_rate("USD")
    if rate is None:
        reporter.detail("  [FX] WARNING: no EUR/USD rate known; using 1.0.")
        return 1.0
    return rate


def deduce_currency(symbol: str) -> str:
    """Guess a listing currency from the ticker suffix (fallback only). Pure.

    ``.L`` is ambiguous on Yahoo (pence, pounds or dollars per line); the stored
    Yahoo currency always wins over this guess.
    """
    if "." not in symbol:
        return "USD"
    suffix = symbol.split(".")[-1].upper()
    if suffix in {"DE", "PA", "AS", "MI", "MC", "BR", "VI", "HE", "F", "IR", "LS"}:
        return "EUR"
    return {"L": "GBp", "SW": "CHF", "CO": "DKK", "OL": "NOK", "ST": "SEK",
            "TO": "CAD", "AX": "AUD", "KS": "KRW"}.get(suffix, "USD")


def symbol_currency(symbol: str) -> str:
    """The currency Yahoo reports for ``symbol`` (stored by refresh), else a guess."""
    if symbol in _ccy_cache:
        return _ccy_cache[symbol]
    ccy = None
    try:
        from quant.data.database import read_only_connection

        with read_only_connection() as conn:
            row = conn.execute("SELECT currency FROM asset_registry WHERE symbol = ? "
                               "AND currency_source = 'yahoo'", [symbol]).fetchone()
        ccy = row[0] if row and row[0] else None
    except Exception:  # noqa: BLE001
        ccy = None
    _ccy_cache[symbol] = ccy or deduce_currency(symbol)
    return _ccy_cache[symbol]


def get_fx_to_eur(symbol: str) -> float:
    """Multiplier from a symbol's quoted price to EUR. Never raises."""
    ccy = symbol_currency(symbol)
    factor = 1.0
    if ccy in ("GBp", "GBX"):          # pence
        factor, ccy = 0.01, "GBP"
    if ccy.upper() == "EUR":
        return factor
    rate = eur_rate(ccy) if ccy.upper() != "USD" else get_eur_rate()
    if rate is None:
        reporter.detail(f"  [FX] WARNING: no EUR/{ccy} rate known; {symbol} unconverted.")
        return factor
    return factor / rate


def _native_close(symbol: str, as_of=None, conn=None) -> float | None:
    """The native-currency close on (or before) a date; None when absent."""
    sql = "SELECT Close FROM market_history WHERE Symbol = ?"
    params: list = [symbol]
    if as_of is not None:
        sql += " AND Date <= ?"
        params.append(str(as_of)[:10])
    sql += " ORDER BY Date DESC LIMIT 1"
    try:
        if conn is not None:
            row = conn.execute(sql, params).fetchone()
        else:
            from quant.data.database import read_only_connection

            with read_only_connection() as c:
                row = c.execute(sql, params).fetchone()
        if not row or row[0] is None:
            return None
        return float(row[0])
    except Exception:  # noqa: BLE001
        return None


def price_in_eur(symbol: str, as_of: Any =None, conn: Any =None) -> float | None:
    """The EUR price of a symbol on (or before) a date (v10.8.0, 2.2).

    Intent: valuation, the daily job, the Overview estimate, buy and sell
    recording, the chart series and the audit all need a price in EUR. The
    native close comes from ``market_history``; the FX multiplier comes from
    :func:`get_fx_to_eur` (the documented nearest-available rate). Returns None
    when no close exists.

    ``as_of`` may be a date, an ISO string, or None (latest). ``conn`` is an
    optional open connection; when omitted a short-lived read-only connection is
    used.
    """
    close = _native_close(symbol, as_of, conn)
    if close is None:
        return None
    return close * get_fx_to_eur(symbol)
