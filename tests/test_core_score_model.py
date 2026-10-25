"""test_core_score_model.py — exact values of the score model (quant/engine/score_model.py).

Every threshold is probed on both sides, every points value is visible in a
score, and the trend formulas are checked against an independent re-derivation
on series where each index and window length changes the result.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from quant.engine import score_model as M

TODAY = date(2024, 6, 30)


# ── helpers ───────────────────────────────────────────────────────────────────

def _rising(n, start=100.0, step=0.1):
    return [start + i * step for i in range(n)]


def _tech_oracle(px):
    sma = sum(px[-200:]) / 200.0
    trend = math.tanh((px[-1] / sma - 1.0) * 5.0)
    if len(px) >= 253:
        mom = px[-22] / px[-253] - 1.0
        return round(50.0 + 25.0 * trend + 25.0 * math.tanh(mom * 2.0), 1)
    return round(50.0 + 50.0 * trend, 1)


# ── _num ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("1.5", 1.5), (2, 2.0), (None, None), ("x", None), (float("nan"), None)])
def test_num(raw, expected):
    assert M._num(raw) == expected


# ── technical ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [199, 200, 252, 253, 254, 300])
def test_technical_score_matches_the_formula(n):
    px = [100.0 + 30.0 * math.sin(i / 17.0) + i * 0.05 for i in range(n)]
    expected = None if n < 200 else _tech_oracle(px)
    assert M.technical_score(px) == expected


def test_technical_score_two_branches_differ_on_a_trend():
    up = _rising(300, 50.0, 0.3)
    assert M.technical_score(up) == _tech_oracle(up)
    assert M.technical_score(up[:252]) == _tech_oracle(up[:252])
    assert M.technical_score(list(reversed(up))) == _tech_oracle(list(reversed(up)))


def test_technical_score_drops_missing_zero_and_negative_closes():
    px = _rising(260)
    dirty = px[:100] + [None, "x", 0.0, -5.0, float("nan")] + px[100:]
    assert M.technical_score(dirty) == _tech_oracle(px)


def test_technical_score_keeps_sub_one_prices():
    px = [0.5 + i * 0.001 for i in range(210)]
    assert M.technical_score(px) == _tech_oracle(px)


def test_technical_score_returns_one_decimal():
    px = [100.0 + 30.0 * math.sin(i / 17.0) for i in range(260)]
    score = M.technical_score(px)
    assert score == round(score, 1) and score != round(score)


# ── fundamentals ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    (None, None), ("x", None), (0.3, 0.3), (10.0, 10.0), (10.5, 0.105), (150.0, 1.5)])
def test_debt_to_equity_normalises_percent(raw, expected):
    got = M._debt_to_equity(raw)
    assert got is None if expected is None else got == pytest.approx(expected)


@pytest.mark.parametrize("ebit, ie, expected", [
    (100.0, -25.0, 4.0), (100.0, 25.0, 4.0), (100.0, 1.0, 100.0),
    (100.0, 0.0, None), (None, 5.0, None), (100.0, None, None)])
def test_interest_cover(ebit, ie, expected):
    assert M.interest_cover(ebit, ie) == expected


def _with(points_max, extra_points, extra_max):
    return round(100.0 * (points_max + extra_points) / (points_max + extra_max), 1)


# Base parts with full points so the probed input's points show in the score.
BASE_PE = {"roe": 0.25, "debt_to_equity": 0.3}                 # 25/25 + 20/20


@pytest.mark.parametrize("pe, points", [
    (-1.0, 0), (0.0, 0), (0.5, 25), (15.0, 25), (15.5, 15), (25.0, 15), (25.5, 5),
    (40.0, 5), (40.5, 0)])
def test_fundamental_pe_points(pe, points):
    assert M.fundamental_score({**BASE_PE, "pe": pe}) == _with(45, points, 25)


@pytest.mark.parametrize("peg, points", [
    (-0.5, 0), (0.0, 0), (0.5, 15), (1.0, 15), (1.5, 8), (2.0, 8), (2.5, 0)])
def test_fundamental_peg_points(peg, points):
    assert M.fundamental_score({**BASE_PE, "peg": peg}) == _with(45, points, 15)


@pytest.mark.parametrize("roe, points", [
    (0.25, 25), (0.20, 25), (0.15, 15), (0.10, 15), (0.05, 5), (0.0, 0), (-0.1, 0)])
def test_fundamental_roe_points(roe, points):
    f = {"pe": 10.0, "debt_to_equity": 0.3, "roe": roe}        # 25/25 + 20/20
    assert M.fundamental_score(f) == _with(45, points, 25)


@pytest.mark.parametrize("de, points", [
    (0.3, 20), (0.5, 20), (50.4, 12), (0.8, 12), (1.0, 12), (1.5, 5), (2.0, 5),
    (2.5, 0), (10.0, 0)])
def test_fundamental_debt_points(de, points):
    f = {"pe": 10.0, "roe": 0.25, "debt_to_equity": de}       # 25/25 + 25/25
    assert M.fundamental_score(f) == _with(50, points, 20)


@pytest.mark.parametrize("icr, points", [
    (10.0, 15), (8.0, 15), (5.0, 8), (3.0, 8), (2.0, 0)])
def test_fundamental_interest_cover_points(icr, points):
    f = {"pe": 10.0, "roe": 0.25, "ebit": icr * 10.0, "interest_expense": -10.0}
    assert M.fundamental_score(f) == _with(50, points, 15)


def test_fundamental_needs_three_inputs_and_a_dict():
    assert M.fundamental_score(None) is None
    assert M.fundamental_score({}) is None
    assert M.fundamental_score({"pe": 10.0, "roe": 0.25}) is None
    assert M.fundamental_score({"pe": 10.0, "roe": 0.25, "peg": 0.5}) == 100.0
    assert M.fundamental_score({"pe": "x", "roe": 0.25, "peg": 0.5}) is None


def test_fundamental_score_is_rounded_to_one_decimal():
    assert M.fundamental_score({"pe": 20.0, "roe": 0.15, "peg": 1.5}) == 58.5


# ── sentiment ─────────────────────────────────────────────────────────────────

def _ago(days):
    return TODAY - timedelta(days=days)


def _sent_oracle(rows):
    tot = wt = 0.0
    for age, s in rows:
        w = 0.5 ** (age / 7.0)
        tot += w * max(-1.0, min(1.0, s))
        wt += w
    return round(50.0 + 50.0 * tot / wt, 1)


def test_sentiment_weights_by_age_and_clips_the_tone():
    rows = [(0, 0.8), (7, -0.4), (14, 3.0), (21, -5.0)]
    ev = [(_ago(a), s) for a, s in rows]
    assert M.sentiment_score(ev, TODAY) == _sent_oracle(rows)


def test_sentiment_needs_three_documents_in_the_window():
    assert M.sentiment_score([(_ago(1), 0.5), (_ago(2), 0.5)], TODAY) is None
    three = [(_ago(1), 0.5), (_ago(2), -0.5), (_ago(3), 0.2)]
    assert M.sentiment_score(three, TODAY) == _sent_oracle([(1, .5), (2, -.5), (3, .2)])
    assert M.sentiment_score(None, TODAY) is None


def test_sentiment_window_edges():
    base = [(_ago(1), 0.1), (_ago(2), 0.1)]
    assert M.sentiment_score(base + [(_ago(30), -1.0)], TODAY) == \
        _sent_oracle([(1, .1), (2, .1), (30, -1.0)])                 # 30 days: in
    assert M.sentiment_score(base + [(_ago(31), -1.0)], TODAY) is None  # 31: out
    assert M.sentiment_score(base + [(_ago(0), -1.0)], TODAY) == \
        _sent_oracle([(1, .1), (2, .1), (0, -1.0)])                  # today: in
    assert M.sentiment_score(base + [(_ago(-1), -1.0)], TODAY) is None  # future: out


def test_sentiment_skips_unusable_rows_and_keeps_reading():
    bad = [(None, 0.5), (_ago(1), None), (_ago(1), "x")]
    good = [(_ago(1), 0.5), (_ago(2), 0.5), (_ago(3), -0.5)]
    assert M.sentiment_score(bad + good, TODAY) == _sent_oracle([(1, .5), (2, .5), (3, -.5)])


def test_sentiment_accepts_iso_strings_and_timestamps():
    ev = [(str(_ago(1)), 0.5), (f"{_ago(2)}T08:30:00", 0.5), (f"{_ago(3)} 08:30", -0.5)]
    assert M.sentiment_score(ev, TODAY) == _sent_oracle([(1, .5), (2, .5), (3, -.5)])


def test_sentiment_total_weight_of_exactly_one():
    rows = [(7, 1.0), (14, -1.0), (14, 0.0)]          # weights 0.5 + 0.25 + 0.25
    assert M.sentiment_score([(_ago(a), s) for a, s in rows], TODAY) == 62.5


# ── composite ─────────────────────────────────────────────────────────────────

def test_applicable():
    full = ("technical", "fundamental", "sentiment")
    assert M.applicable(None) == full
    assert M.applicable("") == full
    assert M.applicable("equity") == full
    assert M.applicable("ETF") == ("technical", "sentiment")
    assert M.applicable("COMMODITY") == ("technical", "sentiment")


def test_composite_all_parts():
    out = M.composite({"technical": 60.0, "fundamental": 80.0, "sentiment": 40.0})
    assert out == {"composite": 64.0, "confidence": "high", "present": 3, "total": 3,
                   "components": {"technical": 60.0, "fundamental": 80.0,
                                  "sentiment": 40.0}}


@pytest.mark.parametrize("parts, cls, value, confidence, present, total", [
    ({"technical": 60.0}, None, 60.0, "medium", 1, 3),                    # 0.50
    ({"fundamental": 80.0, "sentiment": 40.0}, "EQUITY", 68.0, "medium", 2, 3),  # 0.50
    ({"technical": 60.0, "fundamental": 80.0}, None, 68.2, "high", 2, 3),  # 0.85
    ({"technical": 60.0, "sentiment": 40.0}, None, 55.4, "medium", 2, 3),  # 0.65
    ({"sentiment": 40.0}, None, 40.0, "low", 1, 3),                        # 0.15
    ({"technical": 60.0}, "ETF", 60.0, "medium", 1, 2),                    # 0.77
    ({"technical": 60.0, "sentiment": 40.0, "fundamental": 99.0}, "ETF", 55.4, "high", 2, 2),
    ({}, None, None, "low", 0, 3),
])
def test_composite_cases(parts, cls, value, confidence, present, total):
    out = M.composite(parts, cls)
    assert (out["composite"], out["confidence"], out["present"], out["total"]) == \
        (value, confidence, present, total)
    assert out["components"] == {n: parts.get(n) for n in M.applicable(cls)}


def test_composite_rounds_to_one_decimal():
    assert M.composite({"technical": 60.04, "sentiment": 40.0}, "ETF")["composite"] == 55.4


# ── regime ────────────────────────────────────────────────────────────────────

def _regime_oracle(px):
    devs = []
    for k in range(3):
        end = len(px) - k
        devs.append(px[end - 1] / (sum(px[end - 200:end]) / 200.0) - 1.0)
    return devs


def test_trend_regime_needs_203_closes():
    assert M.trend_regime(_rising(202)) is None
    assert M.trend_regime(_rising(200)) is None
    assert M.trend_regime(_rising(203)) is not None


@pytest.mark.parametrize("px, label, confidence", [
    (_rising(260, 100.0, 0.5), "bull", "high"),
    (list(reversed(_rising(260, 100.0, 0.5))), "bear", "high"),
    (_rising(260, 100.0, 0.05), "bull", "medium"),
    ([100.0] * 260, "chop", "low"),
])
def test_trend_regime_labels(px, label, confidence):
    got = M.trend_regime(px)
    assert got[0] == label and got[2] == confidence
    assert got[1] == pytest.approx(_regime_oracle(px)[0])


def test_trend_regime_needs_all_three_days_and_reports_the_latest():
    px = _rising(257, 100.0, 0.5) + [80.0, 300.0, 300.0]   # one day below the band
    assert M.trend_regime(px)[0] == "chop"
    devs = _regime_oracle(px)
    assert devs[0] != devs[1]
    assert M.trend_regime(px)[1] == pytest.approx(devs[0])
    fall = list(reversed(_rising(257, 100.0, 0.5))) + [200.0, 50.0, 50.0]
    assert M.trend_regime(fall)[0] == "chop"


def test_trend_regime_drops_unusable_closes():
    px = _rising(260, 100.0, 0.5)
    dirty = px[:50] + [None, 0.0, -1.0] + px[50:]
    assert M.trend_regime(dirty) == M.trend_regime(px)


def test_sentiment_keeps_reading_after_an_out_of_window_row():
    ev = [(_ago(40), 1.0), (_ago(-2), 1.0), (_ago(1), 0.5), (_ago(2), 0.5), (_ago(3), -0.5)]
    assert M.sentiment_score(ev, TODAY) == _sent_oracle([(1, .5), (2, .5), (3, -.5)])


def test_trend_regime_drops_zero_closes_inside_the_window():
    px = _rising(260, 100.0, 0.5)
    dirty = px[:-30] + [0.0] + px[-30:]
    assert M.trend_regime(dirty) == M.trend_regime(px)


def test_trend_regime_keeps_sub_one_prices():
    px = _rising(210, 0.5, 0.001)
    got = M.trend_regime(px)
    assert got is not None and got[1] == pytest.approx(_regime_oracle(px)[0])
