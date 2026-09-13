"""
test_registry_repair.py — Registry-repair contract (v10.5.2, A6/F1).

Fixture registry with one missing ISIN plus a mocked metadata source; repair
fills it; the blocker clears (resolve_broker returns the ISIN). Curated rows
override metadata and are never overwritten; invalid metadata yields not-found.
"""
from __future__ import annotations

import pandas as pd
import pytest

from quant.data.registry_repair import ensure_registry_rows, repair_isins
from quant.execution.taxonomy import resolve_broker

# A checksum-valid ISIN already present in data/broker_registry.csv (repo vector).
KNOWN_ISIN = "IE00B4L5Y983"


def _fixture(tmp_path):
    reg = tmp_path / "broker_registry.csv"
    pd.DataFrame([
        {"yahoo_ticker": "AMZN", "isin": "US0231351067", "tr_ticker": "AMZN",
         "exchange": "LS Exchange", "currency": "USD", "instrument_class": "EQUITY"},
        {"yahoo_ticker": "5J50.DE", "isin": "", "tr_ticker": "5J50.DE",
         "exchange": "LS Exchange", "currency": "EUR", "instrument_class": "ETF"},
    ]).to_csv(reg, index=False)
    cur = tmp_path / "isin_curated.csv"
    cur.write_text("symbol,isin,verified_by,verified_at\n", encoding="utf-8")
    return str(reg), str(cur)


def test_repair_fills_missing_isin_from_metadata_and_clears_blocker(tmp_path):
    reg, cur = _fixture(tmp_path)
    summary = repair_isins(
        registry_path=reg, curated_path=cur, metadata_source=lambda _s: KNOWN_ISIN,
    )
    assert summary["filled"] == 1
    assert summary["not_found"] == 0

    out = pd.read_csv(reg)
    fixed = out[out["yahoo_ticker"] == "5J50.DE"].iloc[0]
    assert fixed["isin"] == KNOWN_ISIN
    assert fixed["isin_source"] == "yahoo"
    # Pre-existing cell keeps user provenance and is never overwritten.
    amzn = out[out["yahoo_ticker"] == "AMZN"].iloc[0]
    assert amzn["isin"] == "US0231351067"
    assert amzn["isin_source"] == "user"
    # Blocker clears: the routing resolver now returns the ISIN.
    assert resolve_broker("5J50.DE", path=reg).get("isin") == KNOWN_ISIN


def test_repair_is_idempotent(tmp_path):
    reg, cur = _fixture(tmp_path)
    repair_isins(registry_path=reg, curated_path=cur, metadata_source=lambda _s: KNOWN_ISIN)
    second = repair_isins(registry_path=reg, curated_path=cur,
                          metadata_source=lambda _s: KNOWN_ISIN)
    assert second["filled"] == 0
    assert second["not_found"] == 0


def test_curated_row_overrides_metadata(tmp_path):
    reg, cur = _fixture(tmp_path)
    with open(cur, "a", encoding="utf-8") as f:
        f.write(f"5J50.DE,{KNOWN_ISIN},broker app,2026-09-13\n")
    summary = repair_isins(registry_path=reg, curated_path=cur, metadata_source=lambda _s: None)
    assert summary["filled"] == 1
    row = pd.read_csv(reg)
    row = row[row["yahoo_ticker"] == "5J50.DE"].iloc[0]
    assert row["isin_source"] == "curated"


def test_invalid_metadata_is_never_written(tmp_path):
    reg, cur = _fixture(tmp_path)
    summary = repair_isins(registry_path=reg, curated_path=cur,
                           metadata_source=lambda _s: "NOT-AN-ISIN")
    assert summary["filled"] == 0
    assert summary["not_found"] == 1
    row = pd.read_csv(reg)
    row = row[row["yahoo_ticker"] == "5J50.DE"].iloc[0]
    assert pd.isna(row["isin"]) or str(row["isin"]).strip() == ""


def test_invalid_curated_row_aborts_with_plain_error(tmp_path):
    reg, cur = _fixture(tmp_path)
    with open(cur, "a", encoding="utf-8") as f:
        f.write("5J50.DE,BADISIN,Broker app,2026-09-13\n")
    with pytest.raises(ValueError):
        repair_isins(registry_path=reg, curated_path=cur, metadata_source=lambda _s: None)


def test_ensure_registry_rows_then_curated_isin(tmp_path):
    """H3.1: a held, unroutable symbol gains a row; the curated ISIN then lands."""
    reg = tmp_path / "broker_registry.csv"
    pd.DataFrame([{
        "yahoo_ticker": "AMZN", "isin": "US0231351067", "tr_ticker": "AMZN",
        "exchange": "LS Exchange", "currency": "USD", "instrument_class": "EQUITY",
    }]).to_csv(reg, index=False)
    cur = tmp_path / "isin_curated.csv"
    cur.write_text("symbol,isin,verified_by,verified_at\n"
                   "5J50.DE,IE000U9ODG19,user,2026-09-13\n", encoding="utf-8")

    added = ensure_registry_rows(str(reg), symbols=["5J50.DE", "AMZN"])
    assert added["added"] == 1  # AMZN already present

    summary = repair_isins(registry_path=str(reg), curated_path=str(cur),
                           metadata_source=lambda _s: None)
    assert summary["filled"] == 1 and summary["curated"] == 1

    out = pd.read_csv(reg)
    row = out[out["yahoo_ticker"] == "5J50.DE"].iloc[0]
    assert row["isin"] == "IE000U9ODG19"
    assert resolve_broker("5J50.DE", path=str(reg)).get("isin") == "IE000U9ODG19"

    assert ensure_registry_rows(str(reg), symbols=["5J50.DE"])["added"] == 0
