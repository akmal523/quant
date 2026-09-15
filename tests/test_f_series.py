"""
test_f_series.py — fresh-install hotfix (F-series).

Five bugs on a clean install:
  F1 registry rows: -1 on an empty DB (COUNT threw on a missing table).
  F2 the DB path resolved into site-packages for an installed wheel.
  F3 doctor raised a CatalogException (no schema) instead of initializing.
  F4 Explore cards showed the bare ticker (no empty-state handling).
  F5 search failed (themes.csv was not shipped as package data).

Fixes: a dev/production path split (user_data_dir("quant-ai") when installed,
plus a QUANT_DATA_DIR override), first-run seeding of the writable data dir, DB
init in doctor, an Explore empty state, and bundled themes as package data.
"""
from __future__ import annotations

import duckdb

from quant import paths
from quant.data import database
from quant.data.bootstrap import seed_user_data
from quant.ui.cards import explore_card_fields
from quant.ui.search import load_themes, resolve_theme_symbols

# ── F2: dev vs production resolver ───────────────────────────────────────────

def test_development_checkout_is_detected():
    # The test runs from the source checkout (pyproject.toml + .git present).
    assert paths.IS_DEVELOPMENT is True
    assert paths.PROJECT_ROOT == paths.PACKAGE_DIR.parent


def test_is_development_false_for_bare_dir(tmp_path):
    assert paths._is_development(tmp_path) is False


def test_user_data_dir_is_named_quant_ai():
    assert paths._user_data_dir().name == "quant-ai"


def test_quant_data_dir_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("QUANT_DATA_DIR", str(tmp_path))
    assert paths._resolve_root() == tmp_path


def test_package_data_dir_is_inside_the_package():
    assert paths.BUNDLED_DATA_DIR == paths.PACKAGE_DIR / "_data"


# ── F5: bundled themes ship as package data ──────────────────────────────────

def test_bundled_themes_exist():
    assert (paths.BUNDLED_DATA_DIR / "themes.csv").exists()


def test_bundled_themes_resolve_space():
    themes = load_themes(str(paths.BUNDLED_DATA_DIR / "themes.csv"))
    resolved = resolve_theme_symbols(themes)
    assert "space" in resolved
    assert "5J50.DE" in resolved["space"]        # space -> aerospace -> 5J50.DE
    assert "DFEN" in resolved["space"]           # space -> defence -> DFEN


# ── F1/F3: seeding + DB init ─────────────────────────────────────────────────

def test_bootstrap_seeds_missing_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(paths, "BUNDLED_DATA_DIR", paths.PACKAGE_DIR / "_data")
    created = seed_user_data()
    assert "portfolio.csv" in created and "account.yaml" in created
    assert "themes.csv" in created
    assert (tmp_path / "data" / "portfolio.csv").read_text(
        encoding="utf-8").startswith("Symbol,Avg_Entry_Price")
    assert "space" in (tmp_path / "data" / "themes.csv").read_text(encoding="utf-8")


def test_bootstrap_never_overwrites(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    mine = data / "portfolio.csv"
    mine.write_text("Symbol,Avg_Entry_Price,Current_Value_EUR,Broker_PnL_EUR\nAMZN,1,2,0\n",
                    encoding="utf-8")
    monkeypatch.setattr(paths, "DATA_DIR", data)
    seed_user_data()
    assert "AMZN" in mine.read_text(encoding="utf-8")   # user row preserved


def test_init_db_creates_schema_on_empty_db(tmp_path, monkeypatch):
    # A brand-new DB: init_db must create the schema so registry counts read 0,
    # not -1 (the missing-table CatalogException the doctor hit).
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(paths, "BUNDLED_DATA_DIR", paths.PACKAGE_DIR / "_data")
    conn = duckdb.connect(str(tmp_path / "fresh.duckdb"))
    prev = getattr(database._local, "conn", None)
    database.use_connection(conn)
    try:
        database.init_db()
        assert conn.execute("SELECT COUNT(*) FROM asset_registry").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM market_history").fetchone()[0] == 0
    finally:
        conn.close()
        database._local.conn = prev
        if prev is not None:  # restore the session connection binding
            database.use_connection(prev)


# ── F4: Explore empty state ──────────────────────────────────────────────────

def test_card_has_details_false_for_unknown_symbol():
    f = explore_card_fields("ZZZZ.NO")
    assert f["has_details"] is False
    assert f["name"] == "ZZZZ.NO"        # fallback still available internally


def test_card_has_details_true_when_registry_row_present():
    conn = database.get_connection()
    conn.execute("DELETE FROM asset_registry WHERE symbol = 'ZZTEST'")
    conn.execute("INSERT INTO asset_registry (symbol, instrument_class, name, "
                 "universe_status) VALUES ('ZZTEST', 'EQUITY', 'Zed Test', 'ACTIVE')")
    try:
        f = explore_card_fields("ZZTEST")
        assert f["has_details"] is True
        assert f["name"] == "Zed Test"
    finally:
        conn.execute("DELETE FROM asset_registry WHERE symbol = 'ZZTEST'")
