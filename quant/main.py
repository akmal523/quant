"""
main.py — Orchestration engine v9.0.
Architecture:
  1. Load & sort market data from DuckDB
  2. Filter survivors (portfolio + ETFs + fundamentals screener)
  3. Async fetch SEC 8-K / news texts
  4. Score ALL texts in main process with single FinBERT instance
  5. Multiprocess technical + fundamental analysis (no FinBERT in workers)
  6. Data-confidence penalty prevents false BUY from missing NLP data
  7. Batch-write new NLP cache entries
  8. Report: top buys + stocks (non-ETF), full scan, portfolio audit
"""
from __future__ import annotations

import hashlib
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import polars as pl

from quant import paths
from quant.analytics.scoring import (
    allocate_capital_regime,
    apply_fast_filter,
    etf_tactical_grade,
    evaluate_structural_grade,
    evaluate_tactical_grade,
    fit_market_regime,
    stewardship_score_v2,
)
from quant.analytics.scoring import (
    regime_confidence as regime_confidence_of,
)
from quant.cli.output import reporter
from quant.config import WEIGHT_TECHNICAL
from quant.data.currency import get_eur_rate
from quant.data.database import get_connection, init_db
from quant.data.fundamentals import get_fundamentals
from quant.data.universe import is_etf
from quant.execution.taxonomy import get_instrument_class
from quant.features.indicators import add_all_indicators, fast_volatility
from quant.portfolio.portfolio import (
    account_effectiveness,
    enhanced_portfolio_audit,
    load_portfolio,
    print_effectiveness_report,
)
from quant.portfolio.risk import calculate_risk_penalty
from quant.reporting.notifier import notify_daily

# ── Logging ───────────────────────────────────────────────────────────────────
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("yfinance").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ── Worker Process ────────────────────────────────────────────────────────────

# main.py -> process_asset()
def process_asset(symbol: str, f_data: dict, sector: str, nlp_data: dict,
                  market_regime_prob: float, etf_quality_map: dict | None = None) -> dict | None:
    """Score one asset in a worker process. Returns its row, or None on failure."""
    try:
        # FIX: Worker must now fetch its own DataFrame from the database
        from quant.data.database import get_connection
        conn = get_connection()
        df = conn.execute(
            "SELECT * FROM market_history WHERE Symbol = ? ORDER BY Date ASC",
            [symbol]
        ).df()

        if df.empty:
            return None

        price_hist = df.drop(columns=["Symbol", "Sector"], errors="ignore")
        # Get sector from the queried data
        sector = df['Sector'].iloc[0] if 'Sector' in df.columns else "Unknown"

        last_close = price_hist["Close"].iloc[-1] if "Close" in price_hist.columns else None
        if last_close is None or (
                isinstance(last_close, float)
                and (pd.isna(last_close) or np.isnan(last_close))):
            logger.warning("[SKIP] %s: No valid price data (NaN close)", symbol)
            return {
                "Symbol": symbol,
                "Is_ETF": is_etf(symbol),
                "Current_Price": 0.0,
                "Structural_Grade": 0.0,
                "Tactical_Grade": 0.0,
                "Stewardship": 18.0,
                "Horizon": "N/A",
                "Signal": "N/A",
                "Active_Score": 0.0,
                "NLP_Reasoning": nlp_data.get("reasoning", "No price data available"),
            }

        hist_ind = add_all_indicators(price_hist)
        # Pillar 2b: use precomputed MARKET regime prob (fit once on index),
        # not per-asset HMM. Saves ~1s/asset and is statistically sound.
        hmm_prob_bull = market_regime_prob

        with np.errstate(invalid="ignore", divide="ignore"):
            returns = np.log(hist_ind["Close"] / hist_ind["Close"].shift(1)).dropna()
        var_penalty = calculate_risk_penalty(returns)

        # Phase 4 (3.1): bifurcated scoring by instrument_class.
        # Equities run the full pipeline (Fundamentals, NLP, GARCH, Technicals).
        # ETFs bypass Fundamentals/NLP — scored on macro regime + trend + rel strength.
        # Commodities scored on macro regime + trend (inflation/USD proxies).
        asset_class = get_instrument_class(symbol)
        asset_is_etf = asset_class in ("ETF", "CASH")
        asset_is_commodity = asset_class == "COMMODITY"

        if asset_is_etf or asset_is_commodity:
            # Bifurcated path: no fundamentals, no NLP. Score on macro regime
            # (HMM bull prob) + trend (price vs 200 SMA) + relative strength.
            s_val = 18.0
            close = hist_ind["Close"]
            sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else close.mean()
            if asset_is_etf and etf_quality_map and symbol in etf_quality_map:
                # Phase 5 (v10.2): cross-sectional ETF structural grade (0-100),
                # computed in the main process over the ETF subset.
                struct_grade = etf_quality_map[symbol]
                # Continuous tactical grade: regime tilt + momentum z.
                momentum_z = etf_quality_map.get(f"{symbol}_momentum_z", 0.0)
                tact_grade = etf_tactical_grade(hmm_prob_bull, momentum_z)
            else:
                struct_grade = 85.0
                trend_score = 50.0 + 50.0 * (1.0 if close.iloc[-1] > sma200 else -1.0)
                # Macro regime dominates tactical for ETFs/commodities.
                tact_grade = max(0.0, min(100.0, (hmm_prob_bull * 60.0) + (trend_score * 0.4)))
        else:
            s_val = stewardship_score_v2(f_data, sector)
            struct_grade = evaluate_structural_grade(
                pe=f_data.get("PE"), peg=f_data.get("PEG"),
                roe=f_data.get("ROE"), stewardship_val=s_val,
            )
            # data_confidence: 1.0 = real SEC/News data scored, 0.0 = no-data fallback
            data_confidence = nlp_data.get("data_confidence", 1.0)
            tact_grade = evaluate_tactical_grade(
                hmm_prob_bull=hmm_prob_bull,
                finbert_score=nlp_data.get("score", 0.0),
                var_penalty=var_penalty,
                data_confidence=data_confidence,
            )

        allocation = allocate_capital_regime(struct_grade, tact_grade, s_val)

        return {
            "Symbol": symbol,
            "Is_ETF": asset_is_etf,
            "Instrument_Class": asset_class,
            "Current_Price": round(hist_ind["Close"].iloc[-1], 2),
            "Structural_Grade": round(struct_grade, 1),
            "Tactical_Grade": round(tact_grade, 1),
            "Stewardship": s_val,
            "Horizon": allocation["Horizon"],
            "Signal": allocation["Signal"],
            "Active_Score": allocation["Active_Score"],
            "NLP_Reasoning": nlp_data.get("reasoning", "N/A"),
            "doc_hash": nlp_data.get("doc_hash"),
            "nlp_score": nlp_data.get("score"),
        }

    except Exception as e:
        logger.exception("[WORKER CRASH] %s: %s", symbol, str(e))
        return None



# ── Main Orchestration ─────────────────────────────────────────────────────────

def main() -> None:
    """Run the full review: fetch, score, audit, and print the briefing."""
    from quant.infra.observability import ObservabilityCollector
    obs = ObservabilityCollector()

    logger.info("Loading localized database...")
    with obs.step("load_database"):
        conn = get_connection()
        init_db()

    try:
        # Pillar 4: read DuckDB -> Polars natively (Rust, multi-core, no GIL).
        # .pl() avoids pandas intermediate, so no pyarrow dependency needed.
        _q0 = time.perf_counter()
        pl_df = conn.execute("SELECT * FROM market_history ORDER BY Symbol ASC, Date ASC").pl()
        # v10.4.0 (Phase 4): structured telemetry for DuckDB query time.
        obs.record_metric("duckdb_query_ms", (time.perf_counter() - _q0) * 1000.0)
    except Exception:
        logger.error("market_history missing. Run data_updater.py.")
        return

    if "Date" in pl_df.columns:
        pl_df = pl_df.with_columns(pl.col("Date").str.to_datetime())
    # Vectorized log-returns across ALL symbols simultaneously (Rust).
    pl_df = pl_df.sort(["Symbol", "Date"]).with_columns(
        (pl.col("Close").log().diff() * 100).alias("LogReturns").over("Symbol")
    )
    # group_by yields tuple keys ('005930.KS',) — unpack to scalar symbol string.
    grouped_data = {sym[0]: df.to_pandas() for sym, df in pl_df.group_by("Symbol")}

    port_df = load_portfolio(paths.DATA_PORTFOLIO)
    portfolio_symbols = set(port_df["Symbol"].unique()) if not port_df.empty else set()
    logger.info("Loaded Portfolio: %s", list(portfolio_symbols))

    # v10.7.2 (Part 5.2): seed holdings_meta from the broker CSV so the valuation
    # pipeline has shares. First-time only: an existing row keeps its sync_date,
    # so the 35-day reminder still reflects the user's last CSV export.
    try:
        from quant.engine import valuation

        def _latest_close(sym: str) -> float | None:
            from quant.data.currency import price_in_eur

            return price_in_eur(sym, conn=conn)

        _synced = valuation.sync_holdings_meta(
            conn, port_df, _latest_close, first_time_only=True)
        if _synced:
            logger.info("holdings_meta: synced %d new symbol(s)", _synced)
    except Exception as e:  # noqa: BLE001
        logger.warning("holdings_meta sync failed: %s", e)

    # ── Plan 3 (Phase 1): Smart Funnel universe filter ──────────────────────
    # The broad 1000+ universe is filtered to the top survivors ONCE per cycle by
    # data_updater.py and cached in the funnel_survivors table. main.py reads that
    # cache (it does NOT re-run the funnel / re-fetch the universe). Scan universe
    # = cached survivors + ACTIVE registry + CORE ETFs + portfolio holdings.
    active_symbols = set(portfolio_symbols)
    try:
        rows = conn.execute(
            "SELECT symbol FROM asset_registry "
            "WHERE universe_status = 'ACTIVE' AND universe_status != 'DELISTED'"
        ).fetchall()
        active_symbols.update(r[0] for r in rows)
    except Exception:
        pass
    # CORE ETFs are always tracked even if not in registry.
    from quant.data.universe_builder import BROAD_ETFS
    active_symbols.update(BROAD_ETFS.values())
    # Funnel survivors (top ~24): read the cache written by data_updater.py.
    # This keeps the documented 2-step flow (data_updater -> main) and avoids
    # re-fetching the 1000+ symbol universe / re-tripping Yahoo rate limits.
    cached_survivors: list = []
    try:
        from quant.data.funnel import load_survivors
        cached_survivors = load_survivors()
        if cached_survivors:
            active_symbols.update(cached_survivors)
            logger.info("Funnel survivors loaded from cache: %d", len(cached_survivors))
        else:
            logger.warning("No cached funnel survivors - run data_updater.py first. "
                           "Scanning CORE + ACTIVE + portfolio only.")
    except Exception as e:
        logger.warning("Funnel survivor cache read failed: %s", e)
    # v10.7.3 (Part 2.3): funnel survivors not yet in the registry get their names
    # probed and stored, so candidate cards never render "MU (MU)".
    try:
        from quant.data.names import probe_and_store

        for _sym in cached_survivors:
            probe_and_store(_sym, conn)
    except Exception as e:  # noqa: BLE001
        logger.warning("Funnel survivor name probe failed: %s", e)
    grouped_data = {s: df for s, df in grouped_data.items() if s in active_symbols}
    logger.info("Scan universe (funnel + CORE + ACTIVE + portfolio): %d symbols",
                len(grouped_data))

    # Fail fast: an empty scan universe means market_history has no rows for the
    # active symbols (usually a failed/aborted data_updater run). Without this
    # guard the regime HMM fit below calls max() on an empty dict and crashes.
    if not grouped_data:
        logger.error("No market data for the scan universe. "
                     "Run data_updater.py to populate market_history.")
        return

    # ── Pillar 1: Smart Funnel ──────────────────────────────────────────────
    # Tier 1 (microseconds): fast fundamental filter.
    # Tier 2 (milliseconds): fast technical filter (uptrend check).
    # Tier 3 (seconds): heavy NLP/SEC only on top ~30 survivors.
    survivors: dict[str, pd.DataFrame] = {}
    survivor_funds: dict[str, dict] = {}
    tier1_kept = 0
    tier2_kept = 0
    for sym, df in grouped_data.items():
        f_data = get_fundamentals(sym)
        is_portfolio = sym in portfolio_symbols
        is_asset_etf = is_etf(sym)
        if is_portfolio or is_asset_etf or apply_fast_filter(f_data):
            tier1_kept += 1
            # Tier 2: uptrend check — Price > 200 SMA (fast, vectorized).
            if not is_portfolio and not is_asset_etf:
                close = df["Close"]
                if len(close) < 200 or close.iloc[-1] <= close.rolling(200).mean().iloc[-1]:
                    continue
            tier2_kept += 1
            survivors[sym] = df
            survivor_funds[sym] = f_data
    logger.info("Funnel: Tier1 kept %d, Tier2 kept %d (of %d in)",
                tier1_kept, tier2_kept, len(grouped_data))
    logger.info("Survivors after Tier1+Tier2 funnel: %d", len(survivors))

    # ── Pillar 2b: Fit market regime HMM ONCE on a broad index ──────────────
    # Use SPY if present in universe, else the longest-history asset as proxy.
    # v10.5.3 (spec 1.2): write the regime block UNCONDITIONALLY with an explicit
    # state (estimated | insufficient_history | failed) so the UI never guesses.
    market_regime_prob = 0.5
    regime_state = "estimated"
    regime_error_msg = None
    regime_sym = ("SPY" if "SPY" in grouped_data
                  else max(grouped_data, key=lambda s: len(grouped_data[s])))
    try:
        regime_df = grouped_data[regime_sym]
        if len(regime_df["Close"]) < 252:
            regime_state = "insufficient_history"
            market_regime_prob = 0.5
        else:
            regime_vol = fast_volatility(regime_df["Close"])
            regime_raw = fit_market_regime(regime_df["Close"], regime_vol)
            market_regime_prob = regime_raw / float(WEIGHT_TECHNICAL)
            logger.info("Market regime (fit on %s): bull prob=%.2f", regime_sym, market_regime_prob)
    except Exception as e:  # noqa: BLE001
        regime_state = "failed"
        regime_error_msg = str(e)
        market_regime_prob = 0.5
        logger.warning("Market regime fit failed (surfaced in Health): %s", e)

    # ── Phase 5 (v10.2): cross-sectional ETF quality map ────────────────────
    # Compute the ETF structural grade + momentum z in the MAIN process over the
    # ETF subset (workers lack the full subset). Passed to process_asset.
    etf_quality_map: dict[str, float] = {}
    try:
        from quant.analytics.scoring import etf_quality_score
        from quant.features.build_features import latest_features
        etf_syms = [s for s in survivors if is_etf(s)]
        if etf_syms:
            feat = latest_features()
            etf_feat = feat.filter(pl.col("Symbol").is_in(etf_syms)).to_pandas()
            if not etf_feat.empty:
                # Benchmark-relative 6m return vs IWDA.AS (MSCI World).
                bench = feat.filter(pl.col("Symbol") == "IWDA.AS").to_pandas()
                bench_ret = bench["ret_6m"].iloc[-1] if not bench.empty else 0.0
                etf_feat["rel_strength_6m"] = etf_feat["ret_6m"] - bench_ret
                scored = etf_quality_score(etf_feat)
                for _, r in scored.iterrows():
                    etf_quality_map[r["Symbol"]] = float(r["etf_quality"])
                    etf_quality_map[f"{r['Symbol']}_momentum_z"] = float(r["momentum_z"])
                logger.info("ETF quality map: %d ETFs scored cross-sectionally", len(scored))
    except Exception as e:
        logger.warning("ETF quality map failed (fallback to 85.0): %s", e)

    # ── Step 2: Async text fetch (Tier 3 — only on funnel survivors) ────────
    # v10.8.0 (1.3): the optional text-fetch step must degrade visibly, never
    # crash the review. If the fetcher (or its parser dependency) is missing,
    # the review continues with no news text and Health names the missing input.
    import asyncio
    survivor_texts: dict[str, str] = {}
    try:
        from quant.data.async_fetcher import fetch_all_texts_concurrently
        survivor_texts = asyncio.run(
            fetch_all_texts_concurrently(list(survivors.keys())))
    except Exception as e:  # noqa: BLE001
        logger.warning("Text fetch unavailable (news/filings input missing): %s", e)
        try:
            from quant.data.news import record_fetch_result
            record_fetch_result(False)
        except Exception:  # noqa: BLE001
            pass

    # v10.5.3 (R7): route portfolio holdings through the shared news cache so the
    # review and Explore can never disagree for the same symbol on the same day.
    try:
        from quant.data.news import load_news, record_fetch_result

        for _sym in (port_df["Symbol"] if not port_df.empty else []):
            if not survivor_texts.get(_sym):
                _items = load_news(_sym)
                if _items:
                    survivor_texts[_sym] = " | ".join(i["headline"] for i in _items[:5])
        record_fetch_result(any(bool(t) for t in survivor_texts.values()))
    except Exception as e:  # noqa: BLE001
        logger.warning("Holdings news merge failed: %s", e)

    # ── Step 3: NLP scoring in MAIN process (single FinBERT, DI) ────────────
    # v10.8.2: the news stack (torch/transformers) is an optional extra. When it
    # is absent, every symbol gets a neutral, low-confidence score and the rest
    # of the pipeline runs unchanged.
    try:
        from quant.analytics.sentiment import NLPScorer

        scorer = NLPScorer()
    except Exception as e:  # noqa: BLE001
        logger.warning("News sentiment unavailable (%s); using neutral scores.", e)
        scorer = None

    nlp_cache_rows: list[tuple] = []
    nlp_data_map: dict[str, dict] = {}

    for sym, text in survivor_texts.items():
        if not text:
            # No data found — neutral score with ZERO confidence penalty
            nlp_data_map[sym] = {
                "score": 0.0,
                "reasoning": "No SEC/News data available — low confidence neutral score",
                "doc_hash": None,
                "data_confidence": 0.0,
            }
            logger.debug("[NLP] %s: no text data, data_confidence=0.0", sym)
            continue

        h = hashlib.sha256(text.encode('utf-8')).hexdigest()
        row = conn.execute("SELECT score FROM nlp_scores WHERE doc_hash = ?", [h]).fetchone()
        if row:
            nlp_data_map[sym] = {
                "score": row[0],
                "reasoning": "Cache Hit",
                "doc_hash": h,
                "data_confidence": 1.0,
            }
            logger.debug("[NLP] %s: cache hit (score=%.1f)", sym, row[0])
        elif scorer is None:
            nlp_data_map[sym] = {
                "score": 0.0,
                "reasoning": "News sentiment is off (install the 'news' extra).",
                "doc_hash": None,
                "data_confidence": 0.0,
            }
        else:
            nlp_result = scorer.score_document(text)
            nlp_data_map[sym] = {
                "score": nlp_result["score"],
                "reasoning": nlp_result["reasoning"],
                "doc_hash": nlp_result["doc_hash"],
                "data_confidence": 1.0,
            }
            if nlp_result["doc_hash"]:
                nlp_cache_rows.append((nlp_result["doc_hash"], nlp_result["score"]))
            logger.debug("[NLP] %s: scored in main (score=%.1f)", sym, nlp_result["score"])

    # FIX: Free the 500MB model from RAM BEFORE forking the workers!
    del scorer
    import gc
    gc.collect()

    # FIX: Use 'spawn' to avoid inheriting PyTorch state into child processes
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass


    # ── Step 4: Multiprocessing (no FinBERT in workers) ──────────────────────
    results = []
    cpu_cores = min(4, max(1, multiprocessing.cpu_count() - 1))
    logger.info("Processing %d survivors with %d workers...", len(survivors), cpu_cores)

    with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
        futures = {
            executor.submit(
                process_asset,
                sym,
                # df is REMOVED from here
                survivor_funds[sym],
                df['Sector'].iloc[0] if 'Sector' in df.columns else "Other",
                nlp_data_map.get(sym, {"score": 0.0, "reasoning": "No NLP data"}),
                market_regime_prob,
                etf_quality_map,
            ): sym for sym, df in survivors.items()
        }

        for future in as_completed(futures):
            res = future.result()
            if res:
                doc_hash = res.pop("doc_hash", None)
                nlp_score_val = res.pop("nlp_score", None)
                if doc_hash and nlp_score_val is not None:
                    nlp_cache_rows.append((doc_hash, nlp_score_val))
                results.append(res)

    if nlp_cache_rows:
        conn.execute("BEGIN TRANSACTION")
        for h, s in nlp_cache_rows:
            conn.execute(
                "INSERT OR REPLACE INTO nlp_scores (doc_hash, score) VALUES (?, ?)",
                [h, s])
        conn.execute("COMMIT")
        logger.info("NLP cache: %d new entries saved", len(nlp_cache_rows))

    # Ensure all portfolio symbols appear in results, even if missing from DB
    scanned_symbols = {r["Symbol"] for r in results}
    for sym in portfolio_symbols:
        if sym not in scanned_symbols:
            logger.warning("[PORTFOLIO] %s: No market data found — adding placeholder", sym)
            results.append({
                "Symbol": sym,
                "Is_ETF": is_etf(sym),
                "Current_Price": 0.0,
                "Structural_Grade": 0.0,
                "Tactical_Grade": 0.0,
                "Stewardship": 18.0,
                "Horizon": "N/A",
                "Signal": "N/A",
                "Active_Score": 0.0,
                "NLP_Reasoning": "No market data available for this symbol",
            })

    if not results:
        logger.warning("No assets passed the filters or completed scoring.")
        return

    # ── Step 5: Reporting ────────────────────────────────────────────────────
    final_df = pd.DataFrame(results)
    final_df.to_csv(str(paths.OUTPUTS_DIR / "market_scan_v8.csv"), index=False)

    # ── Phase 5 (v10.2): write ETF factor scores into the run artifact ──────
    # The dashboard Z-score section reads factor_scores.parquet. Populate it
    # for ETFs too (trend/RS/low-vol/momentum), not just equities.
    try:
        from quant.reporting.artifacts import new_run_dir, save_artifact
        run_dir = new_run_dir()
        if etf_quality_map:
            etf_syms = [s for s in survivors if is_etf(s)]
            feat = latest_features()
            etf_feat = feat.filter(pl.col("Symbol").is_in(etf_syms)).to_pandas()
            if not etf_feat.empty:
                bench = feat.filter(pl.col("Symbol") == "IWDA.AS").to_pandas()
                bench_ret = bench["ret_6m"].iloc[-1] if not bench.empty else 0.0
                etf_feat["rel_strength_6m"] = etf_feat["ret_6m"] - bench_ret
                from quant.analytics.scoring import etf_factor_scores
                scored = etf_factor_scores(etf_feat)
                save_artifact(run_dir, "factor_scores", scored)
                logger.info("ETF factor scores written to %s/factor_scores.parquet", run_dir)
    except Exception as e:
        logger.warning("ETF factor scores artifact failed: %s", e)

    # ── v10.5.0: terse default output (spec 3.3, max 20 lines) ──────────────
    import datetime as _dt

    from quant import __version__
    from quant.portfolio.account import load_account
    from quant.reporting.actions import build_actions, format_action_line
    from quant.reporting.artifacts import new_run_dir

    run_dir = new_run_dir()
    today = _dt.date.today().isoformat()
    account = load_account()

    # Portfolio audit is the single source of actions (T3).
    audit_res = pd.DataFrame()
    if not port_df.empty:
        market_data = {s: df for s, df in grouped_data.items() if s in set(port_df["Symbol"])}
        audit_res = enhanced_portfolio_audit(
            port_df, final_df, current_date=today, market_data=market_data,
        )
        # A3 (v10.5.2): persist the canonical status with the audit so the
        # holdings table and the action cards read one object.
        status_map = {a["symbol"]: a["status"] for a in build_actions(audit_res)}
        audit_res["Status"] = audit_res["Symbol"].astype(str).map(status_map)
        audit_res.to_csv(str(paths.OUTPUTS_DIR / "portfolio_audit.csv"), index=False)

    actions = build_actions(audit_res)

    # v10.5.3 (R5): attach the friendly display name for tables and search.
    try:
        _names = conn.execute(
            "SELECT symbol, COALESCE(display_name, name, symbol) AS nm FROM asset_registry"
        ).df()
        _nmap = dict(zip(_names["symbol"], _names["nm"]))
        if not audit_res.empty:
            audit_res["Name"] = audit_res["Symbol"].astype(str).map(lambda x: _nmap.get(x, x))
            audit_res.to_csv(str(paths.OUTPUTS_DIR / "portfolio_audit.csv"), index=False)
    except Exception as e:  # noqa: BLE001
        logger.warning("Name column failed: %s", e)

    # v10.5.3 (spec 1.1): persist scores as a run artifact the UI reads via
    # artifacts.read_scores (single accessor; no parquet paths in the UI).
    try:
        from quant.reporting.artifacts import save_artifact
        score_cols = [c for c in ("Symbol", "Structural_Grade", "Tactical_Grade",
                                  "Active_Score") if c in final_df.columns]
        if not final_df.empty and "Symbol" in score_cols:
            save_artifact(run_dir, "scores", final_df[score_cols])
    except Exception as e:  # noqa: BLE001
        logger.warning("Scores artifact failed: %s", e)
    blocked = [a for a in actions if a["blocked"]]

    if regime_state == "estimated":
        regime_label = ("rising" if market_regime_prob >= 0.6
                        else "falling" if market_regime_prob <= 0.4 else "mixed")
        regime_confidence = regime_confidence_of(market_regime_prob)
    else:
        regime_label = "unavailable"
        regime_confidence = None
    regime_block = {
        "state": regime_state,
        "label": regime_label if regime_state == "estimated" else None,
        "prob": round(market_regime_prob, 4) if regime_state == "estimated" else None,
        "confidence": regime_confidence,
        "as_of": today,
        "error": regime_error_msg,
    }
    reporter.line(f"quant run {__version__}")
    reporter.line(f"  regime {regime_label}, p={market_regime_prob:.2f} "
                  f"(fit {regime_sym}, as-of {today})")
    reporter.line(f"  scanned {len(final_df)} symbols; {len(actions)} actions, "
                  f"{len(blocked)} blocked")
    if actions:
        reporter.line("  actions")
        for a in actions:
            reporter.line(format_action_line(a))
    else:
        reporter.line("  No actions required today.")

    total_value = port_df["Amount_EUR"].sum() if not port_df.empty else 0.0
    pnl_eur = float(audit_res["Real_PnL_EUR"].sum()) if (
        not audit_res.empty and "Real_PnL_EUR" in audit_res) else 0.0
    invested = float(audit_res["Invested_EUR"].sum()) if (
        not audit_res.empty and "Invested_EUR" in audit_res) else 0.0
    pnl_pct = (pnl_eur / invested * 100) if invested > 0 else 0.0
    cash_str = f"{account.cash_eur:.2f} EUR" if account.cash_is_set else "not set"
    reporter.line(f"  portfolio {total_value:.2f} EUR; PnL {pnl_eur:+.2f} EUR "
                  f"({pnl_pct:+.2f}%); cash {cash_str}; risk {account.risk_profile}")

    with_news = sum(
        1 for s in survivors if nlp_data_map.get(s, {}).get("data_confidence", 0.0) > 0
    )
    without_news = len(survivors) - with_news
    reporter.line(f"  evidence: {with_news} symbols with news, {without_news} without "
                  f"(sentiment neutral, confidence low)")

    # v10.5.1 (spec 5.1): record the review row + metrics for the UI.
    latest_bar = final_df["Date"].max() if "Date" in final_df.columns else "unknown"
    try:
        from quant.portfolio.history import record_review
        from quant.reporting.artifacts import save_metrics
        record_review(
            value_eur=total_value, invested_eur=invested,
            cash_eur=account.cash_eur, pnl_eur=pnl_eur, conn=conn,
        )
        metrics_payload = {
            "review_ts": today,
            "latest_bar": latest_bar,
            "regime": regime_block,
            "review_status": "ok",
        }
        save_metrics(run_dir, metrics_payload)
    except Exception as e:  # noqa: BLE001
        logger.warning("Review history/metrics failed (non-fatal): %s", e)

    # ── Verbose-only detail (R4: diagnostics go to --verbose + the run log) ──
    if reporter.verbose:
        reporter.detail(f" CURRENCY: 1 EUR = {get_eur_rate():.4f} USD")
        reporter.detail("\n" + "=" * 40)
        reporter.detail("TOP 3 BUY OPPORTUNITIES")
        reporter.detail("=" * 40)
        buys_with_price = final_df[
            (final_df['Signal'] == 'BUY') & (final_df['Current_Price'].notna())]
        top_buys = buys_with_price.sort_values(
            by='Active_Score', ascending=False).head(3)
        if not top_buys.empty:
            reporter.detail(top_buys[
                ['Symbol', 'Active_Score', 'Current_Price', 'NLP_Reasoning']
            ].to_string(index=False))
        else:
            reporter.detail("No high-conviction BUY signals found.")

        reporter.detail("\n" + "=" * 145)
        reporter.detail("FULL MARKET SCAN")
        reporter.detail("=" * 145)
        display_cols = ['Symbol', 'Is_ETF', 'Current_Price', 'Structural_Grade', 'Tactical_Grade',
                        'Stewardship', 'Horizon', 'Signal', 'Active_Score', 'NLP_Reasoning']
        reporter.detail(final_df[display_cols].to_string(index=False))

        if not audit_res.empty:
            reporter.detail("\n" + "=" * 120)
            reporter.detail("FULL PORTFOLIO AUDIT (Tier-Aware)")
            reporter.detail("=" * 120)
            cols = ['Symbol', 'Tier', 'Current_Weight', 'Target_Weight', 'Drift',
                    'Invested_EUR', 'Value_EUR', 'Real_PnL_EUR', 'Real_PnL_Pct',
                    'FX_Impact_EUR', 'Recon_Deviation', 'Recon_Flag',
                    'Current_Price_Native', 'Current_Price_EUR',
                    'Signal', 'Horizon', 'Recommendation']
            reporter.detail(audit_res[cols].to_string(index=False))
            eff = account_effectiveness(audit_res, port_df)
            reporter.detail(print_effectiveness_report.__doc__ or "")
            reporter.detail(str(eff))

        reporter.detail("\n" + obs.summary())

    # ── Phase 4: Daily Push Notification (canonical actions) ─────────────────
    notify_instructions = [
        {
            "route": a["action"],
            "symbol": a["symbol"],
            "isin": "",
            "min_trade_size_eur": a["amount_eur"] or 0.0,
        }
        for a in actions
    ]
    cash_alloc = (account.cash_eur / total_value) if (
        account.cash_is_set and total_value > 0) else 0.0
    notify_daily(
        total_value=total_value,
        cash_allocation=cash_alloc,
        instructions=notify_instructions,
        risk_warnings=[],
    )

    # ── v10.4.0 (Phase 4): persist telemetry + publish scoring_complete ─────
    try:
        import json
        with open(os.path.join(run_dir, "telemetry.json"), "w") as f:
            json.dump(obs.to_json(), f, indent=2, default=str)
    except Exception as e:  # noqa: BLE001
        logger.warning("Telemetry persist failed (non-fatal): %s", e)
    try:
        from quant.infra.event_bus import EVENTS, EventBus
        EventBus().publish(EVENTS["SCORING_COMPLETE"], {"symbols": len(final_df)})
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    main()
