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
from quant import paths
import hashlib
import logging
import multiprocessing
import numpy as np
import pandas as pd
import polars as pl
from concurrent.futures import ProcessPoolExecutor, as_completed

from quant.data.database import get_connection, init_db
from quant.data.currency import apply_fx_conversion, get_eur_rate
from quant.portfolio.portfolio import load_portfolio, audit_portfolio, enhanced_portfolio_audit, account_effectiveness, print_effectiveness_report
from quant.features.indicators import add_all_indicators, fast_volatility
from quant.analytics.sentiment import NLPScorer
from quant.portfolio.risk import calculate_risk_penalty
from quant.config import WEIGHT_TECHNICAL
from quant.analytics.scoring import (
    evaluate_structural_grade,
    evaluate_tactical_grade,
    allocate_capital_regime,
    fit_market_regime,
    stewardship_score_v2,
    apply_fast_filter,
    etf_tactical_grade,
)
from quant.data.fundamentals import get_fundamentals
from quant.data.universe import is_etf
from quant.execution.taxonomy import (
    get_instrument_class, resolve_broker, get_structure,
)
from quant.execution.routing import (
    route_signal, build_execution_instruction, alpha_bps_from_active_score,
)
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


# ── Currency Detection ─────────────────────────────────────────────────────────

def deduce_currency(symbol: str) -> str:
    if "." not in symbol:
        return "USD"
    suffix = symbol.split(".")[-1].upper()
    eur_zones = {"DE", "PA", "AS", "MI", "MC", "BR", "VI", "HE"}
    if suffix in eur_zones: return "EUR"
    if suffix == "L": return "GBX"
    if suffix == "SW": return "CHF"
    if suffix == "CO": return "DKK"
    if suffix == "OL": return "NOK"
    if suffix == "ST": return "SEK"
    if suffix == "TO": return "CAD"
    if suffix == "AX": return "AUD"
    if suffix == "KS": return "KRW"
    return "USD"


# ── Worker Process ────────────────────────────────────────────────────────────

# main.py -> process_asset()
def process_asset(symbol: str, f_data: dict, sector: str, nlp_data: dict,
                  market_regime_prob: float, etf_quality_map: dict | None = None) -> dict | None:
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
        native_ccy = deduce_currency(symbol)
        # Get sector from the queried data
        sector = df['Sector'].iloc[0] if 'Sector' in df.columns else "Unknown"

        last_close = price_hist["Close"].iloc[-1] if "Close" in price_hist.columns else None
        if last_close is None or (isinstance(last_close, float) and (pd.isna(last_close) or np.isnan(last_close))):
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


# ── Part 2: Advanced Portfolio Manager Briefing ───────────────────────────────

def _print_advanced_briefing(port_df, audit_res, final_df, grouped_data) -> None:
    """Assemble and print the unified Part 2 briefing.

    Intent: wire all Part 2 modules (risk monitor, portfolio context, strategy
    engine, cash manager, tax optimizer, attribution, guardrails) into a single
    coherent report. Non-fatal — wrapped in try/except by the caller.
    """
    import numpy as np
    import pandas as pd

    from quant.portfolio.portfolio_context import PortfolioContext
    from quant.portfolio.risk_monitor import RiskMonitor
    from quant.strategy.strategy_engine import StrategyEngine
    from quant.portfolio.cash_manager import CashManager
    from quant.portfolio.tax_optimizer import TaxOptimizer
    from quant.portfolio.behavioral_guardrails import BehavioralGuardrails
    from quant.reporting.reporting_advanced import build_briefing

    # Build a returns matrix from grouped_data (Close pct_change per symbol).
    closes = {}
    for sym, df in grouped_data.items():
        if "Close" in df.columns and not df["Close"].dropna().empty:
            closes[sym] = df["Close"]
    if not closes:
        return
    returns_matrix = pd.DataFrame(closes).pct_change().dropna(how="all")

    # Portfolio context: risk contribution + concentration penalties.
    ctx = PortfolioContext(port_df, returns_matrix)
    risk_contrib = ctx.compute_risk_contribution().to_dict()
    penalties = {s: ctx.concentration_penalty(s) for s in port_df["Symbol"]}

    # Risk monitor: circuit breakers on portfolio value series.
    port_value = port_df["Amount_EUR"].sum()
    # Use a synthetic portfolio value series from the first asset as proxy.
    first_sym = next(iter(closes))
    value_series = closes[first_sym]
    risk_mon = RiskMonitor(value_series)
    risk_status = risk_mon.check_circuit_breakers()

    # Strategy engine: ensemble scores per portfolio symbol.
    engine = StrategyEngine()
    regime = "bull_low_vol" if risk_status.get("drawdown", 0) > -0.05 else "bear"
    strategy_weights = engine.regime_weights(regime)
    ensemble_scores = {}
    for sym in port_df["Symbol"]:
        if sym in returns_matrix.columns:
            ret = returns_matrix[sym].dropna()
            data = {
                "returns_6m": float(ret.tail(126).sum()) if len(ret) else 0.0,
                "volatility_60d": float(ret.tail(60).std()) if len(ret) else 0.0,
                "rsi_14": 50.0,
                "pe_ratio": 0.0,
                "dividend_yield": 0.0,
            }
            ensemble_scores[sym] = engine.compute_ensemble_signal(sym, data, regime)

    # Cash manager: target cash + dip alerts.
    cash_mgr = CashManager()
    cash_target = cash_mgr.target_cash_allocation(regime, vix=18.0, opportunity_score=0.5)
    cash_eur = port_value * 0.10  # placeholder cash
    # Item 6: gate DIP BUY on underweight vs target tier. Only buy dips on
    # positions that are underweight (or not in the audit), never on overweight
    # positions that are already at/above target.
    weight_map = {}
    if not audit_res.empty and {"Current_Weight", "Target_Weight"}.issubset(audit_res.columns):
        for _, r in audit_res.iterrows():
            try:
                cw = float(str(r.get("Current_Weight", "0%")).rstrip("%")) / 100.0
                tw = float(str(r.get("Target_Weight", "0%")).rstrip("%")) / 100.0
                weight_map[r["Symbol"]] = (cw, tw)
            except Exception:
                continue
    dip_alerts = []
    for sym in port_df["Symbol"]:
        if sym in closes:
            s = closes[sym]
            dd = float((s.iloc[-1] - s.max()) / s.max()) if s.max() > 0 else 0.0
            amt = cash_mgr.dip_buying_algorithm(sym, dd, cash_eur)
            if amt > 0:
                cw, tw = weight_map.get(sym, (0.0, 0.0))
                if sym not in weight_map or cw < tw:
                    dip_alerts.append((sym, amt))

    # Tax optimizer.
    tax_df = audit_res.copy()
    if "PnL_EUR" not in tax_df.columns:
        tax_df["PnL_EUR"] = 0.0
    if "Tier" not in tax_df.columns:
        tax_df["Tier"] = "ACTIVE"
    tax_opt = TaxOptimizer(tax_df)
    tax_position = tax_opt.compute_tax_position()
    harvest = tax_opt.harvest_opportunities()

    # Guardrails: block signals on cooldown.
    guardrails = BehavioralGuardrails()
    guardrail_blocks = []
    for sym in port_df["Symbol"]:
        ok, reason = guardrails.check_cooldown(sym)
        if not ok:
            guardrail_blocks.append(f"{sym}: {reason}")

    briefing = build_briefing(
        date_str="2026-09-09",
        portfolio_value=port_value,
        pnl_eur=float(audit_res["PnL_EUR"].sum()) if "PnL_EUR" in audit_res else 0.0,
        pnl_pct=0.0,
        cash_eur=cash_eur,
        cash_pct=cash_eur / port_value if port_value else 0.0,
        risk_status=risk_status,
        risk_contrib=risk_contrib,
        concentration_penalties=penalties,
        regime=regime,
        strategy_weights=strategy_weights,
        ensemble_scores=ensemble_scores,
        cash_target=cash_target,
        dip_alerts=dip_alerts,
        tax_position=tax_position,
        harvest_opportunities=harvest,
        attribution_df=None,
        guardrail_blocks=guardrail_blocks,
    )
    print(briefing)


# ── Main Orchestration ─────────────────────────────────────────────────────────

def main() -> None:
    from quant.infra.observability import ObservabilityCollector
    obs = ObservabilityCollector()

    logger.info("Loading localized database...")
    with obs.step("load_database"):
        conn = get_connection()
        init_db()

    try:
        # Pillar 4: read DuckDB -> Polars natively (Rust, multi-core, no GIL).
        # .pl() avoids pandas intermediate, so no pyarrow dependency needed.
        pl_df = conn.execute("SELECT * FROM market_history ORDER BY Symbol ASC, Date ASC").pl()
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
    grouped_data = {s: df for s, df in grouped_data.items() if s in active_symbols}
    logger.info("Scan universe (funnel + CORE + ACTIVE + portfolio): %d symbols",
                len(grouped_data))

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
    market_regime_prob = 0.5
    regime_sym = "SPY" if "SPY" in grouped_data else max(grouped_data, key=lambda s: len(grouped_data[s]))
    regime_df = grouped_data[regime_sym]
    regime_vol = fast_volatility(regime_df["Close"])
    regime_raw = fit_market_regime(regime_df["Close"], regime_vol)
    market_regime_prob = regime_raw / float(WEIGHT_TECHNICAL)
    logger.info("Market regime (fit on %s): bull prob=%.2f", regime_sym, market_regime_prob)

    # ── Phase 5 (v10.2): cross-sectional ETF quality map ────────────────────
    # Compute the ETF structural grade + momentum z in the MAIN process over the
    # ETF subset (workers lack the full subset). Passed to process_asset.
    etf_quality_map: dict[str, float] = {}
    try:
        from quant.features.build_features import latest_features
        from quant.analytics.scoring import etf_quality_score
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
    import asyncio
    from quant.data.async_fetcher import fetch_all_texts_concurrently
    survivor_texts = asyncio.run(fetch_all_texts_concurrently(list(survivors.keys())))

    # ── Step 3: NLP scoring in MAIN process (single FinBERT, DI) ────────────
    scorer = NLPScorer()

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
            conn.execute("INSERT OR REPLACE INTO nlp_scores (doc_hash, score) VALUES (?, ?)", [h, s])
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

    print(f"\n CURRENCY: 1 EUR = {get_eur_rate():.4f} USD")

    # --- TOP 3 BUY OPPORTUNITIES (Inc. ETFs) ---
    print("\n" + "=" * 40)
    print("TOP 3 BUY OPPORTUNITIES")
    print("=" * 40)
    buys_with_price = final_df[(final_df['Signal'] == 'BUY') & (final_df['Current_Price'].notna())]
    top_buys = buys_with_price.sort_values(by='Active_Score', ascending=False).head(3)
    if not top_buys.empty:
        print(top_buys[['Symbol', 'Active_Score', 'Current_Price', 'NLP_Reasoning']].to_string(index=False))
    else:
        print("No high-conviction BUY signals found.")

    # --- TOP 3 STOCKS (Non-ETF) ---
    print("\n" + "=" * 40)
    print("TOP 3 STOCKS (Non-ETF)")
    print("=" * 40)
    stock_buys = final_df[(final_df['Signal'] == 'BUY') & (final_df['Is_ETF'] == False) & (final_df['Current_Price'].notna())]
    top_stocks = stock_buys.sort_values(by='Active_Score', ascending=False).head(3)
    if not top_stocks.empty:
        print(top_stocks[['Symbol', 'Active_Score', 'Current_Price', 'NLP_Reasoning']].to_string(index=False))
    else:
        print("No stock BUY signals found (only ETFs).")

    # --- FULL MARKET SCAN ---
    print("\n" + "=" * 145)
    print("FULL MARKET SCAN")
    print("=" * 145)
    display_cols = ['Symbol', 'Is_ETF', 'Current_Price', 'Structural_Grade', 'Tactical_Grade',
                    'Stewardship', 'Horizon', 'Signal', 'Active_Score', 'NLP_Reasoning']
    print(final_df[display_cols].to_string(index=False))

    # --- PORTFOLIO AUDIT (tier-aware, drift + fee-aware) ---
    if not port_df.empty:
        # Build market_data dict for liquidity checks from the scan universe.
        market_data = {s: df for s, df in grouped_data.items() if s in set(port_df["Symbol"])}
        audit_res = enhanced_portfolio_audit(
            port_df, final_df, current_date="2026-09-09", market_data=market_data,
        )
        audit_res.to_csv(str(paths.OUTPUTS_DIR / "portfolio_audit.csv"), index=False)

        print("\n" + "=" * 120)
        print("FULL PORTFOLIO AUDIT (Tier-Aware)")
        print("=" * 120)
        cols = ['Symbol', 'Tier', 'Current_Weight', 'Target_Weight', 'Drift',
                'Invested_EUR', 'Value_EUR', 'Real_PnL_EUR', 'Real_PnL_Pct',
                'FX_Impact_EUR', 'Recon_Deviation', 'Recon_Flag',
                'Current_Price_Native', 'Current_Price_EUR',
                'Signal', 'Horizon', 'Recommendation']
        print(audit_res[cols].to_string(index=False))
        print("\n")

        eff = account_effectiveness(audit_res, port_df)
        print_effectiveness_report(eff)

        # ── Part 2: Advanced Portfolio Manager Briefing ──────────────────────
        # Assemble risk monitor, portfolio context, strategy engine, cash
        # manager, tax optimizer, attribution, and guardrails into one report.
        # Wrapped in try/except so a failure never breaks the main pipeline.
        try:
            _print_advanced_briefing(port_df, audit_res, final_df, grouped_data)
        except Exception as e:  # noqa: BLE001
            logger.warning("Advanced briefing failed (non-fatal): %s", e)

    # ── Phase 4: Signal Routing + Daily Push Notification ────────────────────
    # Route each scored asset to SPARPLAN/ACTIVE/CASH and build execution
    # instructions, then push a daily summary to Telegram/Discord.
    instructions = []
    risk_warnings = []
    for _, row in final_df.iterrows():
        sym = row["Symbol"]
        cls = get_instrument_class(sym)
        structure = get_structure(sym)
        route = route_signal(
            structural_grade=float(row.get("Structural_Grade", 0) or 0),
            tactical_grade=float(row.get("Tactical_Grade", 0) or 0),
            instrument_class=cls,
            structure=structure,
        )
        broker = resolve_broker(sym)
        # Dynamic fee hurdle: alpha scales with active score, not a constant.
        active_score = float(row.get("Active_Score", 0) or 0)
        alpha_bps = alpha_bps_from_active_score(active_score)
        inst = build_execution_instruction(
            symbol=sym,
            route=route,
            current_price=float(row.get("Current_Price", 0) or 0),
            capital_eur=100.0,  # placeholder; wire to real allocation in Step 2
            expected_alpha_bps=alpha_bps,
            isin=broker["isin"],
            tr_ticker=broker["tr_ticker"],
        )
        if inst["action"] in ("BUY", "SPARPLAN"):
            instructions.append(inst)

    # Risk warning: Alpha bucket constraint check (placeholder for real weights).
    alpha_pct = final_df[final_df["Is_ETF"] == False]["Active_Score"].mean() if not final_df.empty else 0
    if alpha_pct > 50:
        risk_warnings.append("Alpha Bucket exceeds 50% constraint, rebalancing required.")

    total_value = port_df["Amount_EUR"].sum() if not port_df.empty else 0.0
    notify_daily(
        total_value=total_value,
        cash_allocation=0.10,  # placeholder; wire to optimizer output
        instructions=instructions,
        risk_warnings=risk_warnings,
    )

    # ── Part 3 (Gap #3): Observability summary ──────────────────────────────
    print("\n" + obs.summary())


if __name__ == "__main__":
    main()
