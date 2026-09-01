# Quant-AI v10.0 - Global Stochastic Equity Engine (EUR-Native)

A professional-grade Python pipeline for systematic multi-sector equity analysis. Scans ~300 instruments across 20 sectors, normalises global currencies to EUR, and scores assets using a cross-sectional factor model, market-regime HMM, EWMA volatility, batched FinBERT NLP sentiment, and sector-aware fundamental stewardship.

**Trade Republic Ready.** The engine detects native currency (USD, CHF, GBP, DKK, NOK, SEK, CAD, AUD, KRW, GBX) and converts to EUR using live FX rates from Yahoo Finance and the ECB (Frankfurter API).

**Architecture principle:** Python is the orchestrator; the slow parts are pushed into Rust (Polars), SQL (DuckDB), and C++-backed tools (cvxpy, selectolax).

---

## Architecture

```
quant/
├── main.py                 # Orchestration engine (Smart Funnel -> Batch FinBERT -> Multiprocessing -> Audit)
├── data_updater.py          # Incremental market data fetcher (ThreadPoolExecutor, 10x faster)
├── build_features.py        # Vectorized cross-sectional feature engine (Polars)
├── optimizer.py             # Portfolio optimizer (cvxpy, Ledoit-Wolf covariance)
├── validation.py            # Data validation + liquidity filters
├── artifacts.py             # Run artifacts + structured JSON logging
├── currency.py              # Real-time FX normalisation with graceful degradation
├── async_fetcher.py         # Concurrent SEC 8-K + News RSS network I/O
├── scoring.py               # Factor model, market-regime HMM, stewardship, capital allocation
├── sentiment.py             # Batched FinBERT NLP (NLPScorer + FinBERTBatchScorer)
├── indicators.py            # EWMA volatility (primary) + RSI + ATR
├── risk.py                  # Empirical VaR and Sortino ratio
├── backtest.py              # WFO + cost-aware backtest (T+1 execution)
├── portfolio.py             # Portfolio audit with PnL tracking
├── fundamentals.py          # Hierarchical fundamentals + point-in-time history
├── database.py              # DuckDB thread-local connection management
├── universe.py              # 20-sector asset universe + ETF detection + geo risk tables
├── config.py                # Centralised runtime settings
├── reporting.py             # Terminal output + Excel/CSV export
├── sec_edgar.py             # SEC 8-K downloader (selectolax C-parser)
├── news.py                  # Yahoo Finance RSS headline fetcher
├── mailer.py                # Optional email reporting
│
├── test_scoring.py          # 20 unit tests for scoring engine
├── test_backtest_validity.py # 6 validation tests for backtesting
├── test_factors.py          # 9 tests: factor model, no-lookahead, costs, liquidity
├── test_cache.py            # Fundamentals cache + ICR validation
├── test_db.py               # DuckDB schema I/O validation
├── test_async.py            # Concurrent text fetch speed validation
├── test_e2e_state.py        # End-to-end pipeline state verification
├── test_market_data.py      # DuckDB <-> Pandas bridge test
│
├── portfolio.csv            # Your Trade Republic holdings
├── quant_cache.duckdb       # Unified OLAP storage (market history + fundamentals + NLP)
├── outputs/                 # Market scan reports + run artifacts (run_<timestamp>/)
└── plans/                   # Engineering change proposals
```

---

## New in v10.0

### 1. Smart Funnel Architecture (Tiered Execution)
[`main.py`](main.py) now runs a 3-tier funnel instead of scoring everything:
- **Tier 1 (us):** Fast fundamental filter (PE, ROE).
- **Tier 2 (ms):** Fast technical filter - `Close > 200 SMA` uptrend check (vectorized).
- **Tier 3 (s):** Heavy NLP/SEC scraping only on funnel survivors.

This cuts total execution time by 50-60% by skipping expensive NLP on assets with poor technicals.

### 2. EWMA Volatility (replaces per-asset GARCH MLE)
[`indicators.py`](indicators.py) adds `fast_volatility()` - vectorized EWMA (~1000x faster than GARCH MLE) and makes it the primary volatility source in `add_all_indicators()`. GARCH retained as an optional fallback path.

### 3. Market-Regime HMM (fit once, not per-asset)
[`scoring.py`](scoring.py) adds `fit_market_regime()` - fits a GaussianHMM **once** on a broad index (SPY or longest-history proxy) to derive the macro bull probability, then applies it to all assets. Eliminates ~90s of per-asset HMM fitting and is statistically sounder.

### 4. Batched FinBERT Inference
[`sentiment.py`](sentiment.py) adds `FinBERTBatchScorer.score_texts()` - pools chunks from all documents, pads them into a single tensor, and runs one forward pass per batch (AVX2 vectorization). Also introduces the OOP `NLPScorer` with dependency injection.

### 5. Polars Data Engine
[`main.py`](main.py) reads DuckDB to Polars natively (`.pl()`), computing log-returns vectorized across all symbols in Rust. No pandas intermediate, no GIL.

### 6. Selectolax SEC Parser
[`sec_edgar.py`](sec_edgar.py) replaces BeautifulSoup with the C-based `selectolax` parser (~50x faster on multi-MB SEC filings).

### 7. Incremental Data Updates
[`data_updater.py`](data_updater.py) now fetches only data after each symbol's `MAX(Date)` (with overlap) instead of re-downloading 5 years every run. Reduces update time from minutes to seconds.

### 8. Vectorized Feature Engine
[`build_features.py`](build_features.py) computes momentum (1d/1m/6m/12m), SMA20/200, vol20/60, ADV20, uptrend, trend_strength, and max-drawdown across the whole universe in one Polars pass.

### 9. Cross-Sectional Factor Scoring
[`scoring.py`](scoring.py) adds `factor_scores()` - z-scores value/quality/momentum/low-risk/sentiment across the universe and combines with weights (0.25/0.25/0.20/0.15/0.15), replacing hardcoded absolute thresholds. `sector_neutral_rank()` neutralizes sector bias.

### 10. Point-in-Time Fundamentals (No Lookahead)
[`database.py`](database.py) adds the `fundamentals_history` table; [`fundamentals.py`](fundamentals.py) adds `save_fundamentals_history()` and `get_fundamentals_as_of()` which filter `published_date <= as_of_date` - eliminating lookahead bias in backtests.

### 11. Cost-Aware Backtest
[`backtest.py`](backtest.py) adds `run_cost_aware_backtest()` - executes at T+1 open, applies 15bps round-trip cost, reports net/gross PnL and cost drag.

### 12. Portfolio Optimizer
[`optimizer.py`](optimizer.py) adds `optimize_portfolio()` (cvxpy) with Ledoit-Wolf shrunk covariance and max-weight/sector/turnover constraints - replacing independent BUY/SELL labels with risk-aware weights.

### 13. Data Validation & Liquidity Filters
[`validation.py`](validation.py) adds `validate_market_data()`, `sanitize_fundamentals()`, `liquidity_score()`, `is_liquid()`, and `max_trade_size()` to prevent bad data from corrupting signals.

### 14. Run Artifacts & Structured Logging
[`artifacts.py`](artifacts.py) writes timestamped `outputs/run_<ts>/` directories (factor scores, NLP scores, run config, metrics as parquet/JSON) and provides a JSON `StructuredLogger`.

### 15. Fixed Pre-Existing Bugs
- [`scoring.py`](scoring.py) `kelly_position_size()` and `target_volatility_size()` referenced `KELLY_FRACTION`/`TARGET_VOLATILITY`/`MAX_POSITION_PCT` without importing them - added local imports (unblocked `test_scoring.py`).

---

## New in v9.0

### Robust NaN Data Handling
Yahoo Finance often appends future trading dates with NaN prices as the last row of `history()` output. `data_updater.py` now uses `df.dropna(subset=['Close'])` before reading the latest close price, ensuring all 277 tickers are fetched with valid price data regardless of trailing NaN rows.

Assets with no valid price data are no longer silently dropped. `main.py` returns skeleton scan results for such symbols (rather than `None`), and `portfolio.py` shows `"NO DATA"` in the portfolio audit instead of `"NOT SCANNED"` with NaN PnL.

### FinBERT Runs Once in Main Process
Previous versions loaded the FinBERT NLP model in **every multiprocessing worker** (4 workers x ~800MB = 3.2GB RAM). Now FinBERT is initialised once in the main process before the worker pool starts. Workers receive pre-computed `nlp_data` dicts - no model loading, no `init_worker()` needed. **~2.4GB RAM saved.**

### Proper Logging Infrastructure
All `print()` debugging replaced with structured `logging.info/warning/exception` calls throughout the pipeline. Timestamped output (`HH:MM:SS [LEVEL] message`), with worker crashes captured via `logger.exception()` for full tracebacks.

### Comprehensive Unit Test Suite
- **20 tests** for the scoring engine: HMM regime detection, stewardship scoring (general & financials), structural/tactical grade composition, capital allocation (CORE/SPECULATIVE/HOLD), Kelly/volatility position sizing, fast filter boundaries.
- **6 tests** for the backtesting engine: survivorship bias warning emission, WFO window alignment, insufficient-data handling, macro/historical backtest edge cases.

### GARCH Scale Stability
Auto-scales log-returns to unit variance before GARCH(1,1) fitting, suppressing `DataScaleWarning` for low-price assets (e.g. ETFs trading at 0.67 EUR). Falls back to EWMA when GARCH cannot converge or data < 252 observations.

### Parallel Data Updater
`data_updater.py` rewritten with `ThreadPoolExecutor` (10 workers). 300 tickers fetched in ~30 seconds instead of 150+ seconds sequential.

### Graceful FX Degradation
`currency.py` no longer crashes the entire scan when EUR/USD rate cannot be fetched (ECB API + Yahoo both down). Falls back to 1.0 with a warning instead of `RuntimeError`.

### Clean Configuration
Removed 7 redundant threshold constants from `config.py` that were duplicated under different names (e.g. `MIN_ROE`/`FILTER_MIN_ROE`, `MAX_PE`/`FILTER_MAX_PE`, `GEN_MAX_DE`/`STW_GEN_MAX_DE`). One source of truth per parameter.

### Better NLP Reasoning
"No text data found" changed to `"No SEC/News data available - neutral score applied"` with debug-level logging to distinguish genuine data gaps from scoring issues.

### Fixed Critical Bugs
- **NaN close price crash** in `data_updater.py` - Yahoo Finance appends future trading dates with NaN prices as the last row. `df['Close'].iloc[-1]` picked up NaN for ALL US-listed tickers, blocking 174/277 instruments. Fixed with `df.dropna(subset=['Close'])` before accessing the latest close.
- **Portfolio symbols silently dropped** from scan results in `main.py` - `process_asset()` returned `None` on NaN close, causing portfolio holdings to show as `"NOT SCANNED"` in the audit. Fixed by returning a skeleton result with `Active_Score: 0.0` and `Signal: N/A`.
- **Missing portfolio symbols fallback** - symbols with no market data at all (not in DuckDB) now get a placeholder entry in scan results instead of being silently absent.
- **Duplicate data loading** in `main.py` - market data was loaded twice, the second load discarding chronological sorting. HMM/GARCH received scrambled data.
- **`init_db()` never called** - NLP cache `INSERT` could crash with `CatalogException`.
- **`backtest.py` import error** - `from indicators import rsi, atr` referenced non-existent functions. Both implemented with proper EWMA computation.

---

## Features

- **20-Sector Universe** - Uranium, Energy, Oil & Gas, Defense, Cybersecurity, Gold, Silver, Copper, Lithium, Quantum, Semiconductors, AI/Cloud, Logistics, Banking, Insurance, Healthcare, Water, Agriculture, Real Estate, Broad ETFs.
- **Universal ETF Detection** - Automatically identifies ETFs by sector, display name keywords, and hardcoded fallbacks. ETFs bypass corporate fundamental filters and score purely on market-regime momentum.
- **Global Sentiment Fallback** - US equities scored via SEC 8-K filings; international assets fall back to News RSS.
- **Batched Local FinBERT NLP** - Air-gapped sentiment analysis ([ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert)). `FinBERTBatchScorer` pools chunks across documents into single vectorized forward passes. No API calls.
- **Market-Regime HMM** - Gaussian HMM fit once on a broad index to derive the macro Bull/Bear probability, applied to all assets.
- **Cross-Sectional Factor Model** - Value/Quality/Momentum/Low-Risk/Sentiment z-scores combined with weights, replacing hardcoded thresholds. Sector-neutralized ranking.
- **Smart Funnel** - Tiered execution: fundamental filter, technical uptrend filter, heavy NLP only on survivors.
- **Stewardship Override** - The "Quality Floor": even stocks with strong momentum are downgraded to SPECULATIVE if Debt-to-Equity exceeds limits or Interest Coverage Ratio is inadequate.
- **Dynamic Currency Normalisation** - OHLCV data auto-converted to EUR using live FX rates. Supports USD, CHF, GBP, GBX, DKK, NOK, SEK, CAD, AUD, KRW.
- **Portfolio Audit** - Cross-references `portfolio.csv` against live data. Issues BUY MORE (DCA OK), HOLD, URGENT SELL directives with accurate EUR-native PnL.
- **Cost-Aware Backtest** - T+1 execution with 15bps round-trip costs; reports net/gross PnL and cost drag.
- **Portfolio Optimizer** - cvxpy mean-variance with Ledoit-Wolf covariance, max-weight/sector/turnover constraints.
- **Point-in-Time Fundamentals** - `fundamentals_history` table with `published_date` filtering to eliminate lookahead bias.
- **Walk-Forward Optimisation** - Rolling in-sample / out-of-sample windows prove strategy generalisation. Survivorship bias warning emitted on every call.
- **DuckDB + Polars Analytical Core** - High-performance OLAP storage (340K+ rows) with vectorized Rust feature computation.
- **Run Artifacts** - Each run writes `outputs/run_<timestamp>/` with factor scores, NLP scores, config, and metrics.

---

## Scoring Model (v10.0)

| Category | Weight | Components |
|:---|:---:|:---|
| **Fundamentals** | 30% | PE, PEG, ROE - reward capital efficiency |
| **Stewardship** | 30% | D/E, P/B, ICR - sector-aware balance sheet stress test |
| **Technical (HMM)** | 15% | EWMA volatility -> market-regime HMM -> Bull/Bear probability |
| **Sentiment (FinBERT)** | 25% | SEC 8-K or News RSS -> batched attention-masked inference |

> **v10.0 factor model:** The pipeline also exposes a cross-sectional factor score
> (`factor_scores()` in `scoring.py`) that z-scores Value/Quality/Momentum/Low-Risk/
> Sentiment across the universe and combines them with weights
> (0.25/0.25/0.20/0.15/0.15), with optional sector-neutralized ranking.

### Horizon & Signal Logic

| Horizon | Requirements | Signal Logic |
|:---|:---|:---|
| **CORE (12-Month)** | Structural >= 75, Stewardship >= 15, Tactical >= 60 | BUY if all met; score = 0.4xstruct + 0.6xtact |
| **HOLD** | Stewardship >= 15 but structural < 75 | HOLD; score = structural grade |
| **SPECULATIVE** | Stewardship < 15 OR structural < 50 | BUY if Tactical >= 70, else SELL |

---

## Setup & Execution

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/quant.git
cd quant
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Update Market Data

```bash
python3 data_updater.py
```

Fetches ~300 tickers across 20 sectors with 10 parallel workers. **Incremental mode:**
on subsequent runs it fetches only data after each symbol's last stored date (with a
5-day overlap), reducing update time from minutes to seconds.

### 3. Configure Portfolio

Edit `portfolio.csv` with your Trade Republic holdings (EUR cost basis):

```csv
Symbol,Buy_Price,Amount_EUR
IWDA.AS,120.50,115.00
HEI.DE,185.20,200.00
LMT,509.88,75.00
RHO.DE,356.40,48.26
```

Comments after `#` are automatically stripped.

### 4. Run the Dashboard

```bash
python3 main.py
```

---

## Output Signals

| Decision | Condition |
|:---|:---|
| **BUY MORE (DCA OK)** | High quality (Active Score > 80) and position at or below entry, or within high-conviction window |
| **HOLD** | Fundamentals remain strong but tactical timing or profit level suggests waiting |
| **URGENT SELL** | Significant fundamental decay (low stewardship) or extreme negative sentiment |
| **NO DATA** | Symbol found in portfolio but has no valid price data (e.g. Yahoo Finance unavailable) |
| **NOT SCANNED** | Asset not found in current market universe |

---

## Configuration

All tunable parameters in [`config.py`](config.py):

| Parameter | Default | Description |
|:---|---:|:---|
| `WEIGHT_FUNDAMENTALS` | 30 | PE + PEG + ROE contribution |
| `WEIGHT_STEWARDSHIP` | 30 | D/E + ICR + Payout policy |
| `WEIGHT_TECHNICAL` | 15 | HMM regime score |
| `WEIGHT_SENTIMENT` | 25 | FinBERT NLP sentiment |
| `FILTER_MAX_PE` | 25.0 | Screener PE ceiling |
| `FILTER_MIN_ROE` | 0.20 | Screener ROE floor |
| `STW_GEN_MAX_DE` | 0.5 | Max D/E for non-financials |
| `STW_GEN_MIN_ICR` | 5.0 | Min interest coverage ratio |
| `MIN_STRUCT_GRADE_FOR_BUY` | 75 | Minimum structural grade for CORE classification |
| `MIN_TACT_GRADE_FOR_BUY` | 70 | Minimum tactical grade for SPECULATIVE BUY |
| `KELLY_FRACTION` | 0.25 | Quarter-Kelly position sizing |
| `TARGET_VOLATILITY` | 0.15 | 15% annualised target vol |
| `MAX_POSITION_PCT` | 0.10 | 10% hard cap per position |

---

## Test Suite

```bash
# Scoring engine (20 tests)
python3 test_scoring.py

# Backtesting validity (6 tests)
python3 test_backtest_validity.py

# Factor model, no-lookahead, costs, liquidity (9 tests)
python3 test_factors.py

# Database I/O
python3 test_db.py

# Fundamentals cache + ICR
python3 test_cache.py

# End-to-end pipeline state
python3 test_e2e_state.py
```

---

## Disclaimer

All output is for informational purposes. Probabilistic models and NLP sentiment analysis involve inherent risk. **Past performance does not guarantee future results.** The universe contains only currently-listed instruments - historical backtest figures are systematically overstated due to survivorship bias.
