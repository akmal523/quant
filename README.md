# Quant-AI v9.0 — Global Stochastic Equity Engine (EUR-Native)

A professional-grade Python pipeline for systematic multi-sector equity analysis. Scans ~300 instruments across 20 sectors, normalises global currencies to EUR, and scores assets using Hidden Markov Models (HMM), GARCH(1,1) stochastic volatility, FinBERT NLP sentiment, and sector-aware fundamental stewardship.

**Trade Republic Ready.** The engine detects native currency (USD, CHF, GBP, DKK, NOK, SEK, CAD, AUD, KRW, GBX) and converts to EUR using live FX rates from Yahoo Finance and the ECB (Frankfurter API).

---

## Architecture

```
quant/
├── main.py                 # Orchestration engine (Async Fetch → FinBERT → Multiprocessing → Audit)
├── data_updater.py          # Parallel market data fetcher (ThreadPoolExecutor, 10× faster)
├── currency.py              # Real-time FX normalisation with graceful degradation
├── async_fetcher.py         # Concurrent SEC 8-K + News RSS network I/O
├── scoring.py               # HMM regime detection, stewardship, capital allocation
├── sentiment.py             # Local FinBERT NLP with proper attention masking
├── indicators.py            # GARCH(1,1) + EWMA fallback + RSI + ATR
├── risk.py                  # Empirical VaR and Sortino ratio
├── backtest.py              # Walk-Forward Optimisation + macro/historical backtest
├── portfolio.py             # Portfolio audit with PnL tracking
├── fundamentals.py          # Hierarchical fundamentals fetcher (DuckDB → Alpha Vantage → yfinance)
├── database.py              # DuckDB thread-local connection management
├── universe.py              # 20-sector asset universe + ETF detection + geo risk tables
├── config.py                # Centralised runtime settings
├── reporting.py             # Terminal output + Excel/CSV export
├── sec_edgar.py             # SEC 8-K downloader with daemon-thread timeout
├── news.py                  # Yahoo Finance RSS headline fetcher
├── mailer.py                # Optional email reporting
│
├── test_scoring.py          # 20 unit tests for scoring engine
├── test_backtest_validity.py # 6 validation tests for backtesting
├── test_cache.py            # Fundamentals cache + ICR validation
├── test_db.py               # DuckDB schema I/O validation
├── test_async.py            # Concurrent text fetch speed validation
├── test_e2e_state.py        # End-to-end pipeline state verification
├── test_market_data.py      # DuckDB ←→ Pandas bridge test
│
├── portfolio.csv            # Your Trade Republic holdings
├── quant_cache.duckdb       # Unified OLAP storage (market history + fundamentals + NLP)
├── outputs/                 # Market scan reports (.csv)
└── plans/                   # Engineering change proposals
```

---

## New in v9.0

### FinBERT Runs Once in Main Process
Previous versions loaded the FinBERT NLP model in **every multiprocessing worker** (4 workers × ~800MB = 3.2GB RAM). Now FinBERT is initialised once in the main process before the worker pool starts. Workers receive pre-computed `nlp_data` dicts — no model loading, no `init_worker()` needed. **~2.4GB RAM saved.**

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
"No text data found" changed to `"No SEC/News data available — neutral score applied"` with debug-level logging to distinguish genuine data gaps from scoring issues.

### Fixed Critical Bugs
- **Duplicate data loading** in `main.py` — market data was loaded twice, the second load discarding chronological sorting. HMM/GARCH received scrambled data.
- **`init_db()` never called** — NLP cache `INSERT` could crash with `CatalogException`.
- **`backtest.py` import error** — `from indicators import rsi, atr` referenced non-existent functions. Both implemented with proper EWMA computation.

---

## Features

- **20-Sector Universe** — Uranium, Energy, Oil & Gas, Defense, Cybersecurity, Gold, Silver, Copper, Lithium, Quantum, Semiconductors, AI/Cloud, Logistics, Banking, Insurance, Healthcare, Water, Agriculture, Real Estate, Broad ETFs.
- **Universal ETF Detection** — Automatically identifies ETFs by sector, display name keywords, and hardcoded fallbacks. ETFs bypass corporate fundamental filters and score purely on HMM momentum.
- **Global Sentiment Fallback** — US equities scored via SEC 8-K filings; international assets fall back to News RSS.
- **Local FinBERT NLP** — Air-gapped sentiment analysis ([ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert)). Processes text in 512-token chunks with proper attention masks. No API calls.
- **Stochastic Market State (HMM)** — Gaussian Hidden Markov Model + GARCH(1,1) volatility to identify Bull/Bear regimes.
- **Stewardship Override** — The "Quality Floor": even stocks with strong momentum are downgraded to SPECULATIVE if Debt-to-Equity exceeds limits or Interest Coverage Ratio is inadequate.
- **Dynamic Currency Normalisation** — OHLCV data auto-converted to EUR using live FX rates. Supports USD, CHF, GBP, GBX, DKK, NOK, SEK, CAD, AUD, KRW.
- **Portfolio Audit** — Cross-references `portfolio.csv` against live data. Issues BUY MORE (DCA OK), HOLD, URGENT SELL directives with accurate EUR-native PnL.
- **Walk-Forward Optimisation** — Rolling in-sample / out-of-sample windows prove strategy generalisation. Survivorship bias warning emitted on every call.
- **DuckDB Analytical Core** — High-performance OLAP storage. Handles 340K+ rows of market history with zero-latency analytical queries.

---

## Scoring Model (v9.0)

| Category | Weight | Components |
|:---|:---:|:---|
| **Fundamentals** | 30% | PE, PEG, ROE — reward capital efficiency |
| **Stewardship** | 30% | D/E, P/B, ICR — sector-aware balance sheet stress test |
| **Technical (HMM)** | 15% | GARCH(1,1) → HMM regime → Bull/Bear probability |
| **Sentiment (FinBERT)** | 25% | SEC 8-K or News RSS → chunked attention-masked inference |

### Horizon & Signal Logic

| Horizon | Requirements | Signal Logic |
|:---|:---|:---|
| **CORE (12-Month)** | Structural ≥ 75, Stewardship ≥ 15, Tactical ≥ 60 | BUY if all met; score = 0.4×struct + 0.6×tact |
| **HOLD** | Stewardship ≥ 15 but structural < 75 | HOLD; score = structural grade |
| **SPECULATIVE** | Stewardship < 15 OR structural < 50 | BUY if Tactical ≥ 70, else SELL |

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

Fetches ~300 tickers across 20 sectors with 10 parallel workers.

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

# Database I/O
python3 test_db.py

# Fundamentals cache + ICR
python3 test_cache.py

# End-to-end pipeline state
python3 test_e2e_state.py
```

---

## Disclaimer

All output is for informational purposes. Probabilistic models and NLP sentiment analysis involve inherent risk. **Past performance does not guarantee future results.** The universe contains only currently-listed instruments — historical backtest figures are systematically overstated due to survivorship bias.
