# Quant-AI v8.3 — Global Stochastic Engine

A professional-grade Python pipeline for systematic equity analysis. Scans a multi-sector universe using a hierarchical data architecture, scores assets using Hidden Markov Models (HMM) and sector-aware fundamentals, and audits portfolios with volatility-adjusted technicals.

**Air-Gapped Reliability.** Sentiment analysis runs locally via [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert). The engine forces offline execution for NLP to guarantee zero downtime during third-party API outages.

---

## New in v8

* **Stochastic Market State (HMM)** — Replaces static technicals with a 2-state Gaussian Hidden Markov Model (HMM) coupled with GARCH volatility to calculate real-time "Bull Regime" probabilities.
* **Global Sentiment Fallback** — US equities are scored via SEC 8-K filings; European and international assets automatically fallback to Yahoo Finance RSS feeds for seamless global coverage.
* **Sector-Aware Stewardship** — Financials and banks are now graded dynamically on Price-to-Book (P/B) and Interest Coverage, while Industrials/Tech are graded on Debt-to-Equity (D/E).
* **Parallel Multiprocessing** — The quant engine utilizes a pure CPU-bound `ProcessPoolExecutor` with strict network I/O isolation to prevent thread deadlocks and drastically reduce scan times.
* **Fast-Pass Gatekeeper** — A pre-computation filter instantly discards low-quality assets (negative PE, low ROE) before committing CPU resources to the heavy NLP/HMM pipeline. 
* **NLP Persistent Cache** — An SQLite-backed caching system ("Cache Hit") bypasses redundant neural network inference for assets evaluated within the same window.

---

## Features

* **Dynamic Sector Universe** — Spans strategic sectors (Uranium, Semiconductors, AI, Defense, European Financials, Precious Metals, and more).
* **Local FinBERT NLP** — Deterministic sentiment scores mapped directly to mathematical weights. Processes up to 4000 tokens (8 chunks) to capture executive summaries without CPU starvation.
* **Composite Scoring Logic** — Triangulates three vectors: Fundamental Structural Grade + Stochastic Tactical Grade + Intrinsic Stewardship.
* **Bifurcated Horizons** — Categorizes assets into RETIREMENT (20yr+), BUSINESS COLLATERAL (10yr), LIQUIDITY CORE (12-Month targets), or SPECULATIVE (Short-Term).
* **Portfolio Audit** — Cross-references `portfolio.csv` against live market data to issue URGENT SELL, DCA OK, or HOLD directives based on capital preservation logic.
* **Kelly Criterion Sizing** — Dynamic position sizing recommendations normalized against asset annual volatility (Target Volatility).

---

## Architecture

```text
quant/
├── main.py                    # Orchestration engine (Fast Pass → Multiprocessing → Audit)
├── data_updater.py            # Generates market_data.parquet for offline execution
├── scoring.py                 # HMM, GARCH, Sector-Aware Stewardship & Horizons
├── sentiment.py               # Offline FinBERT engine & chunking logic
├── sec_edgar.py               # Primary text fetcher (US SEC 8-K)
├── news.py                    # Fallback text fetcher (Yahoo RSS for EU/Intl)
├── fundamentals.py            # Hierarchical metrics service (P/B, D/E, ROE, etc.)
├── indicators.py              # Vectorised technical primitives
├── risk.py                    # VaR penalty and position sizing 
├── portfolio.py               # Audit decision engine for user holdings
├── universe.py                # Asset and sector definitions
├── backtest.py                # Historical execution simulation
├── fundamentals_cache.sqlite  # Automated fundamental data persistence
├── nlp_cache.sqlite           # Persistent NLP sentiment storage
└── outputs/                   # Market scans and audit reports (.csv)
```

---

## Scoring Model (v8)

The composite logic prioritizes structural health while using probabilistic models to time entries.

| Category | Components | Logic |
| :--- | :--- | :--- |
| **Structural Grade** | PE, PEG, ROE | Heavily weighted by Stewardship. Rewards efficiency and fair valuation. |
| **Stewardship** | D/E, P/B, ICR | Sector-aware balance sheet stress test. Forms the "Quality Floor." |
| **Tactical Grade** | HMM, FinBERT, VaR | 60% HMM Bull Probability + 20% NLP Sentiment – Value-at-Risk Penalty. |

### Trade Signal Logic

Signals are governed by the **Stewardship Floor**. If an asset fails solvency checks, the system actively downgrades its horizon to SPECULATIVE and tightens buying conditions.

| Condition | Signal |
| :--- | :--- |
| **Tactical Grade ≥ 70 (Speculative) OR Struct ≥ 85 (Retirement)** | **BUY** |
| **Tactical Grade ≥ 50 AND Stewardship is stable** | **HOLD** |
| **Low Tactical Grade + Poor Sentiment (Score < 40)** | **SELL** |

---

## Investment Horizon Classification

Assets are assigned to specific holding environments based on their structural DNA and recent momentum.

| Category | Target Profile |
| :--- | :--- |
| **RETIREMENT (20yr+)** | Impeccable fundamentals, high structural scores. Suitable for "set-and-forget" capital. |
| **BUSINESS COLLATERAL (10yr)** | Stable blue-chips with lower volatility profiles. |
| **LIQUIDITY CORE (12-Month)** | High structural & tactical alignment. Optimized for mid-term capital extraction. |
| **SPECULATIVE (Short-Term)** | Fails the quality floor or relies entirely on short-term HMM/NLP momentum. |

---

## Setup & Execution

### 1. Clone and Install

```bash
git clone https://github.com/akmal523/quant.git
cd quant
pip install -r requirements.txt
```

### 2. Prepare the Environment

The v8 engine relies on a local parquet file for rapid offline data loading.

```bash
# Pull latest market data and cache it locally
python3 data_updater.py
```

### 3. Setup Holdings (Optional)

Create a `portfolio.csv` in the root directory to enable the Portfolio Audit module. Assets listed here will automatically bypass the Fast-Pass filter.

```csv
Symbol,Buy_Price,Amount_EUR
ECL,285.50,150
HAM,14.20,150
```

### 4. Run the Quant Engine

```bash
python3 main.py
```

*Note: On the very first run, `sentiment.py` will download the FinBERT weights. Subsequent runs are strictly air-gapped (`TRANSFORMERS_OFFLINE=1`) to prevent 500 Internal Server Errors from upstream API outages.*

---

## Output Files

| File | Description |
|---|---|
| `outputs/market_scan_v8.csv` | Master report for all surviving assets including HMM stats, Tactical Grades, and Horizon directives. |
| `outputs/portfolio_audit.csv` | Specific, actionable decisions (BUY MORE, HOLD, URGENT SELL) mapped to your personal holdings. |

---

## Disclaimer

All output is for informational and educational purposes only. The engine's probabilistic models and sentiment analysis do not constitute financial advice. Past performance does not guarantee future results.
