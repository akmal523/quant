---

# Quant-AI v8.4 — Global Stochastic Engine (EUR-Native)

A professional-grade Python pipeline for systematic equity analysis. Scans a multi-sector universe, normalizes global currencies to EUR, and scores assets using Hidden Markov Models (HMM) and sector-aware fundamentals.

**Trade Republic Ready.** The engine automatically detects the native currency of an asset (USD, CHF, GBP) and converts it to EUR using live FX rates from Yahoo Finance and the European Central Bank (ECB) via the Frankfurter API.

---

## New in v8.4

* **Dynamic Currency Normalization (`currency.py`)** — Automatically translates global prices (USD, CHF, GBP, etc.) into EUR. Ensures PnL and scoring are mathematically consistent for Euro-based portfolios.
* **DuckDB Analytical Core** — Replaces standard SQLite with high-performance DuckDB (`quant_cache.duckdb`). Handles 300,000+ rows of market history with zero-latency analytical queries.
* **Asynchronous Text Pipeline** — Utilizes `asyncio` and `aiohttp` to fetch SEC 8-K filings and Global News RSS feeds concurrently, reducing network I/O time by ~70%.
* **Live Console Dashboard** — Direct-to-terminal reporting with suppressed library noise, featuring a "Top 3 High-Conviction Buys" summary for rapid decision making.
* **Improved Roche/International Logic** — Refined stewardship math to handle international accounting differences (e.g., Debt-to-Equity percentage vs. ratio scaling).

---

## Features

* **Global Sentiment Fallback** — US equities are scored via SEC 8-K filings; international assets (Roche, SAP, BHP) fallback to News RSS feeds.
* **Local FinBERT NLP** — Air-gapped sentiment analysis using [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert). processes text in 512-token chunks to avoid "long-text" memory spikes.
* **Stochastic Market State (HMM)** — Gaussian Hidden Markov Model + GARCH volatility to identify Bull/Bear regimes.
* **Portfolio Audit** — Cross-references `portfolio.csv` against live data to issue **BUY MORE (DCA OK)**, **HOLD**, or **SELL** directives.

---

## Architecture

```text
quant/
├── main.py                # Orchestration engine (Async Fetch → Multiprocessing → Audit)
├── data_updater.py        # Populates DuckDB with global price history
├── currency.py            # Real-time FX normalization (Yahoo + ECB API)
├── async_fetcher.py       # Concurrent network I/O for SEC and News
├── scoring.py             # HMM, GARCH, and Stewardship logic
├── sentiment.py           # Local FinBERT inference engine
├── database.py            # DuckDB connection and schema management
├── indicators.py          # Vectorized technical indicators
├── portfolio.py           # Audit decision logic for user holdings
├── universe.py            # Global asset definitions & sector maps
├── quant_cache.duckdb     # Unified high-speed storage for history and NLP scores
└── outputs/               # Market scans and audit reports (.csv)
```

---

## Setup & Execution

### 1. Clone and Install

```bash
git clone https://github.com/akmal523/quant.git
cd quant
pip install -r requirements.txt
```

### 2. Update Market Data

The v8.4 engine uses DuckDB for local caching. Run this to fetch the global universe:

```bash
python3 data_updater.py
```

### 3. Setup Euro Portfolio

Edit `portfolio.csv` with your Trade Republic holdings. Use the **Euro price** you paid:

```csv
Symbol,Buy_Price,Amount_EUR
ECL,225.50,150.00
HEI.DE,185.20,200.00
ROG.SW,280.10,100.00
```

### 4. Run the Dashboard

```bash
python3 main.py
```

---

## Scoring Model (v8.4)

| Category | Components | Logic |
| :--- | :--- | :--- |
| **Structural Grade** | PE, PEG, ROE | Fixed for Euro-scaling. Rewards capital efficiency. |
| **Tactical Grade** | HMM, FinBERT, VaR | Probabilistic entry timing based on regime and sentiment. |
| **Stewardship** | D/E, P/B, ICR | Sector-aware balance sheet stress test (Quality Floor). |

---

To make your **README.md** even more professional, we should add a dedicated section on **Asset Classification**. This helps anyone reading the code (including your future self) understand why the engine might suggest "Retirement" for one stock and "Speculative" for another, even if both have a "BUY" signal.

Here is the section formatted for your Markdown file:

---

## Strategic Classification & Horizons

| Horizon | Target Profile | Engine Requirements |
| :--- | :--- | :--- |
| **RETIREMENT (20yr+)** | "Set and Forget" assets. Exceptional compounders with wide moats. | **Structural Grade > 85** + **Stewardship > 20**. High ROE and minimal debt. |
| **BUSINESS COLLATERAL (10yr)** | Stable blue-chips with lower volatility. Assets you could borrow against. | **Structural Grade 70–85**. Reliable interest coverage and price stability. |
| **LIQUIDITY CORE (12-Mo)** | Mid-term growth or value plays optimized for capital extraction. | **High Tactical Grade** (Timing) + **Decent Structural Grade**. |
| **SPECULATIVE (Short-Term)** | Momentum plays or turnarounds. High risk, high reward. | **Structural Grade < 50** OR fails the **Stewardship Floor** (High Debt/Low P/B). |

### The "Quality Floor" Logic
A unique feature of this engine is the **Stewardship Override**. Even if a stock has incredible price momentum (High Tactical Grade), the engine will **force-downgrade** the horizon to **SPECULATIVE** if:
*   The **Debt-to-Equity** ratio exceeds sector-safe limits.
*   The **Interest Coverage Ratio (ICR)** suggests the company is struggling to pay its bills.
*   The **Price-to-Book** ratio (for Financials) indicates the market sees a "black hole" on the balance sheet.

### Why "Hold" vs. "DCA OK"?
The engine differentiates between market signals and your actual bank account:
*   **BUY MORE (DCA OK):** The asset is high quality, and your current position is either at a loss or only slightly profitable. It's time to build the position.
*   **HOLD:** The asset is a "BUY" in the open market, but **you are already up 15%+**. The engine prevents you from "averaging up" too aggressively, protecting your realized gains.

---

## Output Signals

* **BUY MORE (DCA OK)**: Asset is high quality (Active Score > 80) and currently trading at or below your entry, or within a high-conviction window.
* **HOLD**: Fundamentals remain strong, but current profit levels or tactical timing suggest waiting.
* **URGENT SELL**: Significant fundamental decay or extreme negative sentiment detected.

---

## Disclaimer

All output is for informational purposes. Probabilistic models and NLP sentiment analysis involve inherent risk. **Past performance does not guarantee future results.**
