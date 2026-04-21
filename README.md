# Quant-AI v6 — Stewardship & Resilience Engine

A professional-grade Python pipeline for systematic equity analysis. Scans a ~300-asset, 20-sector universe using a hierarchical data architecture, scores assets based on intrinsic financial stewardship, and audits portfolios with volatility-adjusted technicals.

**No cloud dependencies.** Sentiment analysis runs locally via [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert) using batch inference and temporal decay.

---

## New in v6

* **Hierarchical Data Resilience** — Integrated SQLite caching and multi-source fetching to eliminate `yfinance` data gaps.
* **Stewardship-Centric Scoring** — Trade signals now require a "Quality Floor" (Debt-to-Equity and Payout sustainability).
* **Z-Score Technicals** — Normalises price action relative to an asset's own historical volatility (Standard Deviations) rather than static percentages.
* **Temporal Sentiment Decay** — Headlines are weighted by age (24h half-life) and model confidence, processed in high-efficiency batches.
* **Windowed Backtesting** — Scans a 30-day historical window for the first valid entry, simulating real-world execution.

---

## Features

* **Dynamic sector universe** — ~300 instruments across 20 strategic sectors (Uranium, Semiconductors, AI & Cloud, Defense, Precious Metals, and more)
* **Local FinBERT NLP** — deterministic sentiment scores in `[−100, +100]` mapped directly to mathematical weights; no API key, no rate limits
* **EUR-normalised pricing** — full FX conversion chain (GBX→GBP→EUR, USD→EUR, cross-rate fallback) applied before any indicator calculation
* **Composite score (0–100)** across three buckets: Fundamental (PE/PEG/ROE) + Technical (RSI/SMA50) + GeoSentiment (geo-risk heuristic + FinBERT)
* **Sector-relative valuation** — sector median PE bonus for assets trading at a discount to peers
* **Investment horizon classification** — every asset assigned to RETIREMENT (20yr+), BUSINESS COLLATERAL (10yr), or SPECULATIVE (short-term)
* **12-month backtest** — no lookahead bias, ATR-based stop-hit detection
* **Portfolio audit** — reads `portfolio.csv`, issues URGENT SELL / BUY MORE (DCA OK) / HOLD per holding
* **Ranked output** — Global Top 3 + Top 3 per Sector in terminal; `.csv` and conditional-colour `.xlsx` in `/outputs`
* **Resilient news fetching** — Google News RSS primary, 3-retry exponential backoff, 5 User-Agent rotation, yfinance fallback

---

## Architecture

```
quant/
├── main.py                # Pipeline orchestration (Scan → Score → Audit)
├── fundamentals.py        # Hierarchical service (Cache → Primary API → Fallback)
├── scoring.py             # Stewardship, Z-score Technicals, & Quality-floor logic
├── sentiment.py           # Batched FinBERT engine with temporal decay
├── backtest.py            # 30-day windowed historical simulation
├── universe.py            # Expanded 300-asset / 20-sector definition
├── indicators.py          # Vectorised technical primitives
├── currency.py            # Multi-leg FX normalization (Base: EUR)
├── portfolio.py           # Quality-aware audit decision engine
├── fundamentals_cache.sqlite  # Auto-generated local data persistence
└── outputs/               # Conditional-formatted reports (.xlsx, .csv)
```

**Removed from v4.5:** `gemini_ai.py`, `tickers.py`, `analyzer.py`, `macro.py`, `utils.py`

---

## Sector Universe (~300 instruments)

| Sector | Example Instruments |
| --- | --- |
| **Uranium / Nuclear** | Cameco, NexGen Energy, Centrus Energy, BWX Technologies, URA ETF |
| **Energy (Utilities/Renewables)** | Vistra, Constellation Energy, NextEra Energy |
| **Oil & Gas** | ExxonMobil, Chevron, Shell |
| **Defense** | BAE Systems, Rheinmetall, Northrop Grumman, L3Harris |
| **Cybersecurity** | CrowdStrike, Palo Alto Networks, SentinelOne, Fortinet |
| **Gold / Precious Metals** | Newmont, Barrick Gold, Agnico Eagle |
| **Silver & Royalties** | Pan American Silver, Wheaton Precious Metals |
| **Copper & Battery Metals** | Freeport-McMoRan, Southern Copper |
| **Lithium** | Albemarle, SQM |
| **Quantum Computing** | IBM, IonQ, Rigetti Computing, D-Wave |
| **Semiconductors** | Nvidia, AMD, ASML, TSMC |
| **AI & Cloud** | Microsoft, Alphabet, Amazon |
| **Logistics / Shipping** | Deutsche Post, AP Moller-Maersk, Hapag-Lloyd, FedEx |
| **Finance & Banking** | JPMorgan, Goldman Sachs, HSBC |
| **Insurance** | Allianz, Munich Re, Axa |
| **Healthcare / Pharma** | Eli Lilly, Novo Nordisk, Johnson & Johnson |
| **Water & Environment** | Veolia, Xylem, L&G Clean Water ETF |
| **Agriculture / Chemicals** | Yara International, Bayer, Ecolab |
| **Real Estate (REIT)** | Prologis, American Tower |
| **Broad ETFs** | MSCI World (IWDA), EM IMI (EIMI), LIT, Short S&P500 (SH) |

To add or remove instruments, edit `SECTOR_UNIVERSE` in `universe.py`. The rest of the pipeline adapts automatically.

---

## Scoring Model (v6)

The composite score (0–100) prioritizes structural health and fiscal stewardship over transient momentum.

| Category | Weight | Logic |
| :--- | :--- | :--- |
| **Fundamentals** | 30 pts | PE < 20 (10), PEG < 1.2 (10), ROE > 15% (10) |
| **Stewardship** | 20 pts | Debt-to-Equity < 0.5 (10), Sustainable Payout 30-70% (10) |
| **Technicals** | 25 pts | Price Z-Score -0.5 to +1.5 (15), Sector-Relative RSI (10) |
| **GeoSentiment** | 25 pts | Batched FinBERT weighted by temporal decay and confidence |

### Trade Signal Logic

Signals are governed by a **Stewardship Floor**. If an asset fails solvency checks (Score < 5/20), "BUY" signals are suppressed to prevent speculation on fragile balance sheets.

| Condition | Signal |
| :--- | :--- |
| **Adj. Score ≥ 70 AND Upside > 5%** | **BUY** |
| **Adj. Score ≥ 50** | **HOLD** |
| **Stewardship < 5 OR RSI > Sell Threshold** | **SELL** |
| **Otherwise** | **HOLD** |

---

## Investment Horizon Classification

Assets are categorized into three buckets based on their structural "DNA" and suitability for long-term capital preservation.

| Category | Criteria |
| :--- | :--- |
| **RETIREMENT (20yr+)** | Div ≥ 2%, Vol < 25%, ROE > 10%, 0 < PE < 30 |
| **BUSINESS COLLATERAL (10yr)** | Vol < 35%, ROE > 5%, 0 < PE < 40 (Lombard-eligible) |
| **SPECULATIVE (short-term)** | All other assets; high momentum or news-sensitive |

---

## Backtest Methodology (v6)

The simulation engine implements a **30-day Entry Window** to replicate real-world watchlist monitoring rather than naive point-in-time checks.

* **Scan Window:** T-260 to T-230 trading days. Identifies the *first* valid entry.
* **Entry Rule:** RSI(14) in [30, 65] AND Price ≥ 0.97 × EMA50.
* **Path Validation:** Daily `Low` prices are audited against the stop-loss level throughout the holding period.
* **Stop-Loss:** `entry_price − (ATR(14) × 2.5)`.
* **Friction:** Applies a 0.15% round-trip commission and slippage penalty to net P&L.
* **Exit:** Fixed at current market close or stop-loss breach.


## Portfolio Audit Engine

The audit engine cross-references your personal holdings against the live market scan to issue actionable stewardship decisions. It prioritizes capital preservation by enforcing a "Quality Floor" on existing positions.

### 1. Setup Your Holdings
Create a `portfolio.csv` in the root directory with the following structure:

```csv
Symbol,Buy_Price,Amount_EUR
CCJ,35.50,2500
RHM.DE,420.00,5000
NVDA,115.00,3000
```

2. Decision Hierarchy

The engine applies a strict logical chain to determine if a position remains a "faithful" allocation of capital:

| Decision | Trigger |
| :--- | :--- |
| **URGENT SELL** | Signal is SELL OR RSI > 80 (Exhaustion) OR Stewardship < 5 with negative PnL |
| **BUY MORE (DCA OK)** | Signal is BUY AND Unrealised PnL < 15% (Prevents chasing peaks) |
| **HOLD** | Score remains stable; intrinsic quality intact |
| **NOT SCANNED** | Asset is not present in the current 300-instrument universe |
---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/akmal523/quant.git
cd quant
pip install -r requirements.txt
```

Dependencies: `yfinance`, `pandas`, `numpy`, `torch`, `transformers`, `openpyxl`, `python-dotenv`

### 2. FinBERT (first run only)

The model (~500 MB) downloads automatically from HuggingFace on first execution. To pre-cache it:

```bash
python -c "from transformers import pipeline; pipeline('text-classification', model='ProsusAI/finbert')"
```

To run on GPU, set `FINBERT_DEVICE=0` in your `.env`.

### 3. Configure (optional)

```bash
cp .env.example .env
```

```env
# No API keys required for core functionality.
# Optional: email delivery of the HTML report.
SMTP_USER=your_gmail@gmail.com
SMTP_PASSWORD=your_app_password
REPORT_TO=recipient@example.com
```

### 4. Run

```bash
python main.py
```

Output is written to `outputs/market_scan.csv` and `outputs/market_scan.xlsx`. If `portfolio.csv` exists, `outputs/portfolio_audit.csv` is also generated.

---

## Output Files

| File | Description |
|---|---|
| `outputs/market_scan.csv` | Full results — all metrics, scores, signals, horizons |
| `outputs/market_scan.xlsx` | Conditional-colour workbook — green/amber/red by signal and horizon |
| `outputs/portfolio_audit.csv` | Per-holding audit decisions with reasoning strings |

---

## Disclaimer

All output is for informational purposes only and does not constitute financial advice. Past backtest performance does not guarantee future results.
