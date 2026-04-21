# Quant-AI v5 — Sector Scanner & Portfolio Audit Engine

A modular Python pipeline for systematic equity analysis. Scans a 65-asset, 14-sector universe, scores every asset on a deterministic 0–100 composite model, classifies holdings by investment horizon, and audits a live portfolio against current market state.

**No cloud LLM dependencies.** Sentiment analysis runs locally via [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert).

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
├── main.py          Three-phase pipeline: Scan → Score → Report
├── config.py        Runtime settings (.env override)
├── universe.py      Sector universe, geo-risk tables, FX symbol map
├── indicators.py    RSI, ATR, EMA, SMA — pure functions, no side effects
├── currency.py      FX rate fetching, OHLCV normalisation to EUR
├── news.py          RSS fetcher with UA rotation and yfinance fallback
├── sentiment.py     Local FinBERT engine (ProsusAI/finbert)
├── scoring.py       Composite score, horizon classifier, trade signal
├── backtest.py      12-month lookback backtest
├── portfolio.py     Portfolio audit decision engine
├── reporting.py     Terminal output, Excel workbook, CSV export
├── portfolio.csv    Your holdings (Symbol, Buy_Price, Amount_EUR)
├── requirements.txt
├── .env.example
└── outputs/         Generated reports written here
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

## Scoring Model

```
Composite Score (0–100)
├── Fundamental     max 40   PE < 20 → 15 pts │ PEG < 1 → 15 pts │ ROE > 20% → 10 pts
│                            + sector-relative bonus: asset PE ≤ 80% of sector median → +5
├── Technical       max 30   RSI 30–60 → 15 pts │ Price in 0.97–1.15× SMA50 → 15 pts
└── GeoSentiment    max 30   Geo score ≤ 3 → 15 pts │ FinBERT score mapped [−100,+100] → 0–15 pts
                             Annualised volatility > 60% → −3 pts penalty
```

**Trade signal:**

| Condition | Signal |
|---|---|
| Adj. score ≥ 65 AND upside > 5% | BUY |
| Adj. score ≥ 50 | HOLD |
| RSI > 72 OR upside < −10% | SELL |
| Otherwise | HOLD |

Adjusted score = score − 15 when FinBERT score < −50 (confirmed bearish news flow).

---

## Investment Horizon Classification

| Horizon | Criteria |
|---|---|
| **RETIREMENT (20yr+)** | Dividend ≥ 2%, Volatility < 25%, ROE > 10%, 0 < PE < 30 |
| **BUSINESS COLLATERAL (10yr)** | Volatility < 35%, ROE > 5%, 0 < PE < 40 — suitable as Lombard / corporate loan collateral |
| **SPECULATIVE (short-term)** | Everything else — momentum-driven, news-sensitive |

---

## Portfolio Audit

Place your holdings in `portfolio.csv`:

```csv
Symbol,Buy_Price,Amount_EUR
CCJ,35.50,2500
RHM.DE,420.00,5000
CRWD,220.00,3000
```

The audit engine cross-references each position against the live scan and issues:

| Decision | Trigger |
|---|---|
| **URGENT SELL** | Signal == SELL, or FinBERT score < −60, or RSI > 75 |
| **BUY MORE (DCA OK)** | Signal == BUY and unrealised PnL < 20% |
| **HOLD** | All other cases |
| **NOT SCANNED** | Symbol not found in the scanned universe |

---

## Backtest Methodology

The 12-month backtest applies the entry rule on data from exactly 252 trading days ago:

- **Entry rule:** RSI(14) in [30, 65] AND Close ≥ EMA50 × 0.97
- **Entry price:** closing price at the signal date
- **Exit price:** today's closing price
- **Stop-loss:** `entry_price − ATR(14) × 2.5`; flagged if any daily low breached it during the holding period
- **P&L:** `(exit − entry) / entry × 100`

No transaction costs, slippage, or taxes are modelled. Results are hypothetical.

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
