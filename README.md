# Trade Republic Watchlist — Quant-AI Edition v4.5

A quantitative watchlist scanner that combines technical analysis, fundamental screening, macro correlation, geo-risk scoring, and Gemini AI news sentiment into a single ranked table. Outputs a scored CSV, a formatted Excel workbook, and an optional HTML email report.

---

## Features

- **AI sentiment** via Google Gemini — batch-scored from recent news headlines
- **Composite score (0–100)** split into Fundamental, Technical, and GeoSentiment buckets
- **Trade signals** — BUY / HOLD / SELL with ATR-based stop-loss
- **12-month backtest** — no lookahead bias, stop-hit detection
- **Macro correlations** — Pearson r vs oil, gold, S&P 500, US 10Y, TIPS inflation proxy
- **Geo-risk heuristic** — country base score + news keyword bump
- **Multi-currency support** — pence normalisation, optional USD→EUR conversion
- **Google News RSS fallback** when yfinance returns no headlines
- **HTML email report** with formatted tables and top-3 pick cards

---

## Project Structure

```
trade-republic-quant/
├── main.py          Entry point — three-phase pipeline
├── config.py        All settings (loaded from .env)
├── tickers.py       Watchlist, currency symbols, geo tables, macro tickers
├── indicators.py    EMA, RSI, ATR, safe_float — pure functions
├── utils.py         Display/formatting helpers
├── currency.py      EUR/USD rate fetching and price formatting
├── macro.py         Macro series download and correlation analysis
├── news.py          yfinance + Google News RSS fallback
├── gemini_ai.py     Gemini client, probe, and batch sentiment scoring
├── scoring.py       Geo risk, deep score (0–100), trade signal
├── backtest.py      12-month look-back backtest
├── analyzer.py      Per-asset analysis orchestrator
├── reporting.py     HTML email, SMTP sender, Excel export
├── outputs/
│   └── README.txt   Explains every generated file and column
├── .env.example     Secret template — copy to .env
├── .gitignore
└── requirements.txt
```

---

## Watchlist (45 instruments)

| Category | Instruments |
|---|---|
| Uranium / Nuclear | Cameco, NexGen Energy, Centrus Energy, BWX Technologies, URA ETF, URNM ETF |
| Energy & Industrials | Vistra, RWE, Siemens Energy, Constellation Energy, Equinor, Transocean, Technip Energies, Eni |
| Defence | BAE Systems, Rheinmetall |
| Cybersecurity | Crowdstrike, Palo Alto Networks |
| Agriculture / Chemicals | Yara International, Bayer, Ecolab |
| Water & Environment | Veolia, L&G Clean Water ETF |
| Materials / Mining | Albemarle, Zinnwald Lithium, Neo Performance Materials, Northern Star, Heidelberg Materials |
| Photonics / Optics | Coherent, Lumentum |
| Logistics & Shipping | Deutsche Post, Maersk A (A.P. Moller), Hapag-Lloyd |
| Real Estate / Finance | Vonovia, Raiffeisen Bank |
| Life Sciences | Sartorius Vz |
| Technology | Samsung |
| Broad ETFs | LIT, SSLN.L, IAUP.L, EIMI.L, IWDA.AS, SH |

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/your-username/trade-republic-quant.git
cd trade-republic-quant
pip install -r requirements.txt
```

### 2. Configure secrets

```bash
cp .env.example .env
```

Edit `.env`:

```
GEMINI_API_KEY=your_gemini_api_key_here
SMTP_USER=your_gmail@gmail.com
SMTP_PASSWORD=your_app_password
REPORT_TO=recipient@example.com
```

- **Gemini API key**: [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
- **Gmail App Password**: [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords) (required if 2FA is enabled)

### 3. Run

```bash
python main.py
```

---

## How the Score Works

```
Score (0–100)
├── Fundamental  (max 40)  — P/E < 20 → 15 pts, PEG < 1 → 15 pts, ROE > 20% → 10 pts
├── Technical    (max 30)  — RSI 30–60 → 15 pts, Upside > 25% → 15 pts
└── GeoSentiment (max 30)  — Geo score ≤ 3 → 15 pts, AI sentiment mapped to 0–15 pts
                              Volatility > 60% annualised → −3 pts penalty
```

**Trade signal logic:**

| Condition | Signal |
|---|---|
| Adj. score ≥ 65 AND upside > 5% | BUY |
| Adj. score ≥ 50 | HOLD |
| RSI > 72 OR upside < −10% | SELL |
| Otherwise | HOLD |

Adjusted score = score − 15 when AI sentiment < −50 (confirmed bad news).

---

## Output Files

All files are written to the `outputs/` directory.

| File | Description |
|---|---|
| `watchlist_v45.csv` | Full results table — all metrics |
| `watchlist_v45.xlsx` | Formatted Excel — conditional colouring, frozen header |
| `run_summary.txt` | Human-readable summary — top picks, backtest, column glossary |

---

## Backtest Methodology

The 12-month backtest applies the same entry rule that the scanner uses today, but evaluated on data from exactly 12 months ago:

- **Entry rule**: RSI(14) between 30–65 AND price ≥ EMA50 × 0.97 → BUY
- **Entry price**: closing price at the signal date
- **Exit price**: today's closing price
- **Stop-loss check**: if any daily low during the holding period fell below `entry − ATR × 2.5`, the stop is flagged
- **P&L**: `(exit − entry) / entry × 100`

No transaction costs, slippage, or taxes are modelled. Results are hypothetical.

---

## Adding or Removing Instruments

Edit `TICKER_MAP` in `tickers.py`. Yahoo Finance ticker symbols can be looked up at [finance.yahoo.com](https://finance.yahoo.com).

```python
TICKER_MAP = {
    "My New Stock": "XYZ",   # add a line
    ...
}
```

The rest of the pipeline adapts automatically.

---

## Disclaimer

All output is informational only and does not constitute financial advice. Past backtest performance does not guarantee future results.
