# CONTEXT.md — Domain Vocabulary & Architecture Map

Canonical terms for the Quant-AI family office terminal. Keep terminology
consistent across code, docs, and AI agents.

## Domain Vocabulary

| Term | Canonical Meaning |
|------|-------------------|
| **instrument_class** | Asset taxonomy: `EQUITY` \| `ETF` \| `COMMODITY` \| `CASH`. Drives bifurcated scoring. |
| **universe_status** | `ACTIVE` (heavy analysis) \| `WATCHLIST` (light scan) \| `CORE` (always tracked). |
| **Structural Grade** | Long-term fundamental quality (0-100). Stewardship + PE/PEG/ROE. |
| **Tactical Grade** | Short-term timing (0-100). HMM regime + sentiment - risk. |
| **Sparplan** | TR savings plan. 0 EUR buy, 1 EUR sell. Long-term accumulation. |
| **Active Trade** | TR tactical order. 1 EUR buy + 1 EUR sell = 2 EUR round-trip. |
| **Round-trip fee** | 2 EUR (active). The asymmetric hurdle rate for position sizing. |
| **Risk-free rate** | TR cash APY (2.25%), not US Treasury. Daily = `(1+APY)^(1/365)-1`. |
| **Safety Bucket** | Cash & short-term bonds. Constraint: ≥ 10%. |
| **Core Bucket** | Broad ETFs via Sparplan. Constraint: ≥ 40%. |
| **Alpha Bucket** | Active equities. Constraint: ≤ 50%. |
| **Graduation** | Watchlist → ACTIVE on 52-week-high or 3x-volume anomaly. |
| **Demotion** | ACTIVE → Watchlist after 6 months with no signals. |
| **universe_master** | Broad 1000+ ticker pool (S&P 500 + Nasdaq 100 + Russell 1000 + ETFs). Built by `universe_builder.py`. |
| **Funnel** | Two-stage filter: Stage 1 liquidity/viability (price>$5, min $ volume) → ~300-500; Stage 2 trend/momentum → top ~24 survivors. |
| **Broker-synced CSV** | `portfolio.csv` schema `Symbol,Avg_Entry_Price,Current_Value_EUR,Broker_PnL_EUR`. Invested = Value − Broker_PnL; PnL is broker truth, never price-guessed. |
| **Reconciliation** | `System_Estimated_Value = shares × current_price_eur`; `[!]` flag when deviation > €1.00 (stale CSV / high spread). |

## Module Map (Callers)

```
quant.data.universe_builder ─> quant.data.database (universe_master), quant.execution.taxonomy
quant.data.funnel ───────────> quant.config (thresholds), yfinance (snapshots/history)
quant.data.data_updater ─────> quant.data.universe_builder, quant.data.funnel, quant.execution.taxonomy, quant.data.database
quant.main ──────────────────> quant.data.funnel, quant.data.universe_builder, quant.analytics.scoring, quant.execution.taxonomy, quant.execution.routing, quant.reporting.notifier
quant.portfolio.optimizer ───> quant.portfolio.risk (daily rf), quant.config (buckets, fees)
quant.execution.discovery ───> quant.data.universe_builder, quant.execution.taxonomy, quant.data.database (asset_registry)
quant.dashboard ─────────────> quant.data.database, quant.execution.taxonomy, quant.execution.routing, quant.portfolio.risk
quant.reporting.notifier ────> quant.config, quant.portfolio.risk

> Package layout: modules live under `quant/<subpackage>/`. Entry points are the
> thin root wrappers `main.py` and `data_updater.py`. Filesystem paths resolve via
> `quant/paths.py` (CWD-independent).
```

## Data Contract (asset_registry)

| Column | Type | Purpose |
|--------|------|---------|
| symbol | VARCHAR PK | yahoo ticker |
| instrument_class | VARCHAR | EQUITY/ETF/COMMODITY/CASH |
| isin | VARCHAR | TR routing key |
| tr_ticker | VARCHAR | LS Exchange ticker |
| exchange | VARCHAR | LS Exchange / Tradegate |
| currency | VARCHAR | native currency |
| universe_status | VARCHAR | ACTIVE/WATCHLIST/CORE |
| graduated_at | DATE | promotion date |
| last_signal_date | DATE | last signal for demotion |

## ADR Notes

- **Cash as risk-free rate**: TR pays 2.25% APY on uninvested cash. Using this
  as R_f (instead of US Treasuries) is the correct opportunity cost for a
  EUR-based solo family office. Hard to reverse, real trade-off → documented.