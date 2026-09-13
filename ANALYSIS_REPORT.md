# Quant-AI v8.5 — Comprehensive Code Analysis & Optimization Report

> **Historical document.** This review targets the v8.5 codebase and is retained
> for reference only. Line numbers and findings are superseded by the current
> v10.4.0 code. See [`CHANGELOG.md`](CHANGELOG.md) and [`CONTEXT.md`](CONTEXT.md)
> for the current state.

> Based on deep review of all 20+ source files, tests, config, and data files.

---

## 🔴 Critical Bugs (Will Cause Incorrect Results or Crashes)

### 1. Duplicate Data Loading in `main.py` (LINES 148–171)
[`main.py`](main.py:148) loads market data **twice**, destroying the sorted/cleaned DataFrame from the first load:

```python
# First load — correctly sorted by Date
market_data = conn.execute("SELECT * FROM market_history ORDER BY Date ASC").df()
market_data['Date'] = pd.to_datetime(market_data['Date'])
market_data = market_data.sort_values(['Symbol', 'Date'])
grouped_data = {symbol: df for symbol, df in market_data.groupby("Symbol")}

# Second load — UNSORTED, overwrites grouped_data from line 163!
market_data = conn.execute("SELECT * FROM market_history").df()   # <-- NO ORDER BY
grouped_data = {symbol: df for symbol, df in market_data.groupby("Symbol")}
```

**Impact**: All HMM, GARCH, and indicator calculations receive **chronologically scrambled** data, producing garbage scores.

**Fix**: Remove the second duplicate block.

---

### 2. Double Scaling of HMM Probability (LINES 80 + 167 OF main.py/scoring.py)
[`main.py:80`](main.py:80):
```python
hmm_prob_bull = hmm_market_state_score(hist_ind["Close"], garch_vol) / 15.0
```

[`scoring.py:171`](scoring.py:171):
```python
grade = hmm_prob_bull * 60.0  # Re-multiplies the already-divided value!
```

**Math**: `hmm_market_state_score` returns `[0, 15]`. Then divided by 15 → `[0, 1]`. Then multiplied by 60 → `[0, 60]`. This is **correct by accident** (15 * 60 / 15 = 60), but the `/15.0` factor is semantically confusing and fragile. If `max_points` in `hmm_market_state_score` ever changes, the downstream math breaks silently.

**Fix**: Remove the `/ 15.0` in `main.py` and let `evaluate_tactical_grade` handle the scaling consistently.

---

### 3. `backtest.py` Imports Non-Existent Functions (LINE 26)
[`backtest.py:26`](backtest.py:26):
```python
from indicators import rsi as calc_rsi, atr
```

Neither `rsi()` nor `atr()` exist in [`indicators.py`](indicators.py). The file only exports `garch_volatility()` and `add_all_indicators()`. **Running backtest.py will crash with ImportError**.

---

### 4. DuckDB Concurrent Write Violation in ProcessPoolExecutor (LINE 227)
[`main.py:227`](main.py:227):
```python
conn.execute("INSERT OR REPLACE INTO nlp_scores (doc_hash, score) VALUES (?, ?)", [doc_hash, nlp_score])
```

DuckDB supports **exactly one writer process**. The `ProcessPoolExecutor` spawns multiple processes sharing the same DuckDB file. Even though this write happens in the main process after `future.result()`, the concurrent readers in worker processes can cause WAL locking conflicts.

**Fix**: Use a dedicated single-writer queue or batch writes after all workers complete.

---

### 5. `init_db()` Never Called in Main Pipeline (database.py)
[`database.py:16`](database.py:16):
```python
def init_db() -> None:
    # Creates fundamentals + nlp_scores tables
```

[`main.py`](main.py) calls `get_connection()` but never calls `init_db()`. If the tables don't exist, the `INSERT OR REPLACE INTO nlp_scores` on line 227 will raise `CatalogException`.

---

### 6. `test_cache.py` References Non-Existent Functions
[`test_cache.py:3`](test_cache.py:3):
```python
from sentiment import _save_cached_score, _get_cached_score
```

Neither function exists in [`sentiment.py`](sentiment.py). The NLP cache uses DuckDB directly from `main.py`. This test will always fail with ImportError.

---

## 🟠 High-Impact Issues (Performance, Correctness, Maintainability)

### 7. Redundant Constants in `config.py`
[`config.py`](config.py) has **duplicated threshold definitions**:

| Line 72–94 | Line 96–114 | Status |
|---|---|---|
| `MIN_ROE = 0.20` | — | Used only in `scoring.apply_fast_filter` via `FILTER_MIN_ROE` |
| `MAX_PE = 25.0` | — | Same as `FILTER_MAX_PE` |
| `FILTER_MAX_PE = 25.0`, `FILTER_MIN_ROE = 0.20` | — | Duplicates `MAX_PE`/`MIN_ROE` |
| `GEN_MAX_DE = 0.5` | `STW_GEN_MAX_DE = 0.5` | Exact duplicate |
| `GEN_MIN_ICR = 5.0` | `STW_GEN_MIN_ICR = 5.0` | Exact duplicate |
| `STRUCT_BUY_LIMIT = 75` | `MIN_STRUCT_GRADE_FOR_BUY = 75` | Exact duplicate |
| `TACTICAL_BUY_LIMIT = 70` | `MIN_TACT_GRADE_FOR_BUY = 70` | Exact duplicate |

The first set (lines 72–94) is **completely redundant** since the second set (lines 96–114) defines identical values. Remove the first set to eliminate the possibility of drift.

---

### 8. Sequential Data Fetching in `data_updater.py` (~300 Tickers × 0.5s = 150s+)
[`data_updater.py:20`](data_updater.py:20) fetches 300+ tickers sequentially with `time.sleep(0.5)` between each. At ~0.5s per ticker (API latency), the full universe takes **2.5+ minutes**.

**Fix**: Use `ThreadPoolExecutor` with rate-limited async fetching to reduce to ~15–20 seconds.

---

### 9. `currency.py` Raises Fatal Error on FX Failure (LINE 49)
[`currency.py:49`](currency.py:49):
```python
raise RuntimeError("CRITICAL: Could not fetch real EUR/USD exchange rate... Halting to prevent bad math.")
```

If the ECB API + Yahoo both fail (e.g., VPN, rate limit, weekend), **the entire scan crashes**. A cached rate from the previous successful run would be safer, or just fall back to 1.0 with a warning.

---

### 10. GARCH(1,1) as Sole Volatility Model (indicators.py)
[`indicators.py`](indicators.py) only implements GARCH(1,1). For assets with < 252 data points (common for newly listed stocks or IPOs), it returns `None`, producing NaN in GARCH_Vol which then propagates through HMM scoring.

**Fix**: Add fallback: EWMA (exponentially weighted moving average) volatility for short histories, then GARCH for long histories.

---

### 11. FinBERT Chunking Uses Raw Token IDs (sentiment.py, LINE 37)
[`sentiment.py:37`](sentiment.py:37):
```python
input_ids = torch.tensor([[tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]])
```

This bypasses `tokenizer()` attention masks and token type IDs. For long sequences, the model may produce degraded results without proper attention masking.

---

### 12. `evaluate_structural_grade` Reduces Stewardship by 1.5× Multiplier (scoring.py, LINE 137)
[`scoring.py:137`](scoring.py:137):
```python
grade = stewardship_val * 1.5
```

`stewardship_val` is capped at `WEIGHT_STEWARDSHIP = 30`. After 1.5× multiplier, max = 45. Then PE/PEG/ROE bonuses are added (max ~65 + 45 = 110, clipped to 100). The grading logic is extremely sensitive to this 1.5× scaling factor — any change to `WEIGHT_STEWARDSHIP` in config silently changes structural grade behavior.

---

### 13. `portfolio.py` Overcounts Skipped Positions (LINE 138)
[`portfolio.py:138`](portfolio.py:138):
```python
if invested <= 0:
    position_count += 1
    continue
```

`position_count` is incremented but `active_count` is not. Later `active_count` is reported as "with data" which is correct, but `position_count` includes positions with zero investment, making it misleading when displayed alongside "with data."

---

## 🟡 Medium-Priority Improvements

### 14. No Type Annotations on ~40% of Functions
Functions like [`currency.apply_fx_conversion`](currency.py:56), [`scoring.stewardship_score_v2`](scoring.py:63), and [`risk.calculate_risk_penalty`](risk.py:53) are missing return type hints. Adding `-> pd.DataFrame` / `-> float` would prevent subtle bugs and enable static analysis.

### 15. `spawn` Context Underutilized for GPU Memory
[`main.py:208`](main.py:208):
```python
ctx = multiprocessing.get_context("spawn")
```

Good practice. But the `ProcessPoolExecutor` loads FinBERT in every worker process via `init_worker`. Each worker consumes ~800MB RAM for the model. With `cpu_count - 1 = 15` workers, that's **12GB+ RAM** for model copies alone.

**Fix**: Use a shared-memory IPC approach (multiprocessing.Array or Redis Queue) with a single GPU inference server. Or reduce `max_workers` to 4.

### 16. No Logging Configuration in `main()` for Workers
Worker processes in `ProcessPoolExecutor` don't inherit logging config. Any logging calls inside `process_asset` are silently swallowed. The code uses `print()` as a workaround, which is fragile.

### 17. `sentiment.py` Global State Not Thread-Safe
[`sentiment.py:6-7`](sentiment.py:6-7):
```python
tokenizer = None
model = None
```

These globals are set once in `init_worker()`. If a worker process somehow re-enters `init_worker`, the `from_pretrained()` calls will re-download the model. Add a guard:

```python
if tokenizer is None:
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
```

### 18. `requirements.txt` Includes Spam/Unused Packages
[`requirements.txt:13-14`](requirements.txt:13-14):
```
antiorm>=1.2.1 
db>=0.1.1
```

`antiorm` and `db` are not used anywhere in the codebase. `spacy` (line 11) is also unused — the NER config references `en_core_web_sm` but no code actually uses it. These should be removed to reduce `pip install` surface area and potential dependency conflicts.

### 19. Tests Have Zero Assertions on Quant Logic
- [`test_market_data.py`](test_market_data.py) validates DuckDB I/O but not actual market data structure
- [`test_db.py`](test_db.py) tests basic insert/select — not schema constraints or concurrent access
- No tests for [`scoring.py`](scoring.py) HMM, stewardship, or capital allocation logic
- No tests for [`risk.py`](risk.py) VaR or Sortino calculations
- No tests for [`indicators.py`](indicators.py) GARCH convergence

### 20. `MAERSK-A.CO` Uses Danish Krone (DKK) But Missing from `deduce_currency`
[`universe.py:232`](universe.py:232):
```python
"AP Moller-Maersk": "MAERSK-A.CO",
```

`.CO` suffix is Danish (Copenhagen), should map to `DKK` in [`main.py:52`](main.py:52). Currently only `OL` (Oslo) → `NOK` and `ST` (Stockholm) → `SEK` are handled. Add `.CO` → `DKK`.

### 21. `process_asset` Uses Bare Except (LINE 143)
[`main.py:143`](main.py:143):
```python
except Exception as e:
    print(f"\n[FATAL WORKER CRASH] {symbol}: {str(e)}")
    return None
```

Catches `SystemExit`, `KeyboardInterrupt` which could hang the pool. Use `except Exception` properly.

---

## 🏆 Recommended Changes (Priority-Ordered)

| # | File | Change | Category |
|---|------|--------|----------|
| 1 | [`main.py:148-171`](main.py:148) | Remove duplicate data loading block | **CRITICAL** |
| 2 | [`main.py:80`](main.py:80) | Remove `/ 15.0` factor, update `evaluate_tactical_grade` | **CRITICAL** |
| 3 | [`backtest.py:26`](backtest.py:26) | Implement `rsi()` and `atr()` in `indicators.py` | **CRITICAL** |
| 4 | [`database.py`](database.py) | Call `init_db()` at startup in `main.py` | **BUG** |
| 5 | [`config.py`](config.py) | Deduplicate threshold constants | **MAINTENANCE** |
| 6 | [`data_updater.py`](data_updater.py) | Parallelize with ThreadPoolExecutor | **PERFORMANCE** |
| 7 | [`currency.py:49`](currency.py:49) | Fallback instead of crash | **RESILIENCE** |
| 8 | [`indicators.py`](indicators.py) | Add EWMA volatility fallback | **ROBUSTNESS** |
| 9 | [`sentiment.py`](sentiment.py) | Add attention masks, guard global init | **CORRECTNESS** |
| 10 | [`main.py:207`](main.py:207) | Cap `max_workers` at 4 (RAM guard) | **PERFORMANCE** |
| 11 | [`test_cache.py`](test_cache.py) | Fix references to non-existent functions | **TESTING** |
| 12 | [`requirements.txt`](requirements.txt) | Remove `antiorm`, `db`, `spacy` | **MAINTENANCE** |
| 13 | [`universe.py:232`](universe.py:232) | Add `.CO` → `DKK` mapping | **CORRECTNESS** |
| 14 | [`main.py:143`](main.py:143) | Use `except Exception` (not bare) | **ROBUSTNESS** |

---

## 📈 Quantitative Accuracy Summary

| Metric | Current Status | Should Be |
|--------|---------------|-----------|
| Price chronology in HMM/GARCH | **Broken** (duplicate load scrambles sort) | Correct chronological order |
| HMM probability scale | Works by coincidence (15×60/15=60) | Explicit, unit-tested scale |
| FX conversion from DKK | **Missing** (MAERSK-A.CO fails) | Correct DKK→EUR |
| Backtest RSI/ATR | **ImportError** (cannot run) | Fully functional |
| NLP cache persistence | Works if `init_db()` was called | Guaranteed table creation |
| Thread safety in scoring | Fragile (global state + multiprocessing) | Explicit process isolation |