"""
funnel.py — Multi-Stage Market Funnel (Plan 3, Phase 1).

Intent: filter the broad 1000+ universe_master down to the top ~24 survivors
for heavy FinBERT/GARCH analysis. Two cheap stages avoid fetching 5y history
for every ticker:
  Stage 1 (liquidity/viability): 1-day snapshot -> price>$5, min daily $ volume.
  Stage 2 (trend/momentum): 1y history on survivors -> rank -> top N.

State Transition: universe_master -> Stage1 survivors -> Stage2 top-N -> heavy
analysis (main.py) + portfolio merge.

Invariants:
  - Stage 1 returns a subset of the input symbols (never adds new ones).
  - Stage 2 returns at most top_n symbols.
  - Pure scoring functions (momentum_score) have no I/O.

Dependencies: config thresholds, yfinance (lazy), pandas, ThreadPoolExecutor.
"""
from __future__ import annotations

import contextlib
import os

import pandas as pd

from quant.cli.output import reporter

from quant.config import (
    FUNNEL_MIN_PRICE, FUNNEL_MIN_DAILY_VOLUME,
    FUNNEL_STAGE1_TARGET, FUNNEL_TOP_N, FUNNEL_MAX_WORKERS,
)


@contextlib.contextmanager
def _silence_yfinance():
    """Suppress yfinance's stderr noise (delisted-symbol warnings).

    Intent: the broad universe contains some delisted/bad tickers (e.g. BRK.B,
    BF.B from the Russell 1000 list). yfinance prints '$SYM: possibly delisted'
    to stderr for each one. These are non-fatal (the funnel skips them) but
    flood the console. Redirect stderr to devnull for the duration of a fetch.
    Invariants: restores stderr on exit, even on exception.
    """
    stderr_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(stderr_fd, 2)
        os.close(devnull)
        os.close(stderr_fd)


# ── Stage 1: Liquidity & Viability ────────────────────────────────────────────

def fetch_snapshot(symbol: str) -> tuple[str, float, float] | None:
    """Fetch a 1-day price/volume snapshot for a symbol.

    Intent: cheap Stage 1 probe. Only 1 day of data, so 1000+ tickers load fast.
    Invariants: returns (symbol, price, volume) or None on failure/no data.
    Dependencies: yfinance (lazy import keeps pure logic testable).
    """
    try:
        from quant.data.yf_utils import history_with_timeout
        with _silence_yfinance():
            df = history_with_timeout(symbol, period="1d", auto_adjust=True)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        price = float(df["Close"].iloc[-1])
        volume = float(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0.0
        return symbol, price, volume
    except Exception:
        return None


def _stage1_keep(price: float, volume: float,
                 min_price: float, min_daily_volume: float) -> bool:
    """Pure liquidity/viability rule: price >= min AND daily $ volume >= min."""
    return price >= min_price and (price * volume) >= min_daily_volume


def stage1_liquidity(
    symbols: list[str],
    min_price: float = FUNNEL_MIN_PRICE,
    min_daily_volume: float = FUNNEL_MIN_DAILY_VOLUME,
    max_workers: int = FUNNEL_MAX_WORKERS,
    stage1_target: int | None = None,
) -> list[str]:
    """Filter the broad universe by liquidity and viability.

    Intent: drop penny stocks and illiquid assets. Uses daily DOLLAR volume
    (price * shares) as the liquidity proxy. Downloads 1-day snapshots in
    *batches* (one yf.download request per N tickers) instead of N per-ticker
    requests. The 1000+ universe therefore costs ~20 requests, not 1084.
    When `stage1_target` is set, the most-liquid survivors are kept: Stage 2
    otherwise downloads 1y history for ~1000 tickers, which is slow and
    rate-limit prone.
    Invariants: returns a subset of `symbols`; never raises.
    """
    from quant.data.yf_utils import download_batch

    reporter.detail(f"  [FUNNEL] Stage 1: {len(symbols)} symbols (batched snapshots)...")
    frames = download_batch(symbols, period="1d")
    ranked: list[tuple[str, float]] = []
    for sym in symbols:
        df = frames.get(sym)
        if df is None:
            continue
        try:
            price = float(df["Close"].iloc[-1])
            volume = float(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0.0
        except Exception:
            continue
        if _stage1_keep(price, volume, min_price, min_daily_volume):
            ranked.append((sym, price * volume))
    if stage1_target and len(ranked) > stage1_target:
        ranked.sort(key=lambda kv: kv[1], reverse=True)
        ranked = ranked[:stage1_target]
    survivors = [s for s, _ in ranked]
    reporter.detail(f"  [FUNNEL] Stage 1: {len(survivors)} survivors.")
    return survivors


# ── Stage 2: Trend & Momentum ─────────────────────────────────────────────────

def fetch_history(symbol: str, period: str = "1y") -> pd.DataFrame | None:
    """Fetch 1y OHLCV history for a survivor.

    Intent: Stage 2 needs enough history for moving averages / RSI / MACD.
    Invariants: returns a df with a non-empty Close column, or None.
    Dependencies: yfinance (lazy).
    """
    try:
        from quant.data.yf_utils import history_with_timeout
        with _silence_yfinance():
            df = history_with_timeout(symbol, period=period, auto_adjust=True)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        return df.dropna(subset=["Close"])
    except Exception:
        return None


def momentum_score(close: pd.Series) -> float:
    """Composite trend/momentum score (0-100). Pure function, no I/O.

    Intent: rank Stage 2 survivors. Blends trend (price vs SMA), RSI(14)
    momentum, and 6-month return. Higher = stronger momentum.
    Invariants: returns a float in [0, 100]; short series score low.
    """
    close = close.dropna()
    if len(close) < 20:
        return 0.0
    last = float(close.iloc[-1])
    if last <= 0:
        return 0.0

    # Trend: price vs SMA (fall back to mean on short history).
    window = min(200, len(close))
    sma = float(close.rolling(window).mean().iloc[-1])
    trend = 50.0 + 50.0 * (1.0 if last > sma else -1.0)

    # RSI(14) momentum.
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
    if loss and loss > 0:
        rsi = 100.0 - 100.0 / (1.0 + gain / loss)
    else:
        rsi = 50.0

    # 6-month return, mapped to a 0-100 scale around 50.
    ret_6m = (last / float(close.iloc[-126]) - 1.0) * 100.0 if len(close) >= 126 else 0.0
    ret_score = max(0.0, min(100.0, 50.0 + ret_6m))

    return 0.4 * trend + 0.3 * rsi + 0.3 * ret_score


def stage2_momentum(
    symbols: list[str],
    top_n: int = FUNNEL_TOP_N,
    max_workers: int = FUNNEL_MAX_WORKERS,
) -> list[str]:
    """Fetch 1y history for survivors (batched), rank by momentum, return top_n.

    Intent: the final cheap filter before heavy analysis. Batch-downloads 1y
    history so the request count stays far below the per-ticker path.
    Invariants: returns at most top_n symbols, ordered best-first.
    """
    from quant.data.yf_utils import download_batch

    reporter.detail(f"  [FUNNEL] Stage 2: {len(symbols)} symbols (batched 1y history)...")
    frames = download_batch(symbols, period="1y")
    scores: dict[str, float] = {}
    for sym, df in frames.items():
        if "Close" in df.columns:
            scores[sym] = momentum_score(df["Close"])
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top = [s for s, _ in ranked[:top_n]]
    reporter.detail(f"  [FUNNEL] Stage 2: {len(scores)} scored, top {len(top)} kept.")
    return top


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_funnel(
    symbols: list[str],
    stage1_target: int = FUNNEL_STAGE1_TARGET,
    top_n: int = FUNNEL_TOP_N,
) -> dict:
    """Run the full two-stage funnel over a broad universe.

    Intent: single entry point. Stage 1 liquidity -> Stage 2 momentum -> top_n.
    Returns a summary dict with stage counts and the final survivor list.
    Invariants: returns dict with 'survivors' (list) and stage counts.
    """
    stage1 = stage1_liquidity(symbols, stage1_target=stage1_target)
    # Stage 1 now enforces the soft cap by keeping the most-liquid names, so
    # Stage 2 only downloads 1y history for ~stage1_target survivors.
    survivors = stage2_momentum(stage1, top_n=top_n)
    return {
        "input": len(symbols),
        "stage1": len(stage1),
        "stage2": len(survivors),
        "survivors": survivors,
    }


# ── Survivor persistence (Plan 3, Phase 1) ────────────────────────────────────

def save_survivors(survivors: list[str], scores: dict[str, float] | None = None) -> int:
    """Persist the funnel survivor list to DuckDB (funnel_survivors table).

    Intent: the funnel is the one expensive, network-bound step. data_updater.py
    runs it once per cycle and saves the result here so main.py can read the
    survivors without re-fetching the 1000+ universe (the documented 2-step
    flow: data_updater -> main).
    Invariants: replaces the previous survivor set; best-effort (never raises).
    """
    import time
    try:
        from quant.data.database import get_connection, init_db
        init_db()
        conn = get_connection()
        conn.execute("DELETE FROM funnel_survivors")
        for s in survivors:
            conn.execute(
                "INSERT INTO funnel_survivors (symbol, score, updated_at) VALUES (?, ?, ?)",
                [s, (scores or {}).get(s), time.time()],
            )
        return len(survivors)
    except Exception:
        return 0


def load_survivors() -> list[str]:
    """Read the cached funnel survivors saved by data_updater.py.

    Invariants: returns an empty list if the table is missing/empty (caller
    should then warn to run data_updater.py first).
    """
    try:
        from quant.data.database import get_connection
        conn = get_connection()
        rows = conn.execute(
            "SELECT symbol FROM funnel_survivors ORDER BY score DESC NULLS LAST"
        ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


if __name__ == "__main__":
    from quant.data.universe_builder import load_universe_master
    pool = load_universe_master()
    print(f"Funnel input: {len(pool)} symbols from universe_master")
    result = run_funnel(pool)
    print(f"Stage1 survivors: {result['stage1']} | Stage2 top-N: {result['stage2']}")
    print("Survivors:", result["survivors"])