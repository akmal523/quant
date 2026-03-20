"""
gemini_ai.py — Gemini AI client initialisation, health probe, and batch
               sentiment scoring for watchlist tickers.

SDK:   google-genai  (pip install google-genai)
Model: configured via GEMINI_MODEL in config.py
"""
import json
import re
import time
import traceback as _tb

from config import GEMINI_API_KEY, GEMINI_MODEL

try:
    from google import genai as _genai
except ImportError:
    _genai = None  # type: ignore

# Single shared client instance — created on first call.
_gemini_client = None


def _init_gemini():
    """Return a cached genai.Client; creates it on first call."""
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    if not GEMINI_API_KEY:
        print("  [Gemini] GEMINI_API_KEY is not set (check your .env file).")
        return None
    if _genai is None:
        print("  [Gemini] google-genai not installed — run: pip install google-genai")
        return None
    try:
        _gemini_client = _genai.Client(api_key=GEMINI_API_KEY)
        print(f"  [Gemini] Client ready  (model: {GEMINI_MODEL})")
        return _gemini_client
    except Exception as e:
        print(f"  [Gemini] Client init error: {e}")
        return None


def probe_gemini() -> bool:
    """
    Send a trivial request to verify key + model before the main loop.
    Prints a detailed diagnostic on failure.
    Returns True on success, False otherwise.
    """
    client = _init_gemini()
    if client is None:
        print("  [Probe] Skipped — client not initialised.")
        return False
    print(f"  [Probe] Testing {GEMINI_MODEL} ...", end=" ", flush=True)
    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents="Reply with the single word: OK",
        )
        text = (resp.text or "").strip()
        print(f"OK  (response: '{text[:40]}')")
        return True
    except Exception as e:
        print("FAILED")
        msg = str(e)
        tb  = _tb.format_exc()
        print(f"\n  {'='*60}")
        print(f"  [DEBUG] Gemini probe error")
        print(f"  [DEBUG]   type    : {type(e).__name__}")
        print(f"  [DEBUG]   message : {msg}")
        for line in tb.strip().splitlines()[-6:]:
            print(f"  [DEBUG]   {line}")
        if "404" in msg or "not found" in msg.lower():
            print(f"\n  [DIAGNOSIS] Model '{GEMINI_MODEL}' not found.")
        elif "401" in msg or "403" in msg or "api_key" in msg.lower() or "invalid" in msg.lower():
            print(f"\n  [DIAGNOSIS] Auth failed — check GEMINI_API_KEY in .env")
        elif "429" in msg or "quota" in msg.lower() or "resource_exhausted" in msg.lower():
            print(f"\n  [DIAGNOSIS] Quota/rate limit — check billing at console.cloud.google.com")
        elif "500" in msg or "503" in msg:
            print(f"\n  [DIAGNOSIS] Server-side error — Gemini API may be temporarily down.")
        print(f"  {'='*60}\n")
        return False


def get_ai_sentiment_batch(items: list) -> dict:
    """
    Score news sentiment for up to BATCH_SIZE tickers in a single API call.

    Parameters
    ----------
    items : list of {"name": str, "sym": str, "news": list}

    Returns
    -------
    dict of {sym: (score: int, reason: str)}
        score is clamped to [-100, +100].
        Tickers with no headlines receive (0, "No News") without an API call.
        Exponential backoff on 429 (5 s, 10 s). Hard-stop on 404/401/403.
    """
    from config import BATCH_SIZE  # imported here to avoid circular at module level

    results: dict = {}
    client = _init_gemini()

    # Separate tickers that have usable headlines from those that don't.
    with_news, without_news = [], []
    for it in items:
        valid = [
            n for n in (it.get("news") or [])
            if n.get("title") and len(n["title"].strip()) > 5
        ]
        if valid:
            with_news.append({**it, "headlines": valid[:8]})
        else:
            without_news.append(it)

    for it in without_news:
        results[it["sym"]] = (0, "No News")

    if not with_news:
        return results

    if client is None:
        for it in with_news:
            results[it["sym"]] = (0, "Gemini client not initialised")
        return results

    # Build batch prompt.
    syms_in_prompt = []
    ticker_blocks  = []
    for it in with_news:
        lines = "\n".join(
            f"  {i+1}. {h['title'].strip()}"
            for i, h in enumerate(it["headlines"])
        )
        ticker_blocks.append(f'=== {it["sym"]} ({it["name"]}) ===\n{lines}')
        syms_in_prompt.append(it["sym"])

    batch_text = "\n\n".join(ticker_blocks)
    sym_list   = ", ".join(syms_in_prompt)

    prompt = f"""You are an aggressive financial analyst scoring stock news sentiment.

For each company below, assign a sentiment score and a one-line reason.
Score range: -100 (catastrophically bearish) to +100 (overwhelmingly bullish).
Rules:
- Be DECISIVE. Do not cluster around 0.
- Positive signals (earnings beats, contracts, buybacks) → +30 to +90
- Negative signals (lawsuits, losses, downgrades, macro risk) → -30 to -90
- Only give 0 if signals are genuinely balanced with no edge.

{batch_text}

Return ONLY valid JSON — no markdown fences, no extra text:
{{
  "TICKER1": {{"score": <int>, "reason": "<max 12 words>"}},
  "TICKER2": {{"score": <int>, "reason": "<max 12 words>"}},
  ...
}}
Use these exact ticker symbols: {sym_list}"""

    max_attempts = 3
    backoff_base = 5  # seconds

    for attempt in range(1, max_attempts + 1):
        try:
            response  = client.models.generate_content(
                model    = GEMINI_MODEL,
                contents = prompt,
            )
            raw       = (response.text or "").strip()
            raw_clean = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
            parsed    = json.loads(raw_clean)

            for sym in syms_in_prompt:
                entry = parsed.get(sym)
                if entry and isinstance(entry, dict):
                    score  = max(-100, min(100, int(entry.get("score", 0))))
                    reason = str(entry.get("reason", ""))[:120]
                    results[sym] = (score, reason)
                else:
                    results[sym] = (0, "Missing in response")
            return results

        except Exception as e:
            err_str  = str(e)
            err_type = type(e).__name__
            tb_lines = _tb.format_exc().strip().splitlines()

            print(f"\n  [DEBUG] Gemini batch error (attempt {attempt}/{max_attempts})")
            print(f"  [DEBUG]   type    : {err_type}")
            print(f"  [DEBUG]   message : {err_str}")
            for line in tb_lines[-5:]:
                print(f"  [DEBUG]   {line}")

            if "404" in err_str or "not found" in err_str.lower():
                print(f"  [DEBUG]   → Model '{GEMINI_MODEL}' not found.")
                for sym in syms_in_prompt:
                    if sym not in results:
                        results[sym] = (0, f"Model not found: {GEMINI_MODEL}")
                return results

            if "401" in err_str or "403" in err_str or "api_key" in err_str.lower():
                print(f"  [DEBUG]   → Auth error — check GEMINI_API_KEY in .env")
                for sym in syms_in_prompt:
                    if sym not in results:
                        results[sym] = (0, "Invalid API key")
                return results

            is_rate = (
                "429" in err_str
                or "quota" in err_str.lower()
                or "resource_exhausted" in err_str.lower()
            )
            if is_rate and attempt < max_attempts:
                wait = backoff_base * attempt  # 5 s, 10 s
                print(f"  [DEBUG]   → Rate limit. Retrying in {wait}s ...")
                time.sleep(wait)
                continue

            label = "Rate Limit" if is_rate else f"{err_type}: {err_str[:60]}"
            for sym in syms_in_prompt:
                if sym not in results:
                    results[sym] = (0, label)
            return results

    return results


def get_ai_sentiment(ticker_name: str, news: list) -> tuple[int, str]:
    """Single-ticker wrapper around get_ai_sentiment_batch."""
    result = get_ai_sentiment_batch(
        [{"name": ticker_name, "sym": ticker_name, "news": news}]
    )
    return result.get(ticker_name, (0, "No News"))
