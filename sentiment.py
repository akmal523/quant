"""
sentiment.py — Local FinBERT sentiment engine (ProsusAI/finbert).
Replaces gemini_ai.py. Fully deterministic; no external API calls.

Score semantics:
  +100  = strongly positive (all headlines bullish)
  0     = neutral
  -100  = strongly negative (all headlines bearish)
"""
from __future__ import annotations
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-initialised pipeline — loaded on first call to avoid startup cost
# when running tests or modules that don't need NLP.
_pipeline: Optional[object] = None
_load_attempted: bool = False


def _get_pipeline():
    """Return the transformers pipeline, or None if unavailable."""
    global _pipeline, _load_attempted
    if _load_attempted:
        return _pipeline
    _load_attempted = True
    try:
        import transformers
        import transformers.utils.logging as hf_logging
        # Suppress the harmless "bert.embeddings.position_ids UNEXPECTED" load
        # report that appears on newer transformers versions when loading FinBERT
        # via the pipeline API (key is present in checkpoint but absent in current
        # BertForSequenceClassification config — safe to ignore).
        hf_logging.set_verbosity_error()

        from transformers import pipeline as hf_pipeline
        from config import FINBERT_MODEL, FINBERT_DEVICE
        _pipeline = hf_pipeline(
            "text-classification",
            model=FINBERT_MODEL,
            tokenizer=FINBERT_MODEL,
            top_k=None,              # return probabilities for all three classes
            device=FINBERT_DEVICE,   # -1 = CPU
        )
        # Restore normal verbosity after load so other HF warnings still surface
        hf_logging.set_verbosity_warning()
        logger.info("[FinBERT] Model loaded: %s", FINBERT_MODEL)
    except Exception as exc:
        logger.warning("[FinBERT] Could not load model (%s). Sentiment scores will be 0.", exc)
        _pipeline = None
    return _pipeline


# FinBERT label → signed weight
_LABEL_WEIGHT: dict[str, float] = {
    "positive": +1.0,
    "negative": -1.0,
    "neutral":   0.0,
}


def _score_single(text: str) -> float:
    """
    Score one text string with FinBERT.
    Returns a value in [-100, +100].
    Truncates to 512 chars to stay within FinBERT's token limit.
    """
    pipe = _get_pipeline()
    if pipe is None or not text.strip():
        return 0.0
    try:
        result = pipe(text[:512], truncation=True)[0]
        # result: list of {"label": str, "score": float} — one per class
        signed = sum(
            _LABEL_WEIGHT.get(d["label"].lower(), 0.0) * d["score"]
            for d in result
        )
        return float(np.clip(signed * 100, -100, 100))
    except Exception as exc:
        logger.debug("[FinBERT] Scoring error: %s", exc)
        return 0.0


def analyze_news_context(headlines: list[dict]) -> dict:
    """
    Score a list of headline dicts {title, summary, published} with FinBERT.

    Returns:
        {
            "score":           float  in [-100, +100]   — mean signed score
            "drivers":         list[str]                 — top-3 most extreme headlines
            "headline_count":  int
        }
    """
    if not headlines:
        return {"score": 0.0, "drivers": [], "headline_count": 0}

    from config import FINBERT_MAX_HEADLINES
    texts = [
        (h.get("title", "") + ". " + h.get("summary", "")).strip()
        for h in headlines[:FINBERT_MAX_HEADLINES]
        if h.get("title", "").strip()
    ]

    if not texts:
        return {"score": 0.0, "drivers": [], "headline_count": 0}

    scores = [_score_single(t) for t in texts]
    mean_score = float(np.mean(scores))

    # Build driver strings: headline + score for the 3 most extreme items
    paired = sorted(
        zip(scores, [h.get("title", "") for h in headlines[:len(scores)]]),
        key=lambda x: abs(x[0]),
        reverse=True,
    )
    drivers = [
        f"{title[:70]} ({s:+.0f})"
        for s, title in paired[:3]
        if abs(s) >= 20   # only surface meaningful signals
    ]

    return {
        "score":          round(mean_score, 2),
        "drivers":        drivers,
        "headline_count": len(texts),
    }
