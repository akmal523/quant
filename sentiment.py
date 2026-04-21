"""
sentiment.py — Local FinBERT sentiment engine (ProsusAI/finbert).
Batched inference with temporal decay.
"""
from __future__ import annotations
import logging
import time
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

# Lazy-initialised pipeline to avoid startup cost
_pipeline: Optional[object] = None
_load_attempted: bool = False

def _get_pipeline():
    """Return the transformers pipeline, or None if unavailable."""
    global _pipeline, _load_attempted
    if _load_attempted:
        return _pipeline
    _load_attempted = True
    try:
        import transformers.utils.logging as hf_logging
        hf_logging.set_verbosity_error()
        from transformers import pipeline as hf_pipeline
        from config import FINBERT_MODEL, FINBERT_DEVICE
        
        logger.info("Loading FinBERT model into memory...")
        _pipeline = hf_pipeline(
            "text-classification",
            model=FINBERT_MODEL,
            tokenizer=FINBERT_MODEL,
            device=FINBERT_DEVICE
        )
    except Exception as exc:
        logger.error("[FinBERT] Failed to load pipeline: %s", exc)
    return _pipeline

def _calculate_time_weight(published_str: str) -> float:
    """Applies an exponential decay weight based on headline age (24h half-life)."""
    try:
        pub_time = float(published_str)
        age_hours = (time.time() - pub_time) / 3600
        return 0.5 ** (age_hours / 24.0)
    except Exception:
        return 1.0

def analyze_news_context_v2(headlines: list[dict]) -> dict:
    pipe = _get_pipeline()
    if not headlines or not pipe:
        return {"score": 0.0, "drivers": [], "headline_count": 0}

    from config import FINBERT_MAX_HEADLINES
    texts = []
    weights = []
    valid_headlines = []
    
    for h in headlines[:FINBERT_MAX_HEADLINES]:
        title = h.get("title", "").strip()
        if not title: continue
        
        full_text = (title + ". " + h.get("summary", "")).strip()
        texts.append(full_text)
        
        t_weight = _calculate_time_weight(h.get("published_timestamp", time.time()))
        weights.append(t_weight)
        valid_headlines.append(title)

    if not texts:
        return {"score": 0.0, "drivers": [], "headline_count": 0}

    try:
        # Batch Inference for speed
        results = pipe(texts, batch_size=len(texts), truncation=True)
    except Exception as e:
        logger.error(f"[FinBERT] Inference error: {e}")
        return {"score": 0.0, "drivers": [], "headline_count": 0}

    # Weighted Scoring
    weighted_scores = []
    for i, res in enumerate(results):
        label = res['label']
        score = res['score'] # Model confidence
        
        val = 0
        if label == 'positive': val = 100
        elif label == 'negative': val = -100
        
        final_weight = weights[i] * score
        weighted_scores.append(val * final_weight)

    weight_sum = np.sum(weights)
    mean_score = np.sum(weighted_scores) / weight_sum if weight_sum > 0 else 0.0

    return {
        "score": float(mean_score),
        "headline_count": len(texts),
        "drivers": valid_headlines[:3]
    }
