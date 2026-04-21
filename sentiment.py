import numpy as np
import torch
import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def _calculate_time_weight(published_str: str) -> float:
    """Applies an exponential decay weight based on headline age (24h half-life)."""
    try:
        # Convert published_str (from RSS/yfinance) to timestamp
        # Note: This requires standardizing date strings in news.py
        pub_time = float(published_str) # Simplified for example
        age_hours = (time.time() - pub_time) / 3600
        # Weight = 0.5 ^ (age / 24) -> 50% weight after 24 hours
        return 0.5 ** (age_hours / 24.0)
    except:
        return 1.0

def analyze_news_context_v2(headlines: list[dict]) -> dict:
    pipe = _get_pipeline()
    if not headlines or not pipe:
        return {"score": 0.0, "drivers": [], "headline_count": 0}

    # 1. Prepare Data & Batching
    texts = []
    weights = []
    valid_headlines = []
    
    for h in headlines[:32]:
        title = h.get("title", "").strip()
        if not title: continue
        
        full_text = (title + ". " + h.get("summary", "")).strip()
        texts.append(full_text)
        
        # Combine Time Weight and Confidence (Placeholder for Confidence)
        t_weight = _calculate_time_weight(h.get("published_timestamp", time.time()))
        weights.append(t_weight)
        valid_headlines.append(title)

    if not texts:
        return {"score": 0.0, "drivers": [], "headline_count": 0}

    # 2. Batch Inference (Highly optimized for local GPU/CPU)
    # processing 32 headlines in one pass is significantly faster than 32 individual passes
    results = pipe(texts, batch_size=len(texts), truncation=True)

    # 3. Weighted Scoring
    weighted_scores = []
    for i, res in enumerate(results):
        label = res['label']
        score = res['score'] # This is the model confidence (0.0 to 1.0)
        
        # Map labels to signed values
        val = 0
        if label == 'positive': val = 100
        elif label == 'negative': val = -100
        
        # Final weight = Time Decay * Model Confidence
        final_weight = weights[i] * score
        weighted_scores.append(val * final_weight)

    mean_weighted_score = np.sum(weighted_scores) / np.sum(weights) if np.sum(weights) > 0 else 0.0

    return {
        "score": float(mean_weighted_score),
        "headline_count": len(texts),
        "drivers": valid_headlines[:3] # Simplified drivers
    }
