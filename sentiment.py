"""
sentiment.py — FinBERT pipeline with NER semantic pre-filter.

Problem: FinBERT scores isolated headlines with no entity context.
Example: YCA.L (Yellow Cake — uranium) gets scored on culinary baking articles
because the company name 'Yellow Cake' matches recipe headlines. This produces
absurd sentiment signals with no connection to the underlying business.

Solution: spaCy NER (Named Entity Recognition) runs *before* FinBERT.
A headline only proceeds to sentiment scoring if its ORG entities plausibly
match the company being analyzed. Unmatched headlines are discarded.

Graceful degradation: if spaCy is unavailable, NER filter is silently bypassed
and all headlines proceed to FinBERT (original behavior preserved).
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Lazy-initialized singletons — avoid paying startup cost unless needed
_pipeline:          Optional[object] = None
_nlp:               Optional[object] = None
_load_attempted:    bool = False
_nlp_load_attempted: bool = False


# ── FinBERT pipeline ──────────────────────────────────────────────────────────

def _get_pipeline():
    global _pipeline, _load_attempted
    if _load_attempted:
        return _pipeline
    _load_attempted = True
    try:
        import transformers.utils.logging as hf_logging
        hf_logging.set_verbosity_error()
        from transformers import pipeline as hf_pipeline
        from config import FINBERT_MODEL, FINBERT_DEVICE

        logger.info("[FinBERT] Loading model into memory...")
        _pipeline = hf_pipeline(
            "text-classification",
            model=FINBERT_MODEL,
            tokenizer=FINBERT_MODEL,
            device=FINBERT_DEVICE,
        )
    except Exception as exc:
        logger.error("[FinBERT] Failed to load pipeline: %s", exc)
    return _pipeline


# ── spaCy NER ─────────────────────────────────────────────────────────────────

def _get_nlp():
    """Lazy-load spaCy English NER model. Fails silently if not installed."""
    global _nlp, _nlp_load_attempted
    if _nlp_load_attempted:
        return _nlp
    _nlp_load_attempted = True
    try:
        from config import NER_SPACY_MODEL
        import spacy
        _nlp = spacy.load(NER_SPACY_MODEL)
        logger.info("[NER] spaCy '%s' loaded.", NER_SPACY_MODEL)
    except Exception as exc:
        logger.warning(
            "[NER] spaCy unavailable (%s). Headline filtering disabled — "
            "all headlines will pass to FinBERT.",
            exc,
        )
    return _nlp


def _entity_matches_company(text: str, company_name: str, ticker: str) -> bool:
    """
    Returns True if the headline's entities plausibly reference this emitter.

    Matching logic (OR — any condition is sufficient):
      1. Ticker root appears verbatim in text.
         Examples: 'CCJ', 'YCA', 'SHEL' (exchange suffix stripped: .L, .DE, .TO)
      2. Any ORG entity extracted by spaCy shares ≥1 'significant' token
         with the company display name.
         'Significant' = length > 3 chars, not a common corporate suffix.

    If spaCy NLP is unavailable, returns True (pass-through — graceful degradation).
    """
    nlp = _get_nlp()

    # Gate 1: ticker root match (fast, no NLP required)
    clean_ticker = ticker.split(".")[0].replace("=", "").lower()
    if len(clean_ticker) > 1 and clean_ticker in text.lower():
        return True

    if nlp is None:
        return True   # NER unavailable — bypass filter

    doc  = nlp(text)
    orgs = [ent.text.lower() for ent in doc.ents if ent.label_ == "ORG"]

    if not orgs:
        # No organizations named in headline — cannot be about a specific company.
        return False

    # Corporate suffix stoplist — these tokens alone do not constitute a match
    _STOPWORDS = {
        "corp", "inc", "inc.", "plc", "ltd", "group", "energy",
        "holdings", "resources", "technologies", "systems", "the",
    }
    name_tokens = [
        t.lower()
        for t in company_name.split()
        if len(t) > 3 and t.lower() not in _STOPWORDS
    ]

    if not name_tokens:
        # Company name reduced entirely to stopwords (e.g., "Group Holdings Ltd")
        return True   # Cannot discriminate — pass through

    # Gate 2: ORG entity / company name token overlap
    for org in orgs:
        for token in name_tokens:
            if token in org:
                return True

    return False


# ── Time weighting ────────────────────────────────────────────────────────────

def _calculate_time_weight(published_str: str) -> float:
    """Exponential decay with 24-hour half-life."""
    try:
        pub_time  = float(published_str)
        age_hours = (time.time() - pub_time) / 3600.0
        return 0.5 ** (age_hours / 24.0)
    except Exception:
        return 1.0


# ── Public interface ──────────────────────────────────────────────────────────

def analyze_news_context_v2(
    headlines:    list[dict],
    ticker:       str = "",
    company_name: str = "",
) -> dict:
    """
    NER-filtered FinBERT sentiment pipeline.

    Steps:
      1. Each headline passes through _entity_matches_company().
         Rejected headlines are logged and excluded from scoring.
      2. Survivors are batched into FinBERT for inference.
      3. Results are aggregated with exponential time-decay weights.

    Args:
      headlines:    List of {'title', 'summary', 'published_timestamp'} dicts.
      ticker:       Exchange ticker (e.g. 'YCA.L') — used for root-match filter.
      company_name: Display name (e.g. 'Yellow Cake') — used for NER token match.

    Returns dict:
      score          : Weighted sentiment in [-100, +100]
      headline_count : Headlines that passed NER and entered FinBERT
      ner_filtered   : Headlines rejected by NER pre-filter
      drivers        : Top 3 headlines that influenced the score
    """
    pipe = _get_pipeline()
    if not headlines or not pipe:
        return {"score": 0.0, "drivers": [], "headline_count": 0, "ner_filtered": 0}

    from config import FINBERT_MAX_HEADLINES, NER_ENABLED

    texts:           list[str]  = []
    weights:         list[float] = []
    valid_headlines: list[str]  = []
    ner_filtered_count:  int    = 0

    for h in headlines[:FINBERT_MAX_HEADLINES]:
        title = h.get("title", "").strip()
        if not title:
            continue

        # NER gate: only apply when ticker is known and feature is enabled
        if NER_ENABLED and ticker:
            if not _entity_matches_company(title, company_name, ticker):
                ner_filtered_count += 1
                logger.debug(
                    "[NER] Filtered for %s/%s: '%s'",
                    ticker, company_name, title[:70],
                )
                continue

        full_text = (title + ". " + h.get("summary", "")).strip()
        texts.append(full_text)
        weights.append(_calculate_time_weight(h.get("published_timestamp", str(time.time()))))
        valid_headlines.append(title)

    if not texts:
        return {
            "score": 0.0, "drivers": [], "headline_count": 0,
            "ner_filtered": ner_filtered_count,
        }

    try:
        results = pipe(texts, batch_size=len(texts), truncation=True)
    except Exception as exc:
        logger.error("[FinBERT] Inference error: %s", exc)
        return {
            "score": 0.0, "drivers": [], "headline_count": 0,
            "ner_filtered": ner_filtered_count,
        }

    # Weighted aggregation
    weighted_scores: list[float] = []
    for i, res in enumerate(results):
        label = res["label"]
        conf  = res["score"]
        val   = 100.0 if label == "positive" else (-100.0 if label == "negative" else 0.0)
        weighted_scores.append(val * weights[i] * conf)

    weight_sum = float(np.sum(weights))
    mean_score = float(np.sum(weighted_scores)) / weight_sum if weight_sum > 0 else 0.0

    return {
        "score":          mean_score,
        "headline_count": len(texts),
        "ner_filtered":   ner_filtered_count,
        "drivers":        valid_headlines[:3],
    }
