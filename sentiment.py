# sentiment.py
import hashlib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_tokenizer = None
_model = None

def init_worker() -> None:
    """Initialise FinBERT model per worker process. Idempotent (guards re-init)."""
    global _tokenizer, _model
    if _tokenizer is not None:
        return
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    _tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    _model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    _model.eval()

def score_corporate_document(text: str) -> dict:
    if not text or len(text.strip()) < 50:
        return {"score": 0.0, "reasoning": "Missing SEC data.", "doc_hash": None}

    doc_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()

    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        raise RuntimeError("Model not initialized. Call init_worker() first.")

    # Use proper tokenizer with attention masks instead of raw token IDs
    tokens = _tokenizer.encode(text, add_special_tokens=False, truncation=False)
    chunks = [tokens[i:i + 510] for i in range(0, len(tokens), 510)][:8]

    total_sentiment = 0.0
    total_weight = 0

    with torch.no_grad():
        for chunk in chunks:
            encoded = _tokenizer.prepare_for_model(
                chunk,
                add_special_tokens=True,
                return_tensors="pt",
                truncation=False,
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask", None)

            outputs = _model(input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            net_sentiment = probs[0].item() - probs[1].item()

            chunk_weight = len(chunk)
            total_sentiment += net_sentiment * chunk_weight
            total_weight += chunk_weight

    final_score = (total_sentiment / total_weight) * 100 if total_weight > 0 else 0.0

    return {"score": final_score, "reasoning": "SEC 8-K Computed", "doc_hash": doc_hash}
