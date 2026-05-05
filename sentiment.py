# sentiment.py
import hashlib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = None
model = None

def init_worker() -> None:
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()

def score_corporate_document(text: str) -> dict:
    if not text or len(text.strip()) < 50:
        return {"score": 0.0, "reasoning": "Missing SEC data.", "doc_hash": None}

    doc_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()

    if tokenizer is None:
        raise RuntimeError("Model not initialized.")

    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    chunks = [tokens[i:i + 510] for i in range(0, len(tokens), 510)][:8]

    total_sentiment = 0.0
    total_weight = 0

    with torch.no_grad():
        for chunk in chunks:
            input_ids = torch.tensor([[tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]])
            probs = torch.nn.functional.softmax(model(input_ids).logits, dim=-1)[0]
            net_sentiment = probs[0].item() - probs[1].item()
            
            chunk_weight = len(chunk)
            total_sentiment += net_sentiment * chunk_weight
            total_weight += chunk_weight

    final_score = (total_sentiment / total_weight) * 100 if total_weight > 0 else 0.0
    
    return {"score": final_score, "reasoning": "SEC 8-K Computed", "doc_hash": doc_hash}
