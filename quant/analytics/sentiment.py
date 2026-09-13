# sentiment.py
"""
sentiment.py — FinBERT NLP scoring engine.
Pillar 3: Batched (vectorized) forward pass — stack chunks into one tensor,
single forward pass instead of N separate _model.predict() calls.
Pillar 6: OOP NLPScorer with dependency injection — instantiate once, pass in.
Intent: eliminate Python/C++ boundary crossing overhead and global-state memory leaks.
Invariants: model/tokenizer owned by instance; score_document returns dict with
score/reasoning/doc_hash. Pure scoring, no I/O.
Dependencies: torch, transformers.
"""
import hashlib
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from quant.config import FINBERT_MODEL


class NLPScorer:
    """FinBERT scorer. Instantiate ONCE at top level, inject into callers."""

    def __init__(self, model_name: str = FINBERT_MODEL) -> None:
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = "1"
        torch.set_num_threads(1)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def score_document(self, text: str) -> dict:
        """Score a corporate document. Batched single forward pass."""
        if not text or len(text.strip()) < 50:
            return {"score": 0.0, "reasoning": "Missing SEC data.", "doc_hash": None}

        doc_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()

        # Truncate BEFORE tokenizing to bound RAM (4000 tokens ~ 15-20k chars).
        text = text[:20000]

        tokens = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        chunks = [tokens[i:i + 510] for i in range(0, len(tokens), 510)][:8]

        if not chunks:
            return {"score": 0.0, "reasoning": "No tokens.", "doc_hash": doc_hash}

        # 1. Build [CLS] + chunk + [SEP] tensors
        tensors = [
            torch.tensor([self.tokenizer.cls_token_id] + c + [self.tokenizer.sep_token_id])
            for c in chunks
        ]

        # 2. Pad to same length -> matrix (num_chunks, seq_len)
        padded = pad_sequence(tensors, batch_first=True, padding_value=0)
        attention_mask = (padded != 0).int()

        # 3. Single vectorized forward pass (AVX2 CPU path)
        with torch.no_grad():
            outputs = self.model(input_ids=padded, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # net_sentiment per chunk: P(positive) - P(negative)
            net = probs[:, 0] - probs[:, 1]

        # Weight by chunk length (exclude padding)
        lengths = torch.tensor([len(t) for t in tensors], dtype=torch.float)
        total_weight = lengths.sum().item()
        total_sentiment = (net * lengths).sum().item()

        final_score = (total_sentiment / total_weight) * 100 if total_weight > 0 else 0.0
        return {"score": final_score, "reasoning": "SEC 8-K Computed", "doc_hash": doc_hash}


class FinBERTBatchScorer:
    """Batch FinBERT across MANY documents in one model call.

    Intent: process all survivor documents in a global batch queue instead of
    one document at a time. Chunks from all docs are pooled, padded, and run
    through a single forward pass per batch — dramatically faster than
    per-document sequential inference.
    Invariants: score_texts returns one float per input text in same order.
    Dependencies: torch, transformers.
    """

    def __init__(self, model_name: str = FINBERT_MODEL, max_batch_size: int = 16) -> None:
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = "1"
        torch.set_num_threads(1)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.max_batch_size = max_batch_size

    def score_texts(self, texts: list[str]) -> list[float]:
        """Score a list of documents, returning one score per document."""
        all_chunks: list[torch.Tensor] = []
        chunk_map: list[int] = []

        for doc_id, text in enumerate(texts):
            if not text or len(text.strip()) < 50:
                continue
            text = text[:20000]
            tokens = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
            chunks = [tokens[i:i + 510] for i in range(0, len(tokens), 510)][:8]
            for chunk in chunks:
                all_chunks.append(torch.tensor([self.tokenizer.cls_token_id] + chunk + [self.tokenizer.sep_token_id], dtype=torch.long))
                chunk_map.append(doc_id)

        scores = [0.0] * len(texts)
        counts = [0] * len(texts)

        if not all_chunks:
            return scores

        with torch.no_grad():
            for i in range(0, len(all_chunks), self.max_batch_size):
                batch = all_chunks[i:i + self.max_batch_size]
                batch_ids = chunk_map[i:i + self.max_batch_size]

                padded = pad_sequence(batch, batch_first=True, padding_value=0)
                attention_mask = (padded != 0).int()

                out = self.model(input_ids=padded, attention_mask=attention_mask)
                probs = torch.softmax(out.logits, dim=-1)
                net = probs[:, 0] - probs[:, 1]

                for doc_id, score in zip(batch_ids, net.tolist()):
                    scores[doc_id] += score
                    counts[doc_id] += 1

        return [
            (scores[i] / counts[i]) * 100 if counts[i] > 0 else 0.0
            for i in range(len(texts))
        ]


# ── Backward-compatible module-level API ───────────────────────────────────────
# Kept for callers that still use init_worker()/score_corporate_document().
_scorer: NLPScorer | None = None


def init_worker() -> None:
    """Initialise a module-level NLPScorer. Idempotent."""
    global _scorer
    if _scorer is None:
        _scorer = NLPScorer()


def unload_worker() -> None:
    """Free the scorer from RAM to prevent multiprocessing fork inheritance."""
    global _scorer
    _scorer = None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score_corporate_document(text: str) -> dict:
    """Module-level convenience wrapper (uses global scorer)."""
    global _scorer
    if _scorer is None:
        init_worker()
    return _scorer.score_document(text)
