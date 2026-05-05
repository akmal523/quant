# test_cache.py
import os
from sentiment import _save_cached_score, _get_cached_score
from fundamentals import _save_to_cache, _get_from_cache

def test_caches():
    # 1. Test NLP Cache
    _save_cached_score("hash999", 42.0)
    nlp_res = _get_cached_score("hash999")
    assert nlp_res == 42.0, "NLP Cache I/O failed."

    # 2. Test Fundamentals Cache
    dummy_data = {
        "PE": 15.0, "PEG": 1.1, "ROE": 0.2, 
        "DebtToEquity": 0.5, "EBIT": 500, "InterestExpense": 50
    }
    _save_to_cache("TEST_TICKER", dummy_data)
    fund_res = _get_from_cache("TEST_TICKER")
    
    assert fund_res is not None, "Fundamentals Cache empty."
    assert fund_res["PE"] == 15.0, "Fundamentals Cache mismatch."

    print("Task 2 validation passed. Caches operational.")

if __name__ == "__main__":
    test_caches()
