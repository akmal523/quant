import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
# test_cache.py
import os
from quant.data.fundamentals import _save_to_cache, _get_from_cache, _compute_icr

def test_caches():
    # 1. Test Fundamentals Cache
    dummy_data = {
        "PE": 15.0, "PEG": 1.1, "ROE": 0.2, 
        "DebtToEquity": 0.5, "EBIT": 500, "InterestExpense": 50
    }
    _save_to_cache("TEST_TICKER", dummy_data)
    fund_res = _get_from_cache("TEST_TICKER")
    
    assert fund_res is not None, "Fundamentals Cache empty."
    assert fund_res["PE"] == 15.0, "Fundamentals Cache mismatch."
    assert _compute_icr(fund_res) == 10.0, f"ICR computation failed: {_compute_icr(fund_res)}"

    print("Cache validation passed. Fundamentals cache + ICR computation operational.")

if __name__ == "__main__":
    test_caches()
