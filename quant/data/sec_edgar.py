from quant import paths
import os
import threading
import logging
from sec_edgar_downloader import Downloader
from selectolax.parser import HTMLParser

logger = logging.getLogger(__name__)

def _download_and_parse(symbol: str, download_dir: str, result_container: list) -> None:
    # SEC blocks generic agents. Changed to unique ID.
    dl = Downloader("MyUniqueQuant", "data@myuniquequant.com", download_dir) 
    try:
        dl.get("8-K", symbol, limit=1, download_details=False)
        
        filing_path = os.path.join(download_dir, "sec-edgar-filings", symbol, "8-K")
        if not os.path.exists(filing_path):
            return
            
        for root, dirs, files in os.walk(filing_path):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
# sec_edgar.py -> _download_and_parse()
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        # FIX: Read up to 1MB of HTML instead of the entire multi-MB file
                        raw_html = f.read(1024 * 1024)
                        # Pillar 5: selectolax C-based parser ~50x faster than BS4
                        tree = HTMLParser(raw_html)
                        result_container[0] = tree.text(separator=" ").strip()[:50000]
                        return
    except Exception:
        pass

# Increased timeout: 5s -> 15s
def fetch_latest_8k(symbol: str, download_dir: str = paths.SEC_FILINGS_DIR, timeout: int = 15) -> str:
    """
    Fetch SEC 8-K with strict daemon thread.
    """
    result = [""]
    t = threading.Thread(
        target=_download_and_parse, 
        args=(symbol, download_dir, result),
        daemon=True 
    )
    t.start()
    t.join(timeout)
    
    if t.is_alive():
        logger.warning(f"[{symbol}] SEC fetch timeout (> {timeout}s). Bypassing.")
        return ""
        
    return result[0]
