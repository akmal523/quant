import os
import threading
import logging
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup

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
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        raw_html = f.read()
                        soup = BeautifulSoup(raw_html, "html.parser")
                        result_container[0] = soup.get_text(separator=" ", strip=True)
                        return
    except Exception:
        pass

# Increased timeout: 5s -> 15s
def fetch_latest_8k(symbol: str, download_dir: str = "./sec_filings", timeout: int = 15) -> str:
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
