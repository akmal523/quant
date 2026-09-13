import feedparser
import logging

logger = logging.getLogger(__name__)

def fetch_news_headlines(symbol: str) -> str:
    """
    Fetches the latest 5 headlines from Yahoo Finance RSS.
    Acts as a lightweight sentiment source for Non-US tickers.
    """
    url = f"https://finance.yahoo.com/rss/headline?s={symbol}"
    try:
        feed = feedparser.parse(url)
        if not feed.entries:
            return ""
            
        # Join top 5 headlines into a single block for FinBERT
        headlines = [entry.title for entry in feed.entries[:5]]
        return " | ".join(headlines)
    except Exception as e:
        logger.warning(f"[{symbol}] News fetch failed: {e}")
        return ""
