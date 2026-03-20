"""
news.py — news headline retrieval.
Primary source: yfinance ticker.news.
Fallback:       Google News RSS via feedparser.
"""
import re
import yfinance as yf


def fetch_rss_news(ticker: str, company_name: str = "",
                   max_items: int = 10) -> list:
    """
    Fetch headlines from Google News RSS for *ticker*.
    Falls back to a company-name query if the ticker search returns nothing.
    Returns a list of dicts with keys: title, summary.
    """
    try:
        import feedparser
    except ImportError:
        print("    [RSS] feedparser not installed — run: pip install feedparser")
        return []

    def _query_feed(query: str) -> list:
        url = (
            f"https://news.google.com/rss/search"
            f"?q={query}&hl=en-US&gl=US&ceid=US:en"
        )
        try:
            feed    = feedparser.parse(url)
            results = []
            for entry in feed.entries[:max_items]:
                title   = entry.get("title", "").strip()
                summary = entry.get("summary", "").strip()
                summary = re.sub(r"<[^>]+>", "", summary)[:200]
                if title and len(title) > 5:
                    results.append({"title": title, "summary": summary})
            return results
        except Exception as e:
            print(f"    [RSS] Feed error for '{query}': {e}")
            return []

    results = _query_feed(f"{ticker}+stock+news")
    if not results and company_name:
        results = _query_feed(f"{company_name.replace(' ', '+')}+stock")
    return results


def fetch_news_for_ticker(name: str, sym: str) -> list:
    """
    Fetch news for a single ticker.
    Tries yfinance first; falls back to Google News RSS.
    Returns a list of dicts with keys: title, summary.
    """
    news: list = []
    try:
        tk      = yf.Ticker(sym)
        yf_news = tk.news or []
        news    = [n for n in yf_news if n.get("title")]
    except Exception:
        pass
    if not news:
        news = fetch_rss_news(sym, company_name=name)
    return news
