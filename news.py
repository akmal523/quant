import urllib.request
import xml.etree.ElementTree as ET
import difflib
from datetime import datetime

def get_recent_headlines(symbol: str) -> list[dict]:
    """Извлечение структуры: дата публикации и суть события."""
    # Используем полный символ (например, HEI.DE) для точности
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            xml_data = response.read()
            
        root = ET.fromstring(xml_data)
        unique_items = []
        seen_titles = []
        
        for item in root.findall('.//item'):
            title_node = item.find('title')
            date_node = item.find('pubDate')
            
            if title_node is None: continue
            title = title_node.text
            
            pub_date = ""
            if date_node is not None:
                try:
                    dt = datetime.strptime(date_node.text, "%a, %d %b %Y %H:%M:%S %z")
                    pub_date = dt.strftime("%d %b")
                except Exception:
                    pub_date = "Дата скрыта"

            is_duplicate = False
            for u in seen_titles:
                if difflib.SequenceMatcher(None, title.lower(), u.lower()).ratio() > 0.8:
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                seen_titles.append(title)
                unique_items.append({'date': pub_date, 'title': title})
                
        return unique_items
    except Exception:
        return []
