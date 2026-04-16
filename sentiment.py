import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import torch
from transformers import pipeline, logging
import config

logging.set_verbosity_error()
hardware_device = 0 if torch.cuda.is_available() else -1

analyzer = pipeline(
    "sentiment-analysis", 
    model=config.SENTIMENT_MODEL, 
    device=hardware_device
)

def analyze_news_context(news_items: list[dict]) -> dict:
    """Возвращает математический балл и конкретные триггеры падения."""
    if not news_items:
        return {'score': 0.0, 'drivers': []}

    titles = [item['title'] for item in news_items]
    predictions = analyzer(titles)
    
    cumulative_signal = 0.0
    signal_count = 0
    destructive_drivers = []

    for item, pred in zip(news_items, predictions):
        label = pred['label']
        probability = pred['score']

        if probability < 0.75 or label == 'neutral':
            continue

        if label == 'positive':
            cumulative_signal += probability
            signal_count += 1
        elif label == 'negative':
            cumulative_signal -= probability
            signal_count += 1
            # Фиксация источника угрозы
            destructive_drivers.append(f"[{item['date']}] {item['title']}")

    final_score = cumulative_signal / signal_count if signal_count > 0 else 0.0
    
    return {
        'score': final_score,
        'drivers': destructive_drivers[:2] # Ограничение: 2 главных фактора
    }
