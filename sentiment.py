import torch
from transformers import pipeline
import config

# Локализация вычислений: 0 для дискретного GPU, -1 для центрального процессора
hardware_device = 0 if torch.cuda.is_available() else -1

# Статичная загрузка весов в память
analyzer = pipeline(
    "sentiment-analysis", 
    model=config.SENTIMENT_MODEL, 
    device=hardware_device
)

def get_aggregated_sentiment(headlines: list[str]) -> float:
    if not headlines:
        return 0.0

    # Пакетный прямой проход (forward pass)
    predictions = analyzer(headlines)
    
    cumulative_signal = 0.0
    signal_count = 0

    for item in predictions:
        label = item['label']
        probability = item['score']

        # Игнорирование неопределенности и отсутствия сущностного вектора
        if probability < 0.75 or label == 'neutral':
            continue

        if label == 'positive':
            cumulative_signal += probability
            signal_count += 1
        elif label == 'negative':
            cumulative_signal -= probability
            signal_count += 1

    if signal_count == 0:
        return 0.0

    # Возврат нормализованного значения
    return cumulative_signal / signal_count
