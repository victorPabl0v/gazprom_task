---
language:
- ru
tags:
- sentiment
- text-classification
---

# Датасет тональности русскоязычных текстов

Проект содержит полный пайплайн анализа русскоязычных текстов: EDA, тематическое моделирование, работу с признаками, обучение классификатора тональности, оценку скорости и CPU-оптимизацию

## Структура проекта

- `src/train_sentiment.py` - обучение, оценка, бенчмарк, PyTorch INT8 оптимизация и экспорт ONNX
- `src/analyze_text_data.py` - EDA, конструирование признаков, отбор признаков, тематическое моделирование и визуализации
- `configs/config.yaml` - настройки модели, данных, обучения, оценки и выходных файлов
- `data/raw/train.csv`, `data/raw/valid.csv`, `data/raw/datasets.csv` - исходные данные
- `notebooks/` - выполненные ноутбуки с видимыми выводами
- `reports/report.md` - итоговый отчет по техническому заданию
- `models/sentiment/` - сохраненная модель, токенизатор и оптимизированные артефакты
- `results/` - метрики, бенчмарки, графики и логи

## Запуск

Обучение классификатора:

```bash
python src/train_sentiment.py
```

EDA, тематическое моделирование, конструирование признаков и отбор признаков:

```bash
python src/analyze_text_data.py
```

Путь к YAML можно переопределить через `CONFIG_PATH=path/to/config.yaml`

Поддерживаются env-переопределения: `MODEL_NAME`, `MAX_LEN`, `BATCH_SIZE`, `LR`, `EPOCHS`, `TRAIN_MAX`, `VAL_MAX`

## Метрика качества

Основная и отображаемая метрика качества: `Macro F1`

Текущий результат сохраненной модели на test-разбиении:

- `Macro F1`: `0.7712`

Подробности находятся в `reports/report.md`, `results/sentiment_metrics.json` и `results/sentiment_benchmark.csv`

Артефакты EDA и тематического моделирования сохранены в `results/text_analysis.json` и `results/figures/`

## Значения классов

- `0` - NEUTRAL
- `1` - POSITIVE
- `2` - NEGATIVE

## Источники данных

**[Sentiment Analysis in Russian](https://www.kaggle.com/c/sentiment-analysis-in-russian/data)**

Данные тональности русскоязычных новостей из Kaggle competition

**[Russian Language Toxic Comments](https://www.kaggle.com/blackmoon/russian-language-toxic-comments/)**

Небольшой датасет размеченных комментариев из 2ch.hk и pikabu.ru

**[Датасет автомобильных отзывов для машинного обучения](https://github.com/oldaandozerskaya/auto_reviews)**

Датасет автомобильных отзывов для задач анализа тональности

**[Sentiment datasets by Blinov](https://github.com/natasha/corus/issues/14)**

Коллекции отзывов из разных доменов

**[LINIS Crowd](http://www.linis-crowd.org/)**

Коллекция текстов с тональной разметкой и тональный словарь LINIS Crowd SENT

**[Датасет русскоязычных отзывов об отелях](https://drive.google.com/drive/folders/17sa3h4XHcG0MJGrbfOsbL-kDW29CuJul)**

Датасет русскоязычных отзывов об отелях
