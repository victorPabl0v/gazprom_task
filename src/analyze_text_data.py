"""Выполняет EDA, тематическое моделирование и базовую модель с отбором признаков"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


CONFIG_PATH = Path("configs/config.yaml")
LABELS = {0: "NEUTRAL", 1: "POSITIVE", 2: "NEGATIVE"}
RUSSIAN_STOP_WORDS = {
    "а",
    "без",
    "бы",
    "был",
    "была",
    "были",
    "было",
    "быть",
    "в",
    "вам",
    "вас",
    "весь",
    "во",
    "вот",
    "все",
    "всего",
    "всех",
    "вы",
    "где",
    "да",
    "даже",
    "для",
    "до",
    "его",
    "ее",
    "если",
    "есть",
    "еще",
    "же",
    "за",
    "и",
    "из",
    "или",
    "им",
    "их",
    "к",
    "как",
    "ко",
    "когда",
    "кто",
    "ли",
    "либо",
    "мне",
    "может",
    "мы",
    "на",
    "над",
    "надо",
    "наш",
    "не",
    "него",
    "нее",
    "нет",
    "ни",
    "них",
    "но",
    "ну",
    "о",
    "об",
    "однако",
    "он",
    "она",
    "они",
    "оно",
    "от",
    "очень",
    "по",
    "под",
    "при",
    "с",
    "со",
    "так",
    "также",
    "такой",
    "там",
    "те",
    "тем",
    "то",
    "того",
    "тоже",
    "той",
    "только",
    "том",
    "ты",
    "у",
    "уже",
    "хотя",
    "чего",
    "чем",
    "что",
    "чтобы",
    "это",
    "этого",
    "этой",
    "этом",
    "этот",
    "я",
}


def load_raw_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Загружает YAML конфигурацию как словарь"""

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def read_dataset(path: Path) -> pd.DataFrame:
    """Читает датасет и оставляет нужные колонки"""

    df = pd.read_csv(path)
    return df[["text", "sentiment"]].dropna().copy()


def clean_text(text: str) -> str:
    """Нормализует текст для классических текстовых моделей"""

    text = str(text).lower().replace("ё", "е")
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\d+", " NUM ", text)
    text = re.sub(r"[^a-zа-я0-9\s!?.,:-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет ручные текстовые признаки для анализа и базовой модели"""

    result = df.copy()
    text = result["text"].astype(str)
    result["clean_text"] = text.map(clean_text)
    result["char_len"] = text.str.len()
    result["word_count"] = text.str.split().map(len)
    result["avg_word_len"] = result["char_len"] / result["word_count"].clip(lower=1)
    result["exclamation_count"] = text.str.count("!")
    result["question_count"] = text.str.count(r"\?")
    result["digit_count"] = text.str.count(r"\d")
    result["uppercase_ratio"] = text.map(
        lambda value: sum(ch.isupper() for ch in value) / max(len(value), 1)
    )
    return result


def sample_dataframe(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """Возвращает воспроизводимую стратифицированную подвыборку"""

    if len(df) <= sample_size:
        return df
    sample_parts = []
    for _, part in df.groupby("sentiment"):
        sample_parts.append(part.sample(frac=sample_size / len(df), random_state=seed))
    return (
        pd.concat(sample_parts, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def describe_eda(df: pd.DataFrame) -> dict[str, Any]:
    """Собирает числовое описание датасета и текстовых признаков"""

    feature_cols = [
        "char_len",
        "word_count",
        "avg_word_len",
        "exclamation_count",
        "question_count",
        "digit_count",
        "uppercase_ratio",
    ]
    return {
        "rows": int(len(df)),
        "class_distribution": {
            LABELS[int(label)]: int(count)
            for label, count in df["sentiment"].value_counts().sort_index().items()
        },
        "duplicates": int(df["text"].duplicated().sum()),
        "empty_after_clean": int((df["clean_text"].str.len() == 0).sum()),
        "features_by_class": {
            LABELS[int(label)]: group[feature_cols].mean().round(4).to_dict()
            for label, group in df.groupby("sentiment")
        },
        "feature_quantiles": df[feature_cols]
        .quantile([0.25, 0.5, 0.75, 0.95])
        .round(4)
        .to_dict(),
    }


def save_class_distribution(df: pd.DataFrame, path: Path) -> None:
    """Сохраняет график распределения классов"""

    counts = df["sentiment"].map(LABELS).value_counts().reindex(LABELS.values())
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", color=["#4c78a8", "#59a14f", "#e15759"], ax=ax)
    ax.set_title("Распределение классов")
    ax.set_xlabel("Тональность")
    ax.set_ylabel("Тексты")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_length_distribution(df: pd.DataFrame, path: Path) -> None:
    """Сохраняет график распределения длины текстов по классам"""

    fig, ax = plt.subplots(figsize=(8, 4))
    for label_id, label_name in LABELS.items():
        values = df.loc[df["sentiment"] == label_id, "word_count"].clip(upper=250)
        ax.hist(values, bins=40, alpha=0.45, label=label_name)
    ax.set_title("Распределение числа слов с ограничением на 250")
    ax.set_xlabel("Слова")
    ax.set_ylabel("Тексты")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_vectorizer(max_features: int) -> TfidfVectorizer:
    """Создает TF IDF векторизатор для русскоязычных текстов"""

    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.85,
        max_features=max_features,
        sublinear_tf=True,
        stop_words=list(RUSSIAN_STOP_WORDS),
        token_pattern=r"(?u)\b[а-яa-z][а-яa-z]+\b",
    )


def run_topic_modeling(
    texts: pd.Series,
    topic_count: int,
    top_words: int,
    max_features: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Обучает NMF модель тем и возвращает темы с координатами"""

    vectorizer = build_vectorizer(max_features)
    matrix = vectorizer.fit_transform(texts)
    model = NMF(
        n_components=topic_count,
        init="nndsvda",
        random_state=seed,
        max_iter=300,
        l1_ratio=0.0,
    )
    doc_topic = model.fit_transform(matrix)
    feature_names = np.array(vectorizer.get_feature_names_out())
    topics = []
    for idx, weights in enumerate(model.components_):
        top_idx = weights.argsort()[-top_words:][::-1]
        topics.append(
            {
                "topic_id": idx,
                "top_words": feature_names[top_idx].tolist(),
                "weight": float(doc_topic[:, idx].sum()),
            }
        )
    coords = TruncatedSVD(n_components=2, random_state=seed).fit_transform(doc_topic)
    return {"algorithm": "NMF_TFIDF", "topics": topics}, doc_topic, coords


def save_topic_words(topic_info: dict[str, Any], path: Path) -> None:
    """Сохраняет визуализацию ключевых слов для каждой темы"""

    topics = topic_info["topics"]
    fig, axes = plt.subplots(len(topics), 1, figsize=(10, max(12, len(topics) * 1.8)))
    if len(topics) == 1:
        axes = [axes]
    for ax, topic in zip(axes, topics):
        words = topic["top_words"]
        ax.barh(range(len(words)), list(range(len(words), 0, -1)), color="#4c78a8")
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.invert_yaxis()
        ax.set_title(f"Тема {topic['topic_id']}")
        ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_topic_scatter(coords: np.ndarray, doc_topic: np.ndarray, path: Path) -> None:
    """Сохраняет двумерную визуализацию документов по доминирующим темам"""

    dominant = doc_topic.argmax(axis=1)
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], c=dominant, s=8, alpha=0.6, cmap="tab10"
    )
    ax.set_title("Карта документов по доминирующей NMF теме")
    ax.set_xlabel("SVD 1")
    ax.set_ylabel("SVD 2")
    fig.colorbar(scatter, ax=ax, label="Тема")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_manual_feature_matrix(df: pd.DataFrame) -> csr_matrix:
    """Создает матрицу ручных числовых признаков"""

    cols = [
        "char_len",
        "word_count",
        "avg_word_len",
        "exclamation_count",
        "question_count",
        "digit_count",
        "uppercase_ratio",
    ]
    scaled = MinMaxScaler().fit_transform(df[cols])
    return csr_matrix(scaled)


def evaluate_feature_baseline(
    df: pd.DataFrame,
    max_features: int,
    selected_features: int,
    seed: int,
) -> dict[str, Any]:
    """Обучает базовую TF IDF модель с ручными признаками и SelectKBest"""

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=df["sentiment"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_df["sentiment"],
    )
    vectorizer = build_vectorizer(max_features)
    selector = SelectKBest(chi2, k=min(selected_features, max_features))
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="saga",
        n_jobs=-1,
        random_state=seed,
    )

    start = time.perf_counter()
    x_train_tfidf = vectorizer.fit_transform(train_df["clean_text"])
    x_val_tfidf = vectorizer.transform(val_df["clean_text"])
    x_test_tfidf = vectorizer.transform(test_df["clean_text"])
    x_train_selected = selector.fit_transform(x_train_tfidf, train_df["sentiment"])
    x_val_selected = selector.transform(x_val_tfidf)
    x_test_selected = selector.transform(x_test_tfidf)
    x_train = hstack([x_train_selected, make_manual_feature_matrix(train_df)]).tocsr()
    x_val = hstack([x_val_selected, make_manual_feature_matrix(val_df)]).tocsr()
    x_test = hstack([x_test_selected, make_manual_feature_matrix(test_df)]).tocsr()
    classifier.fit(x_train, train_df["sentiment"])
    train_sec = time.perf_counter() - start

    start = time.perf_counter()
    preds = classifier.predict(x_test)
    infer_sec = time.perf_counter() - start
    selected_names = np.array(vectorizer.get_feature_names_out())[
        selector.get_support()
    ]
    scores = selector.scores_[selector.get_support()]
    top_idx = np.argsort(scores)[-30:][::-1]
    return {
        "split_sizes": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "tfidf_features": int(len(vectorizer.get_feature_names_out())),
        "selected_tfidf_features": int(x_train_selected.shape[1]),
        "manual_features": int(x_train.shape[1] - x_train_selected.shape[1]),
        "train_sec": train_sec,
        "test_inference_sec": infer_sec,
        "test_samples_per_sec": len(test_df) / infer_sec,
        "validation_macro_f1": f1_score(
            val_df["sentiment"], classifier.predict(x_val), average="macro"
        ),
        "test": {
            "accuracy": accuracy_score(test_df["sentiment"], preds),
            "f1_macro": f1_score(test_df["sentiment"], preds, average="macro"),
            "f1_weighted": f1_score(test_df["sentiment"], preds, average="weighted"),
            "confusion_matrix": confusion_matrix(test_df["sentiment"], preds).tolist(),
            "classification_report": classification_report(
                test_df["sentiment"],
                preds,
                target_names=[LABELS[idx] for idx in sorted(LABELS)],
                digits=4,
                zero_division=0,
            ),
        },
        "top_selected_features": selected_names[top_idx].tolist(),
    }


def save_json(data: dict[str, Any], path: Path) -> None:
    """Сохраняет словарь в JSON файл"""

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main() -> None:
    """Запускает полный анализ текстовых данных и базовую модель с признаками"""

    config = load_raw_config()
    analysis_config = config["analysis"]
    results_dir = Path(config["results"]["dir"])
    plots_dir = Path(analysis_config["plots_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = add_text_features(read_dataset(Path(analysis_config["full_data_path"])))
    sample_df = sample_dataframe(
        df,
        int(analysis_config["sample_size"]),
        int(analysis_config["random_state"]),
    )
    eda = describe_eda(df)
    save_class_distribution(df, plots_dir / "class_distribution.png")
    save_length_distribution(df, plots_dir / "text_length_distribution.png")

    topic_info, doc_topic, coords = run_topic_modeling(
        sample_df["clean_text"],
        int(analysis_config["topic_count"]),
        int(analysis_config["topic_top_words"]),
        int(analysis_config["tfidf_max_features"]),
        int(analysis_config["random_state"]),
    )
    save_topic_words(topic_info, plots_dir / "topic_words.png")
    save_topic_scatter(coords, doc_topic, plots_dir / "topic_scatter.png")

    baseline = evaluate_feature_baseline(
        sample_df,
        int(analysis_config["tfidf_max_features"]),
        int(analysis_config["selected_features"]),
        int(analysis_config["random_state"]),
    )
    save_json(
        {
            "eda": eda,
            "topic_modeling": topic_info,
            "feature_engineering_selection_baseline": baseline,
        },
        results_dir / analysis_config["analysis_json"],
    )
    print("saved", results_dir / analysis_config["analysis_json"], plots_dir)


if __name__ == "__main__":
    main()
