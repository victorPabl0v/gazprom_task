"""Обучает, оценивает, бенчмаркает и экспортирует классификатор тональности"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


CONFIG_PATH = Path(os.getenv("CONFIG_PATH", "configs/config.yaml"))
ENV_OVERRIDES: dict[str, tuple[str, str, type]] = {
    "MODEL_NAME": ("model", "name", str),
    "MAX_LEN": ("training", "max_len", int),
    "BATCH_SIZE": ("training", "batch_size", int),
    "LR": ("training", "learning_rate", float),
    "EPOCHS": ("training", "epochs", int),
    "TRAIN_MAX": ("data", "train_max", int),
    "VAL_MAX": ("data", "validation_max", int),
}


@dataclass(frozen=True)
class ModelConfig:
    """Хранит название модели и каталог сохранения"""

    name: str
    output_dir: Path


@dataclass(frozen=True)
class DataConfig:
    """Хранит пути к данным и параметры разбиения"""

    train_path: Path
    validation_path: Path
    test_size: float
    train_max: int | None
    validation_max: int | None


@dataclass(frozen=True)
class TrainingConfig:
    """Хранит гиперпараметры обучения"""

    max_len: int
    batch_size: int
    learning_rate: float
    epochs: int
    warmup_ratio: float
    weight_decay: float
    seed: int


@dataclass(frozen=True)
class EvaluationConfig:
    """Хранит параметры оценки, бенчмарка и подписей классов"""

    benchmark_runs: int
    benchmark_max_samples: int
    short_max_len: int
    labels: dict[int, str]


@dataclass(frozen=True)
class ResultsConfig:
    """Хранит каталог и имена файлов с результатами"""

    dir: Path
    training_log: str
    metrics_json: str
    benchmark_csv: str


@dataclass(frozen=True)
class AppConfig:
    """Объединяет все разделы конфигурации приложения"""

    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    results: ResultsConfig


def _empty_to_none(value: Any) -> Any:
    """Преобразует пустые значения YAML и env в None"""

    if value in ("", 0, "0"):
        return None
    return value


def _apply_env_overrides(raw_config: dict[str, Any]) -> dict[str, Any]:
    """Применяет совместимые переопределения через переменные окружения"""

    for env_name, (section, key, caster) in ENV_OVERRIDES.items():
        value = os.getenv(env_name)
        if value is None:
            continue
        raw_config.setdefault(section, {})[key] = _empty_to_none(caster(value))
    return raw_config


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    """Загружает YAML конфигурацию и возвращает типизированные настройки"""

    with path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}
    raw_config = _apply_env_overrides(raw_config)

    labels = {
        int(label_id): label_name
        for label_id, label_name in raw_config["evaluation"]["labels"].items()
    }
    return AppConfig(
        model=ModelConfig(
            name=raw_config["model"]["name"],
            output_dir=Path(raw_config["model"]["output_dir"]),
        ),
        data=DataConfig(
            train_path=Path(raw_config["data"]["train_path"]),
            validation_path=Path(raw_config["data"]["validation_path"]),
            test_size=float(raw_config["data"]["test_size"]),
            train_max=_empty_to_none(raw_config["data"].get("train_max")),
            validation_max=_empty_to_none(raw_config["data"].get("validation_max")),
        ),
        training=TrainingConfig(
            max_len=int(raw_config["training"]["max_len"]),
            batch_size=int(raw_config["training"]["batch_size"]),
            learning_rate=float(raw_config["training"]["learning_rate"]),
            epochs=int(raw_config["training"]["epochs"]),
            warmup_ratio=float(raw_config["training"]["warmup_ratio"]),
            weight_decay=float(raw_config["training"]["weight_decay"]),
            seed=int(raw_config["training"]["seed"]),
        ),
        evaluation=EvaluationConfig(
            benchmark_runs=int(raw_config["evaluation"]["benchmark_runs"]),
            benchmark_max_samples=int(
                raw_config["evaluation"]["benchmark_max_samples"]
            ),
            short_max_len=int(raw_config["evaluation"]["short_max_len"]),
            labels=labels,
        ),
        results=ResultsConfig(
            dir=Path(raw_config["results"]["dir"]),
            training_log=raw_config["results"]["training_log"],
            metrics_json=raw_config["results"]["metrics_json"],
            benchmark_csv=raw_config["results"]["benchmark_csv"],
        ),
    )


def set_seed(seed: int) -> None:
    """Фиксирует seed для numpy и torch"""

    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str | None = None) -> torch.device:
    """Возвращает запрошенное устройство или лучший доступный ускоритель"""

    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TextDataset(Dataset[dict[str, torch.Tensor]]):
    """Готовит датасет тональности для PyTorch DataLoader"""

    def __init__(self, df: pd.DataFrame, tokenizer: Any, max_len: int) -> None:
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["sentiment"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        """Возвращает размер датасета"""

        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Токенизирует один текст и добавляет его метку"""

        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def move_batch_to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    """Переносит токенизированный batch на выбранное устройство"""

    return {key: value.to(device) for key, value in batch.items()}


def sample_balanced(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    """Выбирает сбалансированную подвыборку при заданном лимите строк"""

    if max_rows is None or len(df) <= max_rows:
        return df

    parts: list[pd.DataFrame] = []
    per_class = max(1, max_rows // df["sentiment"].nunique())
    for _, group in df.groupby("sentiment"):
        parts.append(group.sample(min(len(group), per_class), random_state=seed))
    return pd.concat(parts, ignore_index=True)


def to_builtin(value: Any) -> Any:
    """Преобразует numpy значения в сериализуемые Python объекты"""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def calculate_metrics(
    labels: list[int],
    preds: list[int],
    label_names: dict[int, str],
) -> dict[str, Any]:
    """Считает метрики классификации для несбалансированной трехклассовой задачи"""

    label_ids = sorted(label_names)
    target_names = [label_names[label_id] for label_id in label_ids]
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(labels, preds, labels=label_ids),
        "classification_report": classification_report(
            labels,
            preds,
            labels=label_ids,
            target_names=target_names,
            digits=4,
            zero_division=0,
        ),
        "classification_report_dict": classification_report(
            labels,
            preds,
            labels=label_ids,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    label_names: dict[int, str],
) -> dict[str, Any]:
    """Оценивает модель и возвращает значение функции потерь вместе с метриками"""

    model.eval()
    preds: list[int] = []
    labels: list[int] = []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        output = model(**batch)
        total_loss += float(output.loss)
        n_batches += 1
        preds.extend(output.logits.argmax(dim=-1).cpu().numpy().tolist())
        labels.extend(batch["labels"].cpu().numpy().tolist())

    metrics = calculate_metrics(labels, preds, label_names)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


@torch.no_grad()
def benchmark_inference(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    runs: int,
) -> dict[str, Any]:
    """Измеряет скорость инференса модели и возвращает лучший прогон"""

    model.eval()
    for batch in loader:
        model(**move_batch_to_device(batch, device))
        break

    times: list[float] = []
    samples_per_run = 0
    for run_idx in range(runs):
        run_samples = 0
        start = time.perf_counter()
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            model(**batch)
            run_samples += batch["labels"].shape[0]
        times.append(time.perf_counter() - start)
        if run_idx == 0:
            samples_per_run = run_samples

    best = min(times)
    return {
        "engine": "pytorch",
        "device": str(device),
        "samples": samples_per_run,
        "runs": runs,
        "total_sec": best,
        "samples_per_sec": samples_per_run / best,
        "ms_per_sample": (best / samples_per_run) * 1000,
    }


def build_loader(
    df: pd.DataFrame,
    tokenizer: Any,
    max_len: int,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Создает DataLoader для датафрейма с тональностью"""

    return DataLoader(
        TextDataset(df, tokenizer, max_len),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: AppConfig,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, list[dict[str, Any]]]:
    """Обучает классификатор и собирает историю валидации"""

    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.name,
        num_labels=len(config.evaluation.labels),
    )
    model.to(device)

    class_weights = compute_class_weight(
        "balanced",
        classes=np.array(sorted(config.evaluation.labels)),
        y=train_df["sentiment"].values,
    )
    weight_tensor = torch.tensor(class_weights, dtype=torch.float, device=device)

    train_loader = build_loader(
        train_df,
        tokenizer,
        config.training.max_len,
        config.training.batch_size,
        shuffle=True,
    )
    val_loader = build_loader(
        val_df,
        tokenizer,
        config.training.max_len,
        config.training.batch_size,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    total_steps = len(train_loader) * config.training.epochs
    warmup_steps = int(total_steps * config.training.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history: list[dict[str, Any]] = []
    log_path = config.results.dir / config.results.training_log
    for epoch in range(1, config.training.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            labels = batch.pop("labels")
            output = model(**batch)
            loss = torch.nn.functional.cross_entropy(
                output.logits, labels, weight=weight_tensor
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_loss += float(loss.detach())

        val_metrics = evaluate(model, val_loader, device, config.evaluation.labels)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_loader),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_f1_weighted": val_metrics["f1_weighted"],
        }
        history.append(row)
        print(row, flush=True)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(str(row) + "\n")

    return model, tokenizer, history


def optimize_pytorch_cpu(model: torch.nn.Module) -> torch.nn.Module:
    """Применяет динамическую INT8 квантизацию Linear слоев для CPU инференса"""

    if "qnnpack" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "qnnpack"
    elif "fbgemm" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "fbgemm"
    return torch.quantization.quantize_dynamic(
        model.cpu().eval(),
        {torch.nn.Linear},
        dtype=torch.qint8,
    )


def export_onnx(
    model: torch.nn.Module,
    tokenizer: Any,
    onnx_path: Path,
    max_len: int,
) -> tuple[Path, Path | None]:
    """Экспортирует модель в ONNX и пробует создать динамическую INT8 версию"""

    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    class OnnxWrapper(torch.nn.Module):
        """Возвращает только logits для ONNX экспорта"""

        def __init__(self, wrapped_model: torch.nn.Module) -> None:
            super().__init__()
            self.wrapped_model = wrapped_model

        def forward(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor
        ) -> torch.Tensor:
            """Запускает обернутую модель с входами для ONNX"""

            return self.wrapped_model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits

    wrapper = OnnxWrapper(model.eval().to("cpu"))
    dummy = tokenizer(
        "пример текста", return_tensors="pt", truncation=True, max_length=max_len
    )
    input_names = ["input_ids", "attention_mask"]
    dynamic_axes = {name: {0: "batch"} for name in input_names}
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (dummy["input_ids"], dummy["attention_mask"]),
        onnx_path,
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes={**dynamic_axes, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx_path)

    quant_path = onnx_path.with_name(f"{onnx_path.stem}_int8.onnx")
    try:
        quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QUInt8)
    except Exception as exc:
        print("onnx int8 quantization skipped:", exc)
        quant_path = None
    return onnx_path, quant_path


def benchmark_onnx(
    onnx_path: Path,
    texts: list[str],
    tokenizer: Any,
    max_len: int,
    batch_size: int,
) -> dict[str, Any]:
    """Измеряет скорость ONNX Runtime модели на CPU"""

    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def run_batch(batch_texts: list[str]) -> list[np.ndarray]:
        """Токенизирует и запускает один ONNX batch"""

        encoded = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="np",
        )
        feed = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        return session.run(None, feed)

    run_batch(texts[:batch_size])
    start = time.perf_counter()
    n_samples = 0
    for start_idx in range(0, len(texts), batch_size):
        run_batch(texts[start_idx : start_idx + batch_size])
        n_samples += min(batch_size, len(texts) - start_idx)
    elapsed = time.perf_counter() - start
    return {
        "engine": "onnxruntime",
        "path": str(onnx_path),
        "samples": n_samples,
        "total_sec": elapsed,
        "samples_per_sec": n_samples / elapsed,
        "ms_per_sample": (elapsed / n_samples) * 1000,
    }


def prepare_data(config: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Читает CSV файлы и создает train, validation и test выборки"""

    train_full = pd.read_csv(config.data.train_path)
    val_full = pd.read_csv(config.data.validation_path)
    train_df, test_df = train_test_split(
        train_full,
        test_size=config.data.test_size,
        random_state=config.training.seed,
        stratify=train_full["sentiment"],
    )
    train_df = sample_balanced(train_df, config.data.train_max, config.training.seed)
    val_df = sample_balanced(val_full, config.data.validation_max, config.training.seed)
    return train_df, val_df, test_df


def write_metrics(
    config: AppConfig,
    metrics: dict[str, Any],
    benchmark_rows: list[dict[str, Any]],
) -> None:
    """Сохраняет метрики в JSON и бенчмарк в CSV"""

    metrics_path = config.results.dir / config.results.metrics_json
    benchmark_path = config.results.dir / config.results.benchmark_csv
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(
            metrics, metrics_file, ensure_ascii=False, indent=2, default=to_builtin
        )
    pd.DataFrame(benchmark_rows).to_csv(benchmark_path, index=False)


def compact_test_metrics(test_metrics: dict[str, Any]) -> dict[str, Any]:
    """Оставляет детальные test метрики без дублирующей текстовой таблицы"""

    return {
        key: value
        for key, value in test_metrics.items()
        if key != "classification_report"
    }


def benchmark_all(
    config: AppConfig,
    test_df: pd.DataFrame,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Запускает бенчмарки PyTorch, квантизованного PyTorch, коротких последовательностей и ONNX"""

    bench_rows: list[dict[str, Any]] = []
    sample_df = test_df.sample(
        min(config.evaluation.benchmark_max_samples, len(test_df)),
        random_state=config.training.seed,
    )
    bench_loader = build_loader(
        sample_df, tokenizer, config.training.max_len, config.training.batch_size
    )

    for dev_name in [
        "cpu",
        "mps" if torch.backends.mps.is_available() else None,
        "cuda" if torch.cuda.is_available() else None,
    ]:
        if dev_name is None:
            continue
        device = torch.device(dev_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model.output_dir
        ).to(device)
        row = benchmark_inference(
            model, bench_loader, device, config.evaluation.benchmark_runs
        )
        bench_rows.append(row)
        print("speed", row)

    base_cpu = (
        AutoModelForSequenceClassification.from_pretrained(config.model.output_dir)
        .cpu()
        .eval()
    )
    quant_cpu = optimize_pytorch_cpu(base_cpu)
    torch.save(
        quant_cpu.state_dict(), config.model.output_dir / "pytorch_int8_state.pt"
    )

    row_quant = benchmark_inference(
        quant_cpu, bench_loader, torch.device("cpu"), config.evaluation.benchmark_runs
    )
    row_quant["engine"] = "pytorch_dynamic_int8"
    bench_rows.append(row_quant)

    short_loader = build_loader(
        sample_df,
        tokenizer,
        config.evaluation.short_max_len,
        config.training.batch_size,
    )
    row_short = benchmark_inference(
        base_cpu, short_loader, torch.device("cpu"), config.evaluation.benchmark_runs
    )
    row_short["note"] = f"max_len={config.evaluation.short_max_len}"
    bench_rows.append(row_short)

    try:
        onnx_fp, onnx_int8 = export_onnx(
            AutoModelForSequenceClassification.from_pretrained(config.model.output_dir),
            tokenizer,
            config.model.output_dir / "model.onnx",
            config.training.max_len,
        )
        sample_texts = sample_df["text"].astype(str).tolist()
        bench_rows.append(
            benchmark_onnx(
                onnx_fp,
                sample_texts,
                tokenizer,
                config.training.max_len,
                config.training.batch_size,
            )
        )
        if onnx_int8 is not None:
            bench_rows.append(
                benchmark_onnx(
                    onnx_int8,
                    sample_texts,
                    tokenizer,
                    config.training.max_len,
                    config.training.batch_size,
                )
            )
    except Exception as exc:
        print("onnx export/benchmark skipped:", exc)
    return bench_rows


def main() -> None:
    """Запускает полный пайплайн обучения, оценки, бенчмарка и экспорта"""

    config = load_config()
    set_seed(config.training.seed)
    config.model.output_dir.mkdir(parents=True, exist_ok=True)
    config.results.dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = prepare_data(config)
    print(
        "split sizes:",
        "train",
        len(train_df),
        "val",
        len(val_df),
        "test",
        len(test_df),
        flush=True,
    )

    log_path = config.results.dir / config.results.training_log
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(
            f"train={len(train_df)} val={len(val_df)} test={len(test_df)} "
            f"epochs={config.training.epochs} batch={config.training.batch_size}\n"
        )

    train_device = get_device()
    print("training device:", train_device)
    model, tokenizer, history = train_model(train_df, val_df, config, train_device)

    model.save_pretrained(config.model.output_dir)
    tokenizer.save_pretrained(config.model.output_dir)

    test_loader = build_loader(
        test_df, tokenizer, config.training.max_len, config.training.batch_size
    )
    test_metrics = evaluate(
        model.to(train_device), test_loader, train_device, config.evaluation.labels
    )
    print("TEST metrics:")
    print(test_metrics["classification_report"])

    benchmark_rows = benchmark_all(config, test_df, tokenizer)
    metrics = {
        "model": config.model.name,
        "train_max": config.data.train_max,
        "val_max": config.data.validation_max,
        "epochs": config.training.epochs,
        "batch_size": config.training.batch_size,
        "lr": config.training.learning_rate,
        "max_len": config.training.max_len,
        "labels": config.evaluation.labels,
        "history": history,
        "test": compact_test_metrics(test_metrics),
        "benchmark": benchmark_rows,
    }
    write_metrics(config, metrics, benchmark_rows)
    print("saved:", config.model.output_dir, config.results.dir)


if __name__ == "__main__":
    main()
