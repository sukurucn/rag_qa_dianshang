"""使用 Hugging Face Trainer 微调 BERT 查询路由教师模型。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_trian_classify.data import ID_TO_LABEL, as_records, load_examples, stratified_split
from model_trian_classify.metrics import calculate_metrics
from model_trian_classify.runtime import configure_training_logger, cuda_log_payload, require_cuda

MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = MODULE_ROOT / "data_for_classify" / "model_generic_5000.json"
BASE_MODEL_PATH = MODULE_ROOT.parent / "models" / "bert-base-chinese"
TEACHER_MODEL_PATH = MODULE_ROOT / "model" / "bert_finetuning_model"
LOG_DIRECTORY = MODULE_ROOT / "logs"


@dataclass(frozen=True)
class TeacherTrainingConfig:
    """教师模型训练的可复现实验参数。"""

    data_path: Path = DEFAULT_DATA_PATH
    output_path: Path = TEACHER_MODEL_PATH
    log_directory: Path = LOG_DIRECTORY
    epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    max_length: int = 128
    seed: int = 42


def train_teacher(config: TeacherTrainingConfig | None = None) -> dict[str, Any]:
    """在 CUDA 上训练 BERT 教师模型，并保存最佳 F1 checkpoint。"""
    active_config = config or TeacherTrainingConfig()
    cuda_info = require_cuda()
    logger = configure_training_logger(active_config.log_directory)
    logger.info("Starting teacher training with CUDA: %s", cuda_log_payload(cuda_info))
    if not BASE_MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Local bert-base-chinese model was not found: {BASE_MODEL_PATH}")

    import numpy as np
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    examples = load_examples(active_config.data_path)
    train_examples, evaluation_examples = stratified_split(examples, seed=active_config.seed)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_PATH,
        num_labels=len(ID_TO_LABEL),
        id2label=ID_TO_LABEL,
        label2id={label: identifier for identifier, label in ID_TO_LABEL.items()},
        local_files_only=True,
    )
    train_dataset = _tokenize_dataset(
        Dataset.from_list(as_records(train_examples)), tokenizer, active_config.max_length
    )
    evaluation_dataset = _tokenize_dataset(
        Dataset.from_list(as_records(evaluation_examples)), tokenizer, active_config.max_length
    )
    checkpoint_path = active_config.output_path.parent / "teacher_checkpoints"
    training_args = TrainingArguments(
        output_dir=str(checkpoint_path),
        num_train_epochs=active_config.epochs,
        learning_rate=active_config.learning_rate,
        per_device_train_batch_size=active_config.batch_size,
        per_device_eval_batch_size=active_config.batch_size,
        gradient_accumulation_steps=active_config.gradient_accumulation_steps,
        weight_decay=0.01,
        warmup_steps=14,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to=[],
        seed=active_config.seed,
        data_seed=active_config.seed,
    )

    def compute_metrics(prediction: Any) -> dict[str, float]:
        logits, labels = prediction
        predictions = np.argmax(logits, axis=-1).tolist()
        return {
            key: value
            for key, value in calculate_metrics(predictions, labels.tolist()).items()
            if isinstance(value, float)
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=evaluation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    evaluation = trainer.evaluate()
    active_config.output_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(active_config.output_path)
    tokenizer.save_pretrained(active_config.output_path)

    payload = {
        "cuda": cuda_log_payload(cuda_info),
        "train_examples": len(train_examples),
        "evaluation_examples": len(evaluation_examples),
        "metrics": _json_metrics(evaluation),
        "model_path": str(active_config.output_path),
    }
    _write_json(active_config.log_directory / "teacher_metrics.json", payload)
    logger.info("Teacher training completed: %s", payload["metrics"])
    return payload


def _tokenize_dataset(dataset: Any, tokenizer: Any, max_length: int) -> Any:
    return dataset.map(
        lambda batch: tokenizer(batch["query"], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=["query"],
    )


def _json_metrics(metrics: dict[str, Any]) -> dict[str, float | int | str]:
    return {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (float, int, str)) and not isinstance(value, bool)
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
