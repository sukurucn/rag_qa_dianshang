"""四层 MiniRoBERTa 的软标签加硬标签分类蒸馏。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_trian_classify.data import ID_TO_LABEL, as_records, load_examples, stratified_split
from model_trian_classify.metrics import calculate_metrics
from model_trian_classify.runtime import configure_training_logger, cuda_log_payload, require_cuda
from model_trian_classify.teacher_trainer import LOG_DIRECTORY, TEACHER_MODEL_PATH

MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = MODULE_ROOT / "data_for_classify" / "model_generic_5000.json"
STUDENT_MODEL_ID = "uer/chinese_roberta_L-4_H-256"
BEST_MODEL_PATH = MODULE_ROOT / "model" / "best_model"


@dataclass(frozen=True)
class DistillationConfig:
    """学生模型蒸馏的可复现实验参数。"""

    data_path: Path = DEFAULT_DATA_PATH
    teacher_model_path: Path = TEACHER_MODEL_PATH
    output_path: Path = BEST_MODEL_PATH
    log_directory: Path = LOG_DIRECTORY
    student_model_id: str = STUDENT_MODEL_ID
    epochs: int = 3
    learning_rate: float = 3e-5
    batch_size: int = 32
    max_length: int = 128
    temperature: float = 2.0
    hard_loss_weight: float = 0.5
    soft_loss_weight: float = 0.5
    max_accuracy_drop: float = 0.02
    seed: int = 42


class DualTokenizerDataset:
    """同时保存教师和学生 tokenizer 输出，避免错误复用词表。"""

    def __init__(
        self,
        records: Sequence[dict[str, int | str]],
        student_tokenizer: Any,
        teacher_tokenizer: Any,
        max_length: int,
    ) -> None:
        self._records = records
        self._student_tokenizer = student_tokenizer
        self._teacher_tokenizer = teacher_tokenizer
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._records[index]
        query = str(record["query"])
        return {
            "student": self._student_tokenizer(query, truncation=True, max_length=self._max_length),
            "teacher": self._teacher_tokenizer(query, truncation=True, max_length=self._max_length),
            "labels": int(record["labels"]),
        }


class DualTokenizerCollator:
    """分别动态补齐教师和学生输入。"""

    def __init__(self, student_tokenizer: Any, teacher_tokenizer: Any) -> None:
        from transformers import DataCollatorWithPadding

        self._student_collator = DataCollatorWithPadding(tokenizer=student_tokenizer)
        self._teacher_collator = DataCollatorWithPadding(tokenizer=teacher_tokenizer)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        student_features = [feature["student"] for feature in features]
        teacher_features = [feature["teacher"] for feature in features]
        batch = self._student_collator(student_features)
        teacher_batch = self._teacher_collator(teacher_features)
        import torch

        batch["labels"] = torch.tensor([feature["labels"] for feature in features], dtype=torch.long)
        batch["teacher_input_ids"] = teacher_batch["input_ids"]
        batch["teacher_attention_mask"] = teacher_batch["attention_mask"]
        if "token_type_ids" in teacher_batch:
            batch["teacher_token_type_ids"] = teacher_batch["token_type_ids"]
        return batch


def train_student(config: DistillationConfig | None = None) -> dict[str, Any]:
    """训练学生模型，且只在准确率下降不超过 2 个百分点时保存 best_model。"""
    active_config = config or DistillationConfig()
    _validate_config(active_config)
    cuda_info = require_cuda()
    logger = configure_training_logger(active_config.log_directory)
    logger.info("Starting student distillation with CUDA: %s", cuda_log_payload(cuda_info))
    if not active_config.teacher_model_path.is_dir():
        raise FileNotFoundError(
            f"Fine-tuned teacher model was not found: {active_config.teacher_model_path}"
        )

    import numpy as np
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TrainingArguments,
    )

    teacher_tokenizer = AutoTokenizer.from_pretrained(active_config.teacher_model_path, local_files_only=True)
    teacher_model = AutoModelForSequenceClassification.from_pretrained(
        active_config.teacher_model_path,
        local_files_only=True,
    )
    student_configuration = AutoConfig.from_pretrained(active_config.student_model_id)
    if student_configuration.num_hidden_layers != 4:
        raise RuntimeError(
            f"Expected a four-layer MiniRoBERTa, got {student_configuration.num_hidden_layers} layers."
        )
    student_configuration.num_labels = len(ID_TO_LABEL)
    student_configuration.id2label = ID_TO_LABEL
    student_configuration.label2id = {label: identifier for identifier, label in ID_TO_LABEL.items()}
    student_tokenizer = AutoTokenizer.from_pretrained(active_config.student_model_id)
    student_model = AutoModelForSequenceClassification.from_pretrained(
        active_config.student_model_id,
        config=student_configuration,
        ignore_mismatched_sizes=True,
    )

    examples = load_examples(active_config.data_path)
    train_examples, evaluation_examples = stratified_split(examples, seed=active_config.seed)
    train_dataset = DualTokenizerDataset(
        as_records(train_examples), student_tokenizer, teacher_tokenizer, active_config.max_length
    )
    evaluation_dataset = DualTokenizerDataset(
        as_records(evaluation_examples), student_tokenizer, teacher_tokenizer, active_config.max_length
    )
    checkpoint_path = active_config.output_path.parent / "student_checkpoints"
    training_args = TrainingArguments(
        output_dir=str(checkpoint_path),
        num_train_epochs=active_config.epochs,
        learning_rate=active_config.learning_rate,
        per_device_train_batch_size=active_config.batch_size,
        per_device_eval_batch_size=active_config.batch_size,
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
        remove_unused_columns=False,
    )

    def compute_metrics(prediction: Any) -> dict[str, float]:
        logits, labels = prediction
        predictions = np.argmax(logits, axis=-1).tolist()
        return {
            key: value
            for key, value in calculate_metrics(predictions, labels.tolist()).items()
            if isinstance(value, float)
        }

    trainer = ClassificationDistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        temperature=active_config.temperature,
        hard_loss_weight=active_config.hard_loss_weight,
        soft_loss_weight=active_config.soft_loss_weight,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=evaluation_dataset,
        data_collator=DualTokenizerCollator(student_tokenizer, teacher_tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    student_evaluation = trainer.evaluate()
    student_metrics = _evaluate_student(trainer.model, evaluation_dataset, trainer)
    teacher_evaluation = _evaluate_teacher(teacher_model, evaluation_dataset, trainer)
    student_accuracy = float(student_metrics["accuracy"])
    teacher_accuracy = float(teacher_evaluation["accuracy"])
    accuracy_drop = teacher_accuracy - student_accuracy
    accepted = accuracy_drop <= active_config.max_accuracy_drop
    if accepted:
        active_config.output_path.mkdir(parents=True, exist_ok=True)
        trainer.save_model(active_config.output_path)
        student_tokenizer.save_pretrained(active_config.output_path)

    payload = {
        "cuda": cuda_log_payload(cuda_info),
        "teacher_accuracy": teacher_accuracy,
        "student_metrics": student_metrics,
        "student_trainer_metrics": _json_metrics(student_evaluation),
        "teacher_metrics": teacher_evaluation,
        "accuracy_drop": accuracy_drop,
        "max_accuracy_drop": active_config.max_accuracy_drop,
        "accepted": accepted,
        "model_path": str(active_config.output_path) if accepted else None,
    }
    _write_json(active_config.log_directory / "student_metrics.json", payload)
    logger.info("Student distillation completed: accepted=%s accuracy_drop=%.4f", accepted, accuracy_drop)
    return payload


class ClassificationDistillationTrainer:
    """延迟继承 Trainer，避免普通模块导入时加载 transformers。"""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from transformers import Trainer

        class _Trainer(Trainer):
            def __init__(
                self,
                *trainer_args: Any,
                teacher_model: Any,
                temperature: float,
                hard_loss_weight: float,
                soft_loss_weight: float,
                **trainer_kwargs: Any,
            ) -> None:
                super().__init__(*trainer_args, **trainer_kwargs)
                self.teacher_model = teacher_model.eval()
                self.temperature = temperature
                self.hard_loss_weight = hard_loss_weight
                self.soft_loss_weight = soft_loss_weight

            def compute_loss(
                self,
                model: Any,
                inputs: dict[str, Any],
                return_outputs: bool = False,
                **_: Any,
            ) -> Any:
                from torch.nn import functional

                teacher_inputs = {
                    "input_ids": inputs.pop("teacher_input_ids"),
                    "attention_mask": inputs.pop("teacher_attention_mask"),
                }
                teacher_token_type_ids = inputs.pop("teacher_token_type_ids", None)
                if teacher_token_type_ids is not None:
                    teacher_inputs["token_type_ids"] = teacher_token_type_ids
                labels = inputs["labels"]
                outputs = model(**inputs)
                with __import__("torch").no_grad():
                    teacher_outputs = self.teacher_model.to(model.device)(**teacher_inputs)
                hard_loss = functional.cross_entropy(outputs.logits, labels)
                soft_loss = functional.kl_div(
                    functional.log_softmax(outputs.logits / self.temperature, dim=-1),
                    functional.softmax(teacher_outputs.logits / self.temperature, dim=-1),
                    reduction="batchmean",
                ) * (self.temperature**2)
                loss = self.hard_loss_weight * hard_loss + self.soft_loss_weight * soft_loss
                return (loss, outputs) if return_outputs else loss

        return _Trainer(*args, **kwargs)


def _evaluate_teacher(teacher_model: Any, dataset: DualTokenizerDataset, trainer: Any) -> dict[str, Any]:
    import torch

    predictions: list[int] = []
    labels: list[int] = []
    device = trainer.model.device
    teacher_model.to(device).eval()
    for index in range(len(dataset)):
        item = dataset[index]
        teacher_inputs = {
            name: torch.tensor(value, device=device).unsqueeze(0)
            for name, value in item["teacher"].items()
        }
        with torch.no_grad():
            logits = teacher_model(**teacher_inputs).logits
        predictions.append(int(torch.argmax(logits, dim=-1).item()))
        labels.append(int(item["labels"]))
    return calculate_metrics(predictions, labels)


def _evaluate_student(student_model: Any, dataset: DualTokenizerDataset, trainer: Any) -> dict[str, Any]:
    import torch

    predictions: list[int] = []
    labels: list[int] = []
    device = trainer.model.device
    student_model.to(device).eval()
    for index in range(len(dataset)):
        item = dataset[index]
        student_inputs = {
            name: torch.tensor(value, device=device).unsqueeze(0)
            for name, value in item["student"].items()
        }
        with torch.no_grad():
            logits = student_model(**student_inputs).logits
        predictions.append(int(torch.argmax(logits, dim=-1).item()))
        labels.append(int(item["labels"]))
    return calculate_metrics(predictions, labels)


def _validate_config(config: DistillationConfig) -> None:
    if config.hard_loss_weight != 0.5 or config.soft_loss_weight != 0.5:
        raise ValueError("Hard and soft distillation weights must both be 0.5.")
    if config.temperature <= 0:
        raise ValueError("temperature must be positive.")
    if not 0.0 <= config.max_accuracy_drop <= 1.0:
        raise ValueError("max_accuracy_drop must be between 0 and 1.")


def _json_metrics(metrics: dict[str, Any]) -> dict[str, float | int | str]:
    return {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (float, int, str)) and not isinstance(value, bool)
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
