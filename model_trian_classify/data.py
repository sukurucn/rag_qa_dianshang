"""查询路由二分类数据集的加载、校验、去重和分层划分。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split

LABEL_TO_ID = {"通用知识": 0, "专业咨询": 1}
ID_TO_LABEL = {identifier: label for label, identifier in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class QueryExample:
    """一条经校验的二分类样本。"""

    query: str
    label_id: int

    @property
    def label(self) -> str:
        """返回原始中文标签。"""
        return ID_TO_LABEL[self.label_id]


def load_examples(path: str | Path) -> list[QueryExample]:
    """加载一行一个 JSON 对象的数据集，并移除同标签重复问题。"""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)

    examples: list[QueryExample] = []
    seen_queries: dict[str, int] = {}
    for line_number, raw_line in enumerate(data_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {data_path}:{line_number}") from error
        query = record.get("query")
        label = record.get("label")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Missing non-empty query at {data_path}:{line_number}")
        if label not in LABEL_TO_ID:
            raise ValueError(f"Unsupported label {label!r} at {data_path}:{line_number}")
        normalized_query = query.strip()
        label_id = LABEL_TO_ID[label]
        prior_label_id = seen_queries.get(normalized_query)
        if prior_label_id is not None and prior_label_id != label_id:
            raise ValueError(f"Conflicting labels for duplicate query at {data_path}:{line_number}")
        if prior_label_id is not None:
            continue
        seen_queries[normalized_query] = label_id
        examples.append(QueryExample(query=normalized_query, label_id=label_id))

    _validate_class_distribution(examples)
    return examples


def stratified_split(
    examples: list[QueryExample], test_size: float = 0.2, seed: int = 42
) -> tuple[list[QueryExample], list[QueryExample]]:
    """按类别分层固定划分 8:2 训练集和验证集。"""
    _validate_class_distribution(examples)
    labels = [example.label_id for example in examples]
    train, evaluation = train_test_split(
        examples,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    return list(train), list(evaluation)


def as_records(examples: list[QueryExample]) -> list[dict[str, int | str]]:
    """转换为 Hugging Face Dataset 所需的扁平记录。"""
    return [{"query": example.query, "labels": example.label_id} for example in examples]


def _validate_class_distribution(examples: list[QueryExample]) -> None:
    labels = {example.label_id for example in examples}
    if labels != set(ID_TO_LABEL):
        raise ValueError("The data set must include both 通用知识 and 专业咨询 labels.")
