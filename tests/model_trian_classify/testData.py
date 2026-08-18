import json
from pathlib import Path

import pytest

from model_trian_classify.data import LABEL_TO_ID, load_examples, stratified_split


def testLoadsDeduplicatesAndStratifiesJsonlData(tmp_path: Path) -> None:
    dataset_path = tmp_path / "router.json"
    records = [
        {"query": "math", "label": "通用知识"},
        {"query": "python", "label": "通用知识"},
        {"query": "course fee", "label": "专业咨询"},
        {"query": "course location", "label": "专业咨询"},
        {"query": "math", "label": "通用知识"},
    ]
    dataset_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    examples = load_examples(dataset_path)
    train, evaluation = stratified_split(examples, test_size=0.5, seed=42)

    assert len(examples) == 4
    assert {item.label_id for item in train} == set(LABEL_TO_ID.values())
    assert {item.label_id for item in evaluation} == set(LABEL_TO_ID.values())


def testRejectsConflictingDuplicateLabels(tmp_path: Path) -> None:
    dataset_path = tmp_path / "router.json"
    records = [
        {"query": "same", "label": "通用知识"},
        {"query": "same", "label": "专业咨询"},
    ]
    dataset_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    with pytest.raises(ValueError, match="Conflicting labels"):
        load_examples(dataset_path)
