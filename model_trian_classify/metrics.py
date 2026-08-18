"""分类训练与蒸馏共用的可序列化指标。"""

from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from model_trian_classify.data import ID_TO_LABEL


def calculate_metrics(predictions: list[int], labels: list[int]) -> dict[str, Any]:
    """计算 accuracy、macro/weighted 指标、分类报告和混淆矩阵。"""
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have the same length.")
    if not labels:
        raise ValueError("Metrics require at least one label.")

    ordered_ids = sorted(ID_TO_LABEL)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=ordered_ids,
        zero_division=0,
    )
    report = classification_report(
        labels,
        predictions,
        labels=ordered_ids,
        target_names=[ID_TO_LABEL[item] for item in ordered_ids],
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "f1_weighted": float(report["weighted avg"]["f1-score"]),
        "per_class": {
            ID_TO_LABEL[label_id]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label_id in enumerate(ordered_ids)
        },
        "confusion_matrix": confusion_matrix(labels, predictions, labels=ordered_ids).tolist(),
    }
