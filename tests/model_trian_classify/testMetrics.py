from model_trian_classify.metrics import calculate_metrics


def testCalculatesBinaryClassificationMetrics() -> None:
    metrics = calculate_metrics(predictions=[0, 1, 0, 1], labels=[0, 1, 1, 1])

    assert metrics["accuracy"] == 0.75
    assert metrics["per_class"]["通用知识"]["recall"] == 1.0
    assert metrics["confusion_matrix"] == [[1, 0], [1, 2]]
