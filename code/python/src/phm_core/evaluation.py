"""测试误报、漏报与分层指标；阈值仅由独立健康校准集决定。"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
)


def evaluate(
    score: np.ndarray, healthy: np.ndarray, ids: np.ndarray, threshold: float
) -> dict:
    actual = ~healthy[ids]
    predicted = score[ids] > threshold
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[False, True]).ravel()
    return {
        "test_pr_auc": float(average_precision_score(actual, score[ids]))
        if len(np.unique(actual)) == 2
        else None,
        "precision": float(precision_score(actual, predicted, zero_division=0))
        if predicted.any()
        else None,
        "recall": float(recall_score(actual, predicted, zero_division=0))
        if actual.any()
        else None,
        "test_positive_cycles": int(actual.sum()),
        "test_alert_cycles": int(predicted.sum()),
        "normal_cycles": int(tn + fp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "false_positive_per_1000": float(fp / (tn + fp) * 1000) if tn + fp else None,
        "prevalence": float(actual.mean()) if len(actual) else None,
        "false_positive_ids": ids[(~actual) & predicted].tolist(),
        "false_negative_ids": ids[actual & (~predicted)].tolist(),
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
