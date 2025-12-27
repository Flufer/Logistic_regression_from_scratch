from typing import List

def accuracy_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Accuracy = доля правильных предсказаний
    """
    n = len(y_true)
    correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))
    return correct / n

def precision_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Precision = TP (TP + FP)
    """
    tp = sum(yt == yp == 1 for yt, yp in zip(y_true, y_pred))
    fp = sum(yt == 0 and yp == 1 for yt, yp in zip(y_true, y_pred))

    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

def recall_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Recall = TP (TP + FN)
    """
    tp = sum(yt == yp == 1 for yt, yp in zip(y_true, y_pred))
    fn = sum(yt == 1 and yp == 0 for yt, yp in zip(y_true, y_pred))

    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)

def f1_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    F1 = 2 * (precision * recall) / (precision + recall)
    """
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)

    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)
