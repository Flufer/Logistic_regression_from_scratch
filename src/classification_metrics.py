from typing import List

def accuracy_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Accuracy = доля правильных предсказаний
    """
    n = len(y_true)
    correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))
    return correct / n
