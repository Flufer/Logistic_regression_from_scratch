import math
from typing import List

class LogisticRegression:
    """
    Логистическая регрессия с нуля
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.w = 0.0
        self.b = 0.0

    @staticmethod
    def _sigmoid(z: float) -> float:
        """
        Sigmoid activation function

        σ(z) = 1 / (1 + e^(-z))
        """
        return 1 / (1 + math.exp(-z))

    def predict_proba(self, X: List[float]) -> List[float]:
        """
        Возвращает вероятность принадлежности к классу 1
        """
        return [self._sigmoid(self.w * x + self.b) for x in X]
    