import math
from typing import List

class LogisticRegression:
    """
    Логистическая регрессия с градиентным спуском
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

    def _binary_cross_entropy(self, y_true: List[int], y_pred: List[float]) -> float:
        """
        Binary Cross-Entropy (Log Loss)

        L = -1/N * Σ [ y*log(p) + (1-y)*log(1-p) ]
        """

        eps = 1e-15    # Защита от log(0)
        n = len(y_true)

        loss = 0.0
        for i in range(n):
            p = min(max(y_pred[i], eps), 1 - eps)
            loss += y_true[i] * math.log(p) + (1 - y_true[i]) * math.log(1 - p)

        return -loss / n

    def fit(self, X: List[float], y: List[int]):
        """
        Обучение модели с помощью градиентного спуска
        """
        n = len(X)

        for _ in range(self.n_iterations):
            linear_output = [self.w * x + self.b for x in X]
            y_pred = [self._sigmoid(z) for z in linear_output]

            dw = (1 / n) * sum((y_pred[i] - y[i]) * X[i] for i in range(n))
            db = (1 / n) * sum(y_pred[i] - y[i] for i in range(n))

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

        return self

    def predict_proba(self, X: List[float]) -> List[float]:
        """
        Возвращает вероятность принадлежности к классу 1
        """
        return [self._sigmoid(self.w * x + self.b) for x in X]
