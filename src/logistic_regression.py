import numpy as np

class LogisticRegression:
    """
    Логистическая регрессия с градиентным спуском
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.loss_history: list[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function

        σ(z) = 1 / (1 + e^(-z))
        """
        return 1 / (1 + np.exp(-z))

    def _binary_cross_entropy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Binary Cross-Entropy (Log Loss)

        L = -1/N * Σ [ y*log(p) + (1-y)*log(1-p) ]
        """

        eps = 1e-15    # защита от log(0)
        y_pred = np.clip(y_pred, eps, 1 - eps)

        loss = -np.mean(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )
        return loss

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Обучение модели с помощью градиентного спуска
        """
        n_samples, n_features = X.shape

        # инициализация параметров
        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.n_iterations):
            # линейная комбинация
            linear_output = np.dot(X, self.w) + self.b

            # sigmoid
            y_pred = self._sigmoid(linear_output)

            # градиенты
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # обновление параметров
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            # логирование ошибки
            loss = self._binary_cross_entropy(y, y_pred)
            self.loss_history.append(loss)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает вероятность принадлежности к классу 1
        """
        linear_output = np.dot(X, self.w) + self.b
        return self._sigmoid(linear_output)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Возвращает бинарные предсказания (0 или 1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def get_params(self) -> dict:
        """
        Возвращает параметры модели
        """
        return {'w', self.w, 'b', self.b}
    