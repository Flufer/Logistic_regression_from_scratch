class LogisticRegression:
    """
    Логистическая регрессия с нуля
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.w = 0.0
        self.b = 0.0
