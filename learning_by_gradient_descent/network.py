import numpy as np


class Network:

    K: int
    N: int

    v: np.ndarray
    w: np.ndarray
    alpha: float

    c: float = 0.0

    def __init__(self, K: int, N: int, alpha: float = 0.05) -> None:
        self.N = N
        self.K = K
        self.alpha = alpha

        # TODO Initialize the weights as independent random vectors with |w1|^2 = 1 and |w2|^2 = 1.
        self.w = np.random.random((K, N))
        self.v = np.ones(K)

    def output(self, x: np.ndarray) -> float:
        return np.sum([self.v[k] * np.tanh(self.w[k], x) for k in range(self.K)])

    def cost(self, X: np.ndarray, Y: np.ndarray) -> float:
        P = len(X)
        return (1.0 / P) * np.sum([self.contribution(X[i], Y[i]) for i in range(P)])

    def contribution(self, x: np.ndarray, y: float) -> float:
        return np.pow(self.output(x) - y, 2) * 0.5

    # TODO
    def gradient(self, w: np.ndarray, e: float):
        return - self.alpha * e
