from typing import List, Tuple
from numpy import ndarray as array, tanh, dot
import numpy as np

from .data import Data
from .layer import Layer


class Network:

    N: int
    K: int
    learning_rate: float
    layers: List[Layer]

    def __init__(self, N: int, K: int = 2, learning_rate=0.05):
        self.N = N
        self.K = K
        self.learning_rate = learning_rate
        self.layers = [Layer(N) for _ in range(K)]

    def output(self, x: array) -> float:
        return sum(layer.v * tanh(dot(layer.weights, x)) for layer in self.layers)

    def error(self, data: Data) -> float:
        errors = np.array([0.5 * (self.output(x) - y) ** 2 for x, y in data])
        return np.mean(errors)

    def train(self, train: Data, test: Data, t_max=100) -> Tuple[array, array]:
        """Trains the network with the given train dataset"""

        train_errors = np.zeros(t_max + 1)
        test_errors = np.zeros(t_max + 1)

        def update_errors(i):
            train_errors[i] = self.error(train)
            test_errors[i] = self.error(test)

        for epoch in range(t_max):
            update_errors(epoch)

            indexes = np.arange(train.size)
            np.random.shuffle(indexes)

            for i in indexes:
                x, y = train[i]
                delta = self.output(x) - y

                for layer in self.layers:
                    # derivative of the layer's weights implication
                    g_prime = 1.0 - (tanh(dot(layer.weights, x)) ** 2)

                    # updating the weights
                    gradient = delta * layer.v * g_prime * x
                    layer.weights -= self.learning_rate * gradient

        update_errors(t_max)

        return train_errors, test_errors
