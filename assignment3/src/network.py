from typing import Callable, List, Tuple
from numpy import ndarray as array, tanh, dot
import numpy as np

from .data import Data
from .layer import Layer


class Network:

    N: int
    K: int
    learning_rate: float or Callable[[int], float]
    layers: List[Layer]
    gradient_v: bool
    lru: str # "epoch" or "timestep"

    def __init__(self, N: int, K: int = 2, learning_rate: float or Callable[[int], float] = 0.05, init_v=1.0, gradient_v=False, lru="epoch"):
        self.N = N
        self.K = K
        self.learning_rate = learning_rate
        self.layers = [Layer(N, init_v) for _ in range(K)]
        self.gradient_v = gradient_v
        self.lru = lru

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

        t = 1
        for epoch in range(t_max):
            update_errors(epoch)

            indexes = np.arange(train.size)
            np.random.shuffle(indexes)

            if self.lru == "epoch":
                learning_rate = self.learning_rate if type(
                    self.learning_rate) is float else self.learning_rate(epoch + 1)

            for i in indexes:
                x, y = train[i]
                delta = self.output(x) - y

                gradients_w = []
                gradients_v = []

                for layer in self.layers:
                    # derivative of the layer's weights implication
                    g_prime = 1.0 - (tanh(dot(layer.weights, x)) ** 2)

                    # computing gradient w.r.t. weights
                    gradient_w = delta * layer.v * g_prime * x
                    gradients_w.append(gradient_w)

                    # computing gradient w.r.t. units weights
                    if self.gradient_v:
                        gradient_v = delta * np.tanh(np.dot(layer.weights, x))
                        gradients_v.append(gradient_v)
                    else:
                        gradients_v.append(0.0)

                if self.lru == "timestep":
                    learning_rate = self.learning_rate if type(
                        self.learning_rate) is float else self.learning_rate(t)

                for gradient_w, gradient_v, layer in zip(gradients_w, gradients_v, self.layers):
                    layer.weights -= learning_rate * gradient_w
                    layer.v -= learning_rate * gradient_v
                
                t += 1

        update_errors(t_max)

        return train_errors, test_errors
