import math
import numpy as np

class Perceptron:

    def __init__(self, inputs: int, alpha: float):
        self.weights = np.zeros(inputs)
        self.alpha = alpha

    def train(self, inputs, results, niters=10):
        for n in range(niters):
            for i, input in enumerate(inputs):
                result = self.predict(input)
                expected = results[i]
                self.weights = self.weights + self.alpha * (expected - result) * input

    def activate(self, value):
        return np.tanh(value)

    def predict(self, input):
        return self.activate(np.sum(self.weights * input))
