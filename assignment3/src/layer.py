from numpy import ndarray as array, tanh, dot
import numpy as np


class Layer:

    N: int
    v: float
    weights: array

    def __init__(self, N: int, v=1.0):
        self.N = N
        self.v = v
        
        # initial random weight values
        self.weights = np.random.uniform(size=N)
        
        # weights normalization
        weights_length = np.sqrt(np.sum(self.weights ** 2))
        self.weights /= weights_length