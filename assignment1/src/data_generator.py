from abc import ABC, abstractmethod
from typing import Generator, Tuple
import numpy as np
from numpy.random._generator import Generator


class DataGenerator(ABC):

    def __init__(self, N: int, seed: int = None):
        self.seed = seed
        self.N = N
        self.rnd_generator: Generator = np.random.default_rng(seed)

    def generate(self, P: int, mean: float = 0.0, variance: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generates features X of dim (P, N) and labels Y of dim (P, 1)."""

        X = self._generate(mean, variance, P)
        Y = self.rnd_generator.choice([-1, 1], P)

        return X, Y

    @abstractmethod
    def _generate(self, mean: float, variance: float, P: int) -> np.ndarray:
        pass


class NormalGenerator(DataGenerator):

    def _generate(self, mean: float, variance: float, P: int) -> np.ndarray:
        return self.rnd_generator.normal(mean, np.sqrt(variance), (P, self.N))


class UniformGenerator(DataGenerator):

    def _generate(self, mean: float, variance: float, P: int) -> np.ndarray:
        # TODO Mean and variance not yet implemented
        return self.rnd_generator.uniform(-0.5, 0.5, (P, self.N))
