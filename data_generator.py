from typing import Tuple
import numpy as np

from numpy.random import Generator


class DataGenerator:

    def __init__(self, N: int, seed: int = None):
        self.seed = seed
        self.N = N
        self.rnd_generator = self.__get_random_generator(self.seed)

    def __get_random_generator(self, seed: int = None) -> Generator:
        return np.random.default_rng(seed)

    def generate(self, P: int,
                       mean: float = 0.0,
                       variance: float = 1.0,
                       clamp: Tuple[float, float] = (0.0, 1.0)
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates features X of dim (P, N) and labels Y of dim (P, 1)."""

        rng = self.rnd_generator

        X = np.zeros((P, self.N))
        Y = np.zeros(P)

        for i in range(P):
            X[i] = rng.normal(mean, np.sqrt(variance), self.N)
            Y[i] = -1 if rng.random() < 0.5 else 1

        np.clip(X, clamp[0], clamp[1])

        return X, Y