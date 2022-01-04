from typing import Iterable, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class Data:
    """Class for simplified representation of a dataset with features X and labels Y"""

    X: np.ndarray
    Y: np.ndarray
    size: int

    @staticmethod
    def sample(X: np.ndarray, Y: np.ndarray, P=100, Q=100, shuffle=True) -> Tuple["Data", "Data"]:
        """Samples data into training an testing datasets of size P and Q. If shuffle is True, 
        the indexes used for sampling are randomly drawn inside X and Y"""

        indexes = np.arange(Y.size)
        if shuffle:
            np.random.shuffle(indexes)

        train_indexes = indexes[:P]
        test_indexes = indexes[P:P+Q]

        return (
            Data(X[train_indexes], Y[train_indexes], P),
            Data(X[test_indexes], Y[test_indexes], Q)
        )

    def __iter__(self) -> Iterable[Tuple[np.ndarray, float]]:
        return zip(self.X, self.Y)

    def __getitem__(self, i) -> Tuple[np.ndarray, float]:
        return self.X[i], self.Y[i]
