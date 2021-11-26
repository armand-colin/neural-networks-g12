import numpy as np
import pandas as pd

from numpy.random import Generator


class DataGenerator:

    def __init__(self, n_samples: int,
                       samples_dimension: int,
                       seed: int = None) -> None:

        self.seed: int = seed
        self.n_samples: int = n_samples
        self.samples_dimension: int = samples_dimension
        self.rnd_generator: Generator = self.__get_random_generator(self.seed)

    
    def __get_random_generator(self, seed: int = None) -> Generator:
        return np.random.default_rng(seed)


    def generate(self, n_samples: int = None,
                       samples_dimension: int = None,
                       seed: int = None,
                       mean: float = 0.0,
                       variance: float = 1.0,
                       ) -> pd.DataFrame:

        if n_samples is None: n_samples = self.n_samples
        if samples_dimension is None: samples_dimension = self.samples_dimension

        rng = self.rnd_generator if seed == self.seed else self.__get_random_generator(seed)

        data = { 'x': [], 'y': [] }

        for _ in range(n_samples):
            data['x'].append(rng.normal(mean, np.sqrt(variance), samples_dimension))
            data['y'].append(-1 if rng.random() < 0.5 else 1)

        return pd.DataFrame(data)


if __name__ == '__main__':
    # Usage examples
    data_generator = DataGenerator(n_samples=10, samples_dimension=2)
    
    # n_samples=10, samples_dimension=2, random seed
    print(data_generator.generate())
    # n_samples=10, samples_dimension=2, seed=42
    print(data_generator.generate(seed=42))
    # n_samples=5, samples_dimension=2, random seed
    print(data_generator.generate(n_samples=5))
    # n_samples=7, samples_dimension=3, random seed
    print(data_generator.generate(n_samples=7, samples_dimension=3))