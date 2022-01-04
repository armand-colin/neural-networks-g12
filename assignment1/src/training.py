from dataclasses import dataclass, field
import numpy as np
import multiprocessing
from .data_generator import DataGenerator, NormalGenerator
from .perceptron import Perceptron


@dataclass
class TrainParameters:
    nd: int
    nmax: int
    N: int
    P: int = None
    c: float = 0
    generator: DataGenerator = field(default_factory=lambda: NormalGenerator(0))

    def clone(self) -> "TrainParameters":
        return TrainParameters(**self.__dict__)

@dataclass
class TrainResults:
    success_count: int
    embedding_strengths: np.ndarray


def train(parameters: TrainParameters, result: TrainResults) -> TrainResults:
    embedding_strengths = np.zeros(parameters.P)
    success_count = 0

    for _ in range(parameters.nd):
        perceptron = Perceptron(parameters.N, parameters.c)
        X, Y = parameters.generator.generate(parameters.P)

        if perceptron.train(X, Y, parameters.nmax):
            success_count += 1

        embedding_strengths += perceptron.strengths

    result.success_count += success_count
    result.embedding_strengths += embedding_strengths

    return result


class Trainer:

    @staticmethod
    def train(parameters: TrainParameters, thread_count: int = 1) -> TrainResults:
        
        if parameters.generator is None:
            parameters.generator = NormalGenerator(parameters.N)

        result = TrainResults(0, np.zeros(parameters.P))

        if thread_count == 1:
            return train(parameters, result)

        processes: list[multiprocessing.Process] = []

        for i in range(thread_count):
            thread_parameters = parameters.clone()

            thread_parameters.nd = parameters.nd // thread_count
            if i == 0:
                thread_parameters.nd += parameters.nd % thread_count

            process = multiprocessing.Process(
                target=train, 
                args=(thread_parameters, result)
            )

            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        return result

    @staticmethod
    def train_range(parameters: TrainParameters, alpha_range: np.ndarray, thread_count = 1) -> np.ndarray:
        results = np.zeros(alpha_range.size)
        
        for i, alpha in enumerate(alpha_range):
            _parameters = parameters.clone()
            _parameters.P = int(parameters.N * alpha)

            result = Trainer.train(_parameters, thread_count=thread_count)
            results[i] = result.success_count / parameters.nd
        
        return results