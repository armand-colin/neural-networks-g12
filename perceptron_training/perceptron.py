import numpy as np

class Perceptron:

    def __init__(self, N: int, c: float = 0):
        self.N = N
        self.c = c
        self.weights: np.ndarray = np.zeros((1, N))
        self.strengths: np.ndarray = None

    def train(self, X, Y, max_epoch=20) -> bool:
        """ Trains the perceptron with data X and labels Y. X must have N-dim features. 
            Returns True if the model predicts all the features perfectly """

        self.strengths = np.zeros(len(Y))

        for _ in range(max_epoch):
            
            correct = True

            for i, (x, y) in enumerate(zip(X, Y)):

                e = np.dot(self.weights, x) * y

                if e > self.c:
                    continue

                # order modified for np optimization
                # (a * b) * np.array better than np.array * a * b
                self.weights = self.weights + (y / self.N) * x 
                correct = False
                self.strengths[i] += 1
            
            if correct:
                return True

        return False