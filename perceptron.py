import numpy as np

class Perceptron:

    def __init__(self, N: int):
        self.N = N
        self.weights: np.ndarray = np.zeros(N)

    def train(self, X, Y, max_epoch=20) -> bool:
        """ Trains the perceptron with data X and labels Y. X must have N-dim features. 
            Returns True if the model predicts all the features perfectly """

        for _ in range(max_epoch):
            
            correct = True

            for x, y in zip(X, Y):

                e = np.dot(self.weights, x) * y

                if e > 0:
                    continue

                # order modified for np optimization
                # (a * b) * np.array better than np.array * a * b
                self.weights = self.weights + (y / self.N) * x 
                correct = False
            
            if correct:
                return True

        return False