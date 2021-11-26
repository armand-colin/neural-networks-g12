import numpy as np

class Perceptron:

    def __init__(self, N: int, alpha: float):
        self.N = N
        self.weights = np.zeros(N)
        self.alpha = alpha

    def train(self, X, Y, epochs=20):
        """Trains the perceptron with data X and labels Y. X must have N-dim features."""
        
        for _ in range(epochs):

            correct = True
            for x, y in zip(X, Y):
                
                e = self.weights * x * y

                if e > 0:
                    continue

                # order modified for np optimization
                # (a * b) * np.array better than np.array * a * b
                self.weights = self.weights + (y / self.N) * x 
                correct = False
            
            if correct:
                break
