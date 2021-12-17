import numpy as np

class Classifier:

    def __init__(self, data, normalize=True):
        labels = dict()
        values = np.zeros(len(data))

        for i, x in enumerate(data):
            if not x in labels:
                labels[x] = len(labels)
            values[i] = labels[x]
        
        if normalize:
            values = values / len(labels)

        self.values = values
        self.labels = labels