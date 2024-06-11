from models.model import ClassificationModel
import numpy as np
import random


class KNeighbours(ClassificationModel):
    def __init__(self, k=5, radius=None):
        super().__init__()
        self.k = k
        self.radius = radius
        self.X_train = None
        self.y_train = None

    @staticmethod
    def euclidean_dist_square(x1, x2):
        return np.sum((x1 - x2) ** 2)

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        n = len(X)
        y_pred = []
        for x in X:
            dist = []
            for x_train, y_train in zip(self.X_train, self.y_train):
                dist.append((self.euclidean_dist_square(x, x_train), y_train))
            dist.sort()
            if self.radius is not None:
                 dist = list(filter(lambda x: x[0] <= self.radius, dist))
            if not dist:
                return [random.choice([0, 1]) for _ in range(n)]
            dist = list(map(lambda x: x[1], dist))
            class_pred = np.average(dist[:self.k], weights=list(range(n, 0, -1)))
            y_pred.append(0 if class_pred < 0.5 else 1)
        return y_pred