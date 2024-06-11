from models.model import ClassificationModel
from sklearn.tree import DecisionTreeClassifier
import numpy as np


class RandomForest(ClassificationModel):
    def __init__(self, n_trees=500, sample_size=20000, min_split_sample=20):
        super().__init__()
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.min_split_sample = min_split_sample
        self.trees = [DecisionTreeClassifier() for _ in range(n_trees)]

    def fit(self, X, Y):
        Y = np.reshape(Y, (len(Y), 1))
        data = np.hstack([X, Y])
        id = 0
        for tree in self.trees:
            print('training', id, 1000 * (id % 5), 1000 * (1 + id % 5))
            id += 1
            X_sample, Y_sample = X[1000 * (id % 5):1000 * (1 + id % 5)], Y[1000 * (id % 5):1000 * (1 + id % 5)]
            # sample = np.random.choice(data.shape[0], size=min(self.sample_size, len(data)), replace=False)
            # data_sample = data[sample]
            # X_sample, Y_sample = data_sample[:, :-1], np.reshape(data_sample[:, -1], (len(data_sample[:, -1]), 1))
            tree.fit(X_sample, Y_sample)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees]).T
        y_pred = np.empty(len(X))
        for i in range(len(X)):
            values, counts = np.unique(predictions[i], return_counts=True)
            pred_class = values[np.argmax(counts)]
            y_pred[i] = pred_class
        return y_pred