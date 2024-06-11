from models.model import ClassificationModel
from models.decision_tree import DecisionTree
import numpy as np


class RandomForest(ClassificationModel):
    def __init__(self, n_trees=10, sample_size=100, min_split_sample=20):
        super().__init__()
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.min_split_sample = min_split_sample
        self.trees = [DecisionTree(min_split_sample=self.min_split_sample) for _ in range(n_trees)]

    def fit(self, X, Y):
        data = np.hstack([X, Y])
        print(X.shape[1])
        for tree in self.trees:
            sample = np.random.choice(data.shape[0], size=min(self.sample_size, len(data)), replace=False)
            data_sample = data[sample]
            X_sample, Y_sample = data_sample[:, :-1], np.reshape(data_sample[:, -1], (len(data_sample[:, -1]), 1))
            tree.fit(X_sample, Y_sample)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees]).T
        y_pred = np.empty(len(X))
        for i in range(len(X)):
            values, counts = np.unique(predictions[i], return_counts=True)
            pred_class = values[np.argmax(counts)]
            y_pred[i] = pred_class
        return y_pred