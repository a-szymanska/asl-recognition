from models.model import ClassificationModel
import numpy as np
import random
from collections import Counter, deque


class Node:
    def __init__(self, data=None, children=None, leaf=False, split_attr=None, split_val=None, class_pred=None):
        if children is None:
            children = [None, None]
        self.data = data
        self.left, self.right = children
        self.leaf = leaf
        self.split_attr = split_attr
        self.split_val = split_val
        self.class_pred = class_pred


class DecisionTree(ClassificationModel):
    def __init__(self, min_split_sample=20):
        super().__init__()
        self.min_split_sample = min_split_sample
        self.root = None

    def fit(self, X, Y):
        Y = np.reshape(Y, (len(Y), 1))
        data = np.hstack([X, Y])
        self.root = Node(data=data)
        self.split()

    @staticmethod
    def get_cost(Y):
        n = len(Y)
        if n == 0:
            return 4
        counts = dict(Counter(Y))
        value_counts = list(counts.values())
        entropy = 1 - max(value_counts) / n
        return entropy

    @staticmethod
    def get_pred_class(Y):
        if len(Y) == 0:
            return random.choice([0, 1])
        counts = Counter(Y)
        class_pred = counts[1] / (counts[0] + counts[1])  # max(counts, key=counts.get)
        return class_pred

    def split(self):
        stack = deque([self.root])
        while stack:
            node = stack.pop()
            data = node.data
            y = data[:, -1]
            if self.get_cost(y) <= 0 or len(data) <= self.min_split_sample:
                node.leaf = True
                node.class_pred = self.get_pred_class(y)
                continue
            min_cost = None
            n = len(data[0])

            for i in range(n - 1):
                cost, val = self.split_by_attr(data, i)
                if min_cost is None or cost < min_cost:
                    attr = i
                    min_val = val
                    min_cost = cost

            node.left = Node(data=data[data[:, attr] <= min_val])
            node.right = Node(data=data[data[:, attr] > min_val])
            node.split_attr, node.split_val = attr, min_val
            stack.extend([node.left, node.right])

    def split_by_attr(self, data, attr):
        values = np.unique(data[:, attr])
        min_cost, min_val = None, None
        for val in values:
            data_l = data[data[:, attr] <= val]
            data_r = data[data[:, attr] > val]
            cost = self.get_cost(data_l[:, -1]) + self.get_cost(data_r[:, -1])
            if min_cost is None or cost < min_cost:
                min_cost, min_val = cost, val
        return min_cost, min_val

    def predict(self, X):
        n = len(X)
        y_pred = np.empty(n)
        for i in range(n):
            x = X[i]
            node = self.root
            while not node.leaf:
                attr, val = node.split_attr, node.split_val
                if x[attr] <= val:
                    node = node.left
                else:
                    node = node.right
            y_pred[i] = node.class_pred
        return y_pred
