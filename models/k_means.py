import numpy as np
import random


# TODO Make this model better

class KMeans:
    def __init__(self, n_clusters=5, n_iterations=100000):  # n_iterations - max number of iterations
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.centroids = []

    def k_means_plus(self, X):
        n = len(X)
        self.centroids = [random.choice(X)]
        for _ in range(self.n_clusters - 1):
            distances = np.zeros(n)
            for centroid in self.centroids:
                distances += self.euclidean_dist(centroid, X)
            closeness = distances / np.sum(distances)  # Normalize the distances
            new_centroid_idx = np.argmax(np.random.multinomial(1, closeness))

            self.centroids.append(X[new_centroid_idx])

    def fit(self, X):
        self.k_means_plus(X)
        i = 0
        prev_centroids = None
        while i < self.n_iterations and np.not_equal(self.centroids, prev_centroids).any():
            i += 1
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in X:
                centroid_idx = np.argmin(self.euclidean_dist(x, self.centroids))
                sorted_points[centroid_idx].append(x)
            new_centroids = []
            for cluster in sorted_points:
                new_centroids.append(np.mean(cluster, axis=0))
            prev_centroids, self.centroids = self.centroids, new_centroids
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():
                    self.centroids[i] = prev_centroids[i]

    def predict(self, X):
        n = len(X)
        pred_centroids = np.zeros((n, len(X[0])))
        pred_centroid_idxs = np.zeros(n, dtype=int)
        for i, x in enumerate(X):
            min_idx = np.argmin(self.euclidean_dist(x, self.centroids))
            pred_centroid_idxs[i] = min_idx
            pred_centroids[i] = self.centroids[min_idx]
        return pred_centroids, pred_centroid_idxs

    @staticmethod
    def euclidean_dist(point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2, axis=1))
