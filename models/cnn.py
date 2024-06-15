from models.model import ClassificationModel
import numpy as np
from scipy.signal import correlate2d


class CNN(ClassificationModel):
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    class Filter:
        def __init__(self, shape, stride=1):
            self.F = np.random.random(shape)
            self.stride = stride
            self.shape = shape

        def apply(self, M):
            return np.inner(M.ravel(), self.F.ravel())

    class Layer:
        def forward(self, input):
            pass

        def output(self, input):
            pass

    class ConvolutionLayer(Layer):
        def __init__(self, filters_shape, no_filters, activation="relu"):
            self.filters_shape = filters_shape
            self.filters = [CNN.Filter(filters_shape) for _ in range(no_filters)]
            self.no_filters = no_filters
            self.activation = getattr(CNN, activation)

        def forward(self, input):
            x, y, c = input.shape
            a, b = self.filters_shape
            x_out = x - a + 1
            y_out = y - b + 1

            # Initialize result array
            res = np.zeros((x_out, y_out, self.no_filters))

            for k, f in enumerate(self.filters):
                for channel in range(c):
                    res[:, :, k] = correlate2d(input[:, :, channel], f.F, mode="valid")
            return self.activation(res)

        def backwards(self, in_gradient):
            pass

    class MaxPoolLayer(Layer):
        def __init__(self, pool_shape):
            self.pool_shape = pool_shape

        def forward(self, input):
            x, y, c = input.shape
            a, b = self.pool_shape
            res = np.zeros((x // a, y // a, c))
            x_pooled = x // a
            y_pooled = y // b
            input_cropped = input[:x_pooled * a, :y_pooled * b, :]
            input_reshaped = input_cropped.reshape(x_pooled, a, y_pooled, b, c)
            res = input_reshaped.max(axis=(1, 3))
            return res

        def backwards(self, in_gradient):
            pass

    class DenseLayer(Layer):
        def __init__(self, output_size, activation="relu"):
            self.output_size = output_size
            self.activation = getattr(CNN, activation)
            self.weights = None
            self.bias = np.zeros((1, output_size))

        def _initialize_weights(self, in_size, out_size):
            self.weights = np.random.random((in_size, out_size))

        def forward(self, input):
            input = input.flatten()
            if self.weights is None:
                self._initialize_weights(input.shape[0], self.output_size)
            z = input @ self.weights + self.bias
            return self.activation(z)

        def backwards(self, in_gradient):
            pass

    def __init__(self):
        self.layers = []

    def add_layer(self, L):
        self.layers.append(L)

    def train_batch(self, X, Y):
        res = np.zeros_like(Y)
        for i, x in enumerate(X):
            Y_pred = self.predict(x)
            Y_pred /= np.sum(Y_pred)
            Y_true = np.zeros(26)
            Y_true[Y[i]] = 1
            loss = -np.sum(Y_true * np.log(Y_pred)) / Y.shape[0]
            gradient = -Y_true / Y_pred / Y.shape[0]
            self.back_prop(gradient)
            print(gradient)

    def fit(self, X, Y):
        for i in range(0, X.shape[0] - 16, 16):
            self.train_batch(X[i:i + 16], Y[i:i + 16])

    def back_prop(self, gradient):
        for L in reversed(self.layers):
            gradient = L.backwards(gradient)

    def predict(self, x):
        for L in self.layers:
            x = L.forward(x)
        return x
