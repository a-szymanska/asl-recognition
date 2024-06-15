from models.model import ClassificationModel
import numpy as np
from scipy.signal import correlate2d,convolve2d


class CNN(ClassificationModel):
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    def relu_prime(x):
        return np.where(x > 0, 1, 0)
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
        def __init__(self, filters_shape, no_filters, activation="relu",learning_rate=0.01):
            self.filters_shape = filters_shape
            self.learining_rate=learning_rate
            self.filters = [np.random.rand(filters_shape[0],filters_shape[1]) for _ in range(no_filters)]
            self.no_filters = no_filters
            self.activation = getattr(CNN, activation)
            self.act_derivative=getattr(CNN,f'{activation}_prime')
            self.biases=None

        def _initialize_weights(self, in_size, out_size):
            self.biases = np.random.random((in_size, out_size))

        def forward(self, input):
            self.input=input
            x, y, c = input.shape
            a, b = self.filters_shape
            x_out = x - a + 1
            y_out = y - b + 1
            res = np.zeros((x_out, y_out, self.no_filters))
            if self.biases is None:
                self.biases=np.zeros((x_out, y_out, self.no_filters))
            for k, f in enumerate(self.filters):
                for channel in range(c):
                    res[:, :, k]+= correlate2d(input[:, :, channel], f, mode="valid")
            res/=np.sum(res)
            self.output=self.activation(res)
            self.before_act=res
            return self.output

        def backwards(self, in_gradient):
            in_gradient=np.reshape(in_gradient,self.before_act.shape)
            G=np.multiply(in_gradient,self.act_derivative(self.before_act))
            K_grad=np.zeros((self.no_filters,self.input.shape[2],self.filters_shape[0],self.filters_shape[1]))
            out_grad=np.zeros_like(self.input,dtype=float)
            for i in range(self.no_filters):
                for j in range(self.input.shape[2]):
                    K_grad[i,j]=correlate2d(self.input[:,:,j],G[:,:,i],"valid")
                    out_grad[:,:,j]+=convolve2d(G[:,:,i],self.filters[i],"full")
            for i in range(self.no_filters):
                for j in range(self.input.shape[2]):
                    self.filters[i]-=self.learining_rate*K_grad[i,j]
            self.biases-=self.learining_rate*G
            return out_grad

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
        def __init__(self, output_size, activation="relu",learning_rate=0.01):
            self.output_size = output_size
            self.activation = getattr(CNN, activation)
            self.act_derivative=getattr(CNN,f'{activation}_prime')
            self.weights = None
            self.bias = np.zeros((1, output_size))
            self.learning_rate=learning_rate
        def _initialize_weights(self, in_size, out_size):
            self.weights = np.random.random((out_size, in_size))

        def forward(self, input):
            input = input.flatten()[np.newaxis,:]
            
            self.input=input
            if self.weights is None:
                self._initialize_weights(input.shape[1], self.output_size)
            z =np.dot(input,self.weights.T) + self.bias
            z/=np.sum(z)
            self.before_act=z
            self.output=self.activation(z)
            return self.output

        def backwards(self, in_gradient):
            G=np.multiply(in_gradient,self.act_derivative(self.before_act))
            W_grad=np.dot(self.input.T,G)
            out_grad=np.dot(G,self.weights)
            self.weights-=self.learning_rate*W_grad.T
            self.bias-=self.learning_rate*G
            return out_grad

    def __init__(self):
        self.layers = []

    def add_layer(self, L):
        self.layers.append(L)

    def train_batch(self, X, Y):
        res = np.zeros_like(Y)
        for i, x in enumerate(X):
            for L in self.layers:
                x = L.forward(x)
            Y_pred=x
            Y_pred+=0.00001
            Y_pred /= np.sum(Y_pred)
            Y_true = np.zeros(26)
            Y_true[Y[i]] = 1
            loss = -np.sum(Y_true * np.log(Y_pred)) / Y.shape[0]
            gradient = -Y_true / (Y_pred) / Y.shape[0]
            #print(Y_true.shape,Y_pred.shape)
            print(loss)
            self.back_prop(gradient)

    def fit(self, X, Y):
        for i in range(0, X.shape[0] - 16, 16):
            self.train_batch(X[i:i + 16], Y[i:i + 16])

    def back_prop(self, gradient):
        for L in reversed(self.layers):
            gradient = L.backwards(gradient)

    def predict(self, X):
        n = len(X)
        y_pred = np.empty(n)
        for i in range(n):
            x=X[i]
            for L in self.layers:
                x = L.forward(x)
            y_pred[i]=np.argmax(x)
        return y_pred

