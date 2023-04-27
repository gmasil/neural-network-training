import numpy as np

from layer import Layer


class FCLayer(Layer):
    def __init__(self, input_size, output_size, seed=-1):
        super().__init__()
        if seed == -1:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(12345)
        self.weights = self.rng.random((input_size, output_size)) - 0.5
        self.bias = self.rng.random((1, output_size)) - 0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
