import numpy as np


class NeuralNetwork:
    def __init__(self, learning_rate):
        self.weights = np.array([np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        return prediction

    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        prediction_error = np.square(prediction - target)

        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = layer_2 * (1 - layer_2)
        dlayer1_dweights = input_vector

        derror_dweights = derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        derror_dbias = derror_dprediction * dprediction_dlayer1

        return prediction_error, derror_dweights, derror_dbias

    def _update_parameters(self, derror_dweights, derror_dbias):
        self.weights = self.weights - (derror_dweights * self.learning_rate)
        self.bias = self.bias - (derror_dbias * self.learning_rate)

    def train(self, input_vectors, targets, iterations):
        training_errors = []

        for i in range(iterations):
            # print(f"iteration {i}")
            training_errors_of_iteration = []

            for i in range(len(input_vectors)):
                prediction_error, derror_dweights, derror_dbias = self._compute_gradients(input_vectors[i], targets[i])
                training_errors_of_iteration += [prediction_error]
                self._update_parameters(derror_dweights, derror_dbias)

            training_errors += [np.average(training_errors_of_iteration)]

        return training_errors
