import matplotlib.pyplot as plt
import numpy as np


def make_prediction(input_vector, weights, bias):
    layer_1 = np.dot(input_vector, weights) + bias
    layer_2 = sigmoid(layer_1)
    prediction = layer_2
    return prediction


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


def calc_prediction_error(prediction, target):
    x = prediction - target
    mse = np.square(x)  # can be writtes as x ** 2
    return mse


def calc_prediction_error_derivative(prediction, target):
    return 2 * (prediction - target)


def update_weights(weights, derivative):
    return weights - derivative


def print_result(prediction, target, error):
    if (int(prediction + 0.5) == target):
        postfix = " ### correct"
    else:
        postfix = ""
    print(f"The prediction is: {round(prediction,4)}, target={target} error={round(error,4)} {postfix}")


# data to learn from
input_vectors = np.array(
    [
        [3, 1.7],
        [2, 1],
        [4, 1.5],
        [3, 4],
        [3.5, 0.5],
        [2, 0.5],
        [5.5, 1],
        [1, 1],
    ]
)
targets = np.array([1, 0, 1, 1, 1, 0, 1, 0])

# setup
weights = np.array([np.random.randn(), np.random.randn()])
bias = np.random.randn()

learning_rate = 0.1

iterations = 10000
training_errors = []

for iteration in range(iterations):
    print(f"Iteration {iteration}")
    training_errors_of_iteration = []
    for i in range(len(input_vectors)):

        layer_1 = np.dot(input_vectors[i], weights) + bias
        layer_2 = sigmoid(layer_1)
        prediction = layer_2

        prediction_error = calc_prediction_error(prediction, targets[i])
        print_result(prediction, targets[i], prediction_error)
        training_errors_of_iteration += [prediction_error]

        derror_dprediction = calc_prediction_error_derivative(prediction, targets[i])
        dprediction_dlayer1 = layer_2 * (1 - layer_2)  # take sigmoid result from prediction = sigmoid_derivative(layer_2)

        dlayer1_dweights = input_vectors[i]

        dlayer_dbias = 1  # derivative of x+b = 1

        derror_dweights = derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        derror_dbias = derror_dprediction * dprediction_dlayer1 * dlayer_dbias

        weights = weights - (derror_dweights * learning_rate)
        bias = bias - (derror_dbias * learning_rate)

    training_errors += [np.average(training_errors_of_iteration)]

new_input_vectors = np.array(
    [
        [0.12, 0.76],
        [1.44, 0.05],
        [3.44, 1.05],
        [2.5, 2.42]
    ]
)
new_targets = np.array([0, 0, 1, 1])

print("### NEW VECTORS ###")
for i in range(len(new_input_vectors)):
    prediction = make_prediction(new_input_vectors[i], weights, bias)
    prediction_error = calc_prediction_error(prediction, new_targets[i])
    print_result(prediction, new_targets[i], prediction_error)

print("### NETWORK ###")
print(f"weights={weights}")
print(f"bias={bias}")

plt.plot(training_errors)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
# plt.yscale('log')
# plt.savefig("cumulative_error.png")
plt.show()
plt.close()
