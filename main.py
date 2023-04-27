import os
from typing import Dict

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from activation_functions import tanh, tanh_prime
from activation_layer import ActivationLayer
from data_generator import DataGenerator
from dynamic_neural_network import DynamicNeuralNetwork
from fc_layer import FCLayer
from loss_functions import mse, mse_prime
from result_plot import ResultPlot


class Main:

    def __init__(self):
        self.data_generator = DataGenerator()
        self.network = None
        self.result_plotter = ResultPlot()
        self.training_input_vectors = None
        self.training_targets = None
        self.dots = None
        matplotlib.use("Agg")

    def train_network(self, network_layers, neurons_per_layer, seed, iterations, learning_rate, plot_log):
        if self.training_input_vectors is None or self.training_targets is None:
            raise gr.Error("Generate training data first")
        self.network = DynamicNeuralNetwork()
        # input layer
        self.network.add(FCLayer(2, neurons_per_layer, seed=seed))
        self.network.add(ActivationLayer(tanh, tanh_prime))
        # layers
        for i in range(network_layers):
            self.network.add(FCLayer(neurons_per_layer, neurons_per_layer, seed=seed))
            self.network.add(ActivationLayer(tanh, tanh_prime))
        # output layer
        self.network.add(FCLayer(neurons_per_layer, 1, seed=seed))
        self.network.add(ActivationLayer(tanh, tanh_prime))
        self.network.use(mse, mse_prime)
        training_errors = self.network.fit(self.training_input_vectors, self.training_targets, epochs=iterations, learning_rate=learning_rate)
        fig = plt.figure()
        plt.plot(training_errors)
        plt.xlabel("Iterations")
        plt.ylabel("Mean square error")
        if plot_log:
            plt.yscale('log')
        else:
            axis = plt.gca()
            axis.set_ylim([0, 1])
        plt.close()

        # generate result plot
        self.dots = self.result_plotter.create_dots(predict_function=self.network.predict)
        result_plot = self.result_plotter.create_plot(self.training_input_vectors.copy(), self.training_targets.copy(), dots=self.dots)

        return fig, np.amin(training_errors), result_plot

    def predict_data(self, vector_x, vector_y):
        if self.network is None:
            return "The network has to be trained first", "", None
        vector = np.array([[vector_x, vector_y]])
        prediction = self.network.predict(vector)[0][0][0]
        prediction_binary = 1 if int(prediction + 0.5) == 1 else 0

        if int(prediction + 0.5) == 1:
            certainty = prediction
        else:
            certainty = 1 - prediction
        certainty = f"{round(certainty * 100, 2)}%"

        fig = self.result_plotter.create_plot(self.training_input_vectors.copy(), self.training_targets.copy(), input_vector=vector, input_target=prediction_binary, dots=self.dots)

        return prediction, certainty, fig

    def generate_training_data(self, data_size, seed):
        self.training_input_vectors, self.training_targets = self.data_generator.generate_training_data(data_size, nested=True, value_function=self.data_generator.even_value_function, seed=seed)
        return self.result_plotter.create_plot(self.training_input_vectors.copy(), self.training_targets.copy())

    def ui(self):
        with gr.Row() as ui:
            with gr.Column():
                with gr.Tab("Training Data"):
                    num_data_size = gr.Slider(label="Amount of test data", value=30, minimum=10, maximum=200, step=1)
                    num_network_seed = gr.Number(label="Seed", value=5, precision=0)
                    btn_generate = gr.Button("Generate data")
                    plot_training_data = gr.Plot()
                    btn_generate.click(fn=self.generate_training_data, inputs=[num_data_size, num_network_seed], outputs=[plot_training_data])
            with gr.Column():
                with gr.Tab("Network"):
                    with gr.Row():
                        num_network_layers = gr.Number(label="Layers", value=1, precision=0)
                        num_neurons = gr.Number(label="Neurons per layer", value=5, precision=0)
                    num_network_seed = gr.Number(label="Seed", value=73, precision=0)
                    with gr.Row():
                        num_iterations = gr.Number(label="Iterations", value=500, precision=0)
                        num_learning_rate = gr.Number(label="Learning rate", value=0.1, precision=4)
                    chb_plot_log = gr.Checkbox(label="Plot Y axis logarithmic", value=True)
                    btn_train = gr.Button("Train network")
                    plot_mse = gr.Plot()
                    txt_mse = gr.Text(label="Minimum MSE")
            with gr.Column():
                with gr.Tab("Result"):
                    plot_result = gr.Plot()

            btn_train.click(fn=self.train_network, inputs=[num_network_layers, num_neurons, num_network_seed,
                            num_iterations, num_learning_rate, chb_plot_log], outputs=[plot_mse, txt_mse, plot_result])
#            with gr.Column():
#                with gr.Tab("Result"):
#                    with gr.Row():
#                        num_testvector_x = gr.Slider(label="Vector X value", value=2.0, minimum=0, maximum=5, step=0.1)
#                        num_testvector_y = gr.Slider(label="Vector Y value", value=1.5, minimum=0, maximum=5, step=0.1)
#                    btn_predict = gr.Button("Predict")
#                    with gr.Row():
#                        txt_prediction = gr.Text(label="Prediction")
#                        txt_certainty = gr.Text(label="Certainty")
#                    plot_visualize = gr.Plot()
#                btn_predict.click(fn=self.predict_data, inputs=[num_testvector_x, num_testvector_y], outputs=[txt_prediction, txt_certainty, plot_visualize])
#                num_testvector_x.change(fn=self.predict_data, inputs=[num_testvector_x, num_testvector_y], outputs=[txt_prediction, txt_certainty, plot_visualize])
#                num_testvector_y.change(fn=self.predict_data, inputs=[num_testvector_x, num_testvector_y], outputs=[txt_prediction, txt_certainty, plot_visualize])
        return ui


with gr.Blocks(theme=gr.themes.Default(), title="Neural Network") as demo:
    Main().ui()


if __name__ == "__main__":

    auth_file = ".auth"
    auth_function = None

    if os.path.isfile(auth_file):
        auth: Dict[str, str] = {}

        with open(auth_file, encoding="UTF-8") as f:
            lines = f.read().splitlines()
            for line in lines:
                name, pw = line.split("=")
                auth[name] = pw

        def file_auth_function(username, password):
            return username in auth and auth[username] == password

        auth_function = file_auth_function

    demo.launch(server_name="0.0.0.0", auth=auth_function)
