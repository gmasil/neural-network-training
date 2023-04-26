import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from neural_network import NeuralNetwork
from data_generator import DataGenerator
from result_plot import ResultPlot


class Main:

    def __init__(self):
        self.network = None
        self.result_plotter = ResultPlot()
        self.training_input_vectors, self.training_targets = DataGenerator().generate_training_data(50)
        self.dots = None
        matplotlib.use("Agg")

    def train_network(self, iterations, learning_rate, plot_log):
        self.training_input_vectors, self.training_targets = DataGenerator().generate_training_data(50)
        self.network = NeuralNetwork(learning_rate)
        training_errors = self.network.train(self.training_input_vectors, self.training_targets, iterations)
        self.dots = self.result_plotter.create_dots(fn=self.network.predict)
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
        return fig, np.amin(training_errors)

    def predict_data(self, vector_x, vector_y):
        if self.network is None:
            return "The network has to be trained first", "", None
        vector = np.array([vector_x, vector_y])
        prediction = self.network.predict(vector)
        prediction_binary = 1 if int(prediction + 0.5) == 1 else 0

        if int(prediction + 0.5) == 1:
            certainty = prediction
        else:
            certainty = 1 - prediction
        certainty = f"{round(certainty * 100, 2)}%"

        fig = self.result_plotter.create_plot(self.training_input_vectors.copy(), self.training_targets.copy(), vector, prediction_binary, dots=self.dots)

        return prediction, certainty, fig

    def ui(self):
        with gr.Row() as ui:
            with gr.Column():
                with gr.Tab("Training"):
                    with gr.Row():
                        num_iterations = gr.Number(label="Iterations", value=100, precision=0)
                        num_learning_rate = gr.Number(label="Learning rate", value=0.1, precision=4)
                    chb_plot_log = gr.Checkbox(label="Plot Y axis logarithmic")
                    btn_train = gr.Button("Train network")
                    plot_mse = gr.Plot()
                    txt_mse = gr.Text(label="Minimum MSE")
                btn_train.click(fn=self.train_network, inputs=[num_iterations, num_learning_rate, chb_plot_log], outputs=[plot_mse, txt_mse])
            with gr.Column():
                with gr.Tab("Prediction"):
                    with gr.Row():
                        num_testvector_x = gr.Slider(label="Vector X value", value=2.0, minimum=0, maximum=5, step=0.1)
                        num_testvector_y = gr.Slider(label="Vector Y value", value=1.5, minimum=0, maximum=5, step=0.1)
                    btn_predict = gr.Button("Predict")
                    with gr.Row():
                        txt_prediction = gr.Text(label="Prediction")
                        txt_certainty = gr.Text(label="Certainty")
                    plot_visualize = gr.Plot()
                btn_predict.click(fn=self.predict_data, inputs=[num_testvector_x, num_testvector_y], outputs=[txt_prediction, txt_certainty, plot_visualize])
                num_testvector_x.change(fn=self.predict_data, inputs=[num_testvector_x, num_testvector_y], outputs=[txt_prediction, txt_certainty, plot_visualize])
                num_testvector_y.change(fn=self.predict_data, inputs=[num_testvector_x, num_testvector_y], outputs=[txt_prediction, txt_certainty, plot_visualize])
        return ui


with gr.Blocks() as demo:
    Main().ui()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
