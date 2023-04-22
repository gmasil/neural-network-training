import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from neural_network import NeuralNetwork


class Main:

    def __init__(self):
        self.network = None
        matplotlib.use("Agg")
        self.training_input_vectors = np.array(
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
        self.training_targets = np.array([1, 0, 1, 1, 1, 0, 1, 0])

    def train_network(self, iterations, learning_rate, plot_log):
        self.network = NeuralNetwork(learning_rate)
        training_errors = self.network.train(self.training_input_vectors, self.training_targets, iterations)
        fig = plt.figure()
        plt.plot(training_errors)
        plt.xlabel("Iterations")
        plt.ylabel("Mean square error")
        if plot_log:
            plt.yscale('log')
        plt.close()
        return fig, np.amin(training_errors)

    def predict_data(self, vector_x, vector_y):
        if self.network is None:
            return "The network has to be trained first", "", None
        vector = np.array([vector_x, vector_y])
        prediction = self.network.predict(vector)
        interpretation = "green" if int(prediction + 0.5) == 1 else "red"

        def color(value):
            if value == 0:
                return 'r'
            return 'g'

        vectors = np.append(self.training_input_vectors.copy(), [vector], axis=0)
        targets = np.array([color(v) for v in self.training_targets])
        targets = np.append(targets, [['b']])
        origin = np.zeros((len(vectors), 2))

        fig = plt.figure()
        axis = plt.gca()
        axis.set_xlim([-1, 7])
        axis.set_ylim([-1, 7])
        plt.quiver(origin[:, 0], origin[:, 1], vectors[:, 0], vectors[:, 1], color=targets, angles='xy', scale_units='xy', scale=1)
        plt.close()

        return prediction, interpretation, fig

    def ui(self):
        with gr.Row() as ui:
            with gr.Column():
                with gr.Tab("Training"):
                    with gr.Row():
                        num_iterations = gr.Number(label="Iterations", value=10, precision=0)
                        num_learning_rate = gr.Number(label="Learning rate", value=1, precision=4)
                    chb_plot_log = gr.Checkbox(label="Plot Y axis logarithmic")
                    btn_train = gr.Button("Train network")
                    plot_mse = gr.Plot()
                    txt_mse = gr.Text(label="Minimum MSE")
                btn_train.click(fn=self.train_network, inputs=[num_iterations, num_learning_rate, chb_plot_log], outputs=[plot_mse, txt_mse])
            with gr.Column():
                with gr.Tab("Prediction"):
                    with gr.Row():
                        num_testvector_x = gr.Number(label="Vector X value", value=2.5, precision=3)
                        num_testvector_y = gr.Number(label="Vector Y value", value=1.9, precision=3)
                    btn_predict = gr.Button("Predict")
                    with gr.Row():
                        txt_prediction = gr.Text(label="Prediction")
                        txt_interpretation = gr.Text(label="Interpretation")
                    plot_visualize = gr.Plot()
                btn_predict.click(fn=self.predict_data, inputs=[num_testvector_x, num_testvector_y], outputs=[txt_prediction, txt_interpretation, plot_visualize])
        return ui


with gr.Blocks() as demo:
    Main().ui()

if __name__ == "__main__":
    demo.launch()
