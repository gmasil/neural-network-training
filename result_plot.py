import matplotlib.pyplot as plt
import numpy as np
from data_generator import DataGenerator


class ResultPlot:

    def _get_target_color(self, value):
        if value == 0:
            return 'r'
        return 'g'

    def create_plot(self, vectors, targets, input_vector, input_target, show_plot=False):

        target_colors = np.array([self._get_target_color(v) for v in targets])
        input_target_color = self._get_target_color(input_target)

        fig = plt.figure()
        axis = plt.gca()
        axis.set_xlim([0, 3])
        axis.set_ylim([0, 3])
        plt.scatter(vectors[:, 0], vectors[:, 1], color=target_colors, s=20)
        plt.scatter(input_vector[0], input_vector[1], color=input_target_color, s=100, marker='*')
        if show_plot:
            plt.show()
        plt.close()
        return fig


if __name__ == "__main__":
    training_input_vectors, training_targets = DataGenerator().generate_training_data(100)
    ResultPlot().create_plot(training_input_vectors, training_targets, np.array([0, 0]), 1, show_plot=True)
