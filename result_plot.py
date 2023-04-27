import matplotlib.pyplot as plt
import numpy as np

from data_generator import DataGenerator


class ResultPlot:

    def _get_target_color(self, value):
        if value == 0:
            return 'r'
        return 'g'

    def _get_dot_color(self, value):
        value = abs(value - 0.5) * 2
        value = value ** 0.1
        value = np.clip(value, 0, 1)
        return [value, value, value]

    def _float_range(self, r_min, r_max, r_step):
        return [round(x * r_step, 5) for x in range(int(r_min/r_step), int(r_max/r_step)+2)]

    def create_dots(self, xmin=0, xmax=5, ymin=0, ymax=5, predict_function=None, step=0.05):
        dots = np.zeros((1, 3))
        if predict_function is None:
            return dots
        for x in self._float_range(xmin, xmax, step):
            for y in self._float_range(ymin, ymax, step):
                value = predict_function(np.array([[x, y]]))[0][0][0]
                dots = np.append(dots, np.array([[x, y, value]]), axis=0)
        return dots

    def create_plot(self, vectors, targets, input_vector=None, input_target=None, show_plot=False, dots=None):

        # unpack vector
        vectors = np.array([v[0] for v in vectors])

        target_colors = np.array([self._get_target_color(v) for v in targets])

        fig = plt.figure()
        axis = plt.gca()
        axis.set_xlim([0, 3])
        axis.set_ylim([0, 3])

        # draw dots
        if dots is not None:
            dot_colors = np.array([self._get_dot_color(v) for v in dots[:, 2]])
            plt.scatter(dots[:, 0], dots[:, 1], color=dot_colors, s=15)

        plt.scatter(vectors[:, 0], vectors[:, 1], color=target_colors, s=20)
        if input_vector is not None and input_target is not None:
            # unpack vector
            input_vector = input_vector[0]
            input_target_color = self._get_target_color(input_target)
            plt.scatter(input_vector[0], input_vector[1], color=input_target_color, s=100, marker='*')
        if show_plot:
            plt.show()
        plt.close()
        return fig


if __name__ == "__main__":
    result_plot = ResultPlot()
    dots = result_plot.create_dots(0, 5, 0, 5, step=0.1)
    training_input_vectors, training_targets = DataGenerator().generate_training_data(50, nested=True)
    result_plot.create_plot(training_input_vectors, training_targets, show_plot=True, dots=dots)
