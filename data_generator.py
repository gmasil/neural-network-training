import numpy as np


class DataGenerator:

    def generate_training_data(self, training_data_amount, distance=2):
        training_input_vectors = 3 * np.random.rand(training_data_amount, 2)
        training_targets = np.array([1 if np.linalg.norm(v) > distance else 0 for v in training_input_vectors])
        return training_input_vectors, training_targets
