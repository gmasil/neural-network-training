import numpy as np


class DataGenerator:

    def distance_value_function(self, value, distance=2):
        return 1 if np.linalg.norm(value) > distance else 0

    def even_value_function(self, value):
        return 1 if int(np.linalg.norm(value)) % 2 == 0 else 0

    def generate_training_data(self, training_data_amount, nested=False, value_function=None, seed=-1):
        if value_function is None:
            value_function = self.distance_value_function
        if seed == -1:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(12345)
        training_input_vectors = 3 * rng.random((training_data_amount, 2))
        training_targets = np.array([value_function(v) for v in training_input_vectors])
        if nested:
            return np.array([[v] for v in training_input_vectors]), np.array([[v] for v in training_targets])
        return training_input_vectors, training_targets
