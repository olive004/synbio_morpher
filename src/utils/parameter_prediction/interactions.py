import numpy as np


class InteractionData():
    def __init__(self, data, simulator):
        self.simulator = simulator
        self.data, self.matrix = self.parse(data)

    def parse(self, data):
        self.matrix = self.make_matrix(data)
        return data
    
    def make_matrix(self, data):
        matrix = np.zeros((len(data), len(data)))
        for i, sample_i, sample_interactions in enumerate(data.items()):
            for j, sample_j, raw_data in enumerate(sample_interactions.items()):
                matrix[i, j] = self.calculate_interaction(raw_data)

            sample_interactions

    def calculate_interaction(self, data):
        if data == False:
            return 0
        interaction_keys = 
