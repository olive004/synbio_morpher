

import numpy as np
from src.srv.parameter_prediction.simulator import RawSimulationHandling


class InteractionData():

    def __init__(self, data, simulation_handler: RawSimulationHandling,
                 test_mode=False):
        self.simulation_protocol = simulation_handler.get_protocol()
        self.simulation_postproc = simulation_handler.get_postprocessing()
        if not test_mode:
            self.data, self.matrix = self.parse(data)
        else:
            self.data = self.simulation_protocol()
        self.units = simulation_handler.units

    def parse(self, data):
        matrix = self.make_matrix(data)
        return data, matrix

    def make_matrix(self, data):
        matrix = np.zeros((len(data), len(data)))
        for i, (sample_i, sample_interactions) in enumerate(data.items()):
            for j, (sample_j, raw_sample) in enumerate(sample_interactions.items()):
                matrix[i, j] = self.get_interaction(raw_sample)
        matrix = self.simulation_postproc(matrix)
        return matrix

    def get_interaction(self, sample):
        if sample == False:
            return 0
        return self.simulation_protocol(sample)
