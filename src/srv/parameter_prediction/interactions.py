

import numpy as np
import os
import pandas as pd
from src.srv.parameter_prediction.simulator import RawSimulationHandling
from src.utils.misc.type_handling import flatten_listlike
from src.srv.io.loaders.misc import load_csv
from src.utils.data.data_format_tools.common import determine_data_format
from src.utils.misc.io import load_experiment_config
from src.utils.misc.numerical import square_matrix_rand
from src.utils.misc.type_handling import flatten_listlike


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


class InteractionMatrix():

    def __init__(self,  # config_args=None,
                 matrix=None,
                 matrix_path: str = None,
                 num_nodes: int = None,
                 toy=False,
                 units=''):
        super().__init__()

        self.name = None
        self.toy = toy
        self.units = units

        if matrix is not None:
            self.matrix = matrix
        elif matrix_path is not None:
            self.matrix, self.units = self.load(matrix_path)
        elif toy:
            self.matrix = self.make_toy_matrix(num_nodes)
        else:
            self.matrix = self.make_rand_matrix(num_nodes)

    def load(self, filepath):
        filetype = determine_data_format(filepath)
        self.name = os.path.basename(filepath).replace('.'+filetype, '').replace(
            'interactions_', '').replace('_interactions', '')
        if filetype == 'csv':
            matrix = load_csv(filepath, load_as='numpy')
        else:
            raise TypeError(
                f'Unsupported filetype {filetype} for loading {filepath}')
        self.units = self.load_units(filepath)
        return matrix, self.units

    def load_units(self, interactions_filepath):
        experiment_config = load_experiment_config(
            experiment_folder=os.path.dirname(interactions_filepath))
        simulator_cfgs = experiment_config.get('interaction_simulator')
        if simulator_cfgs.get('name') == 'IntaRNA':
            if simulator_cfgs.get('postprocess'):
                return SIMULATOR_UNITS['IntaRNA']['rate']
            else:
                return SIMULATOR_UNITS['IntaRNA']['energy']
        else:
            return SIMULATOR_UNITS['IntaRNA']['rate']

    def make_rand_matrix(self, num_nodes):
        if num_nodes is None or num_nodes == 0:
            num_nodes = 1
        return square_matrix_rand(num_nodes)

    def make_toy_matrix(self, num_nodes=None):
        if not num_nodes:
            min_nodes = 2
            max_nodes = 15
            num_nodes = np.random.randint(min_nodes, max_nodes)
        return self.make_rand_matrix(num_nodes)

    def get_stats(self):
        idxs_interacting = self.get_unique_interacting_idxs()
        interacting = self.get_interacting_species(idxs_interacting)
        self_interacting = self.get_selfinteracting_species(idxs_interacting)

        nonzero_matrix = self.matrix[np.where(self.matrix > 0)]
        if len(nonzero_matrix):
            min_interaction = np.min(nonzero_matrix)
        else:
            min_interaction = np.min(self.matrix)

        stats = {
            "name": self.name,
            "interacting": interacting,
            "self_interacting": self_interacting,
            "num_interacting": len(set(flatten_listlike(interacting))),
            "num_self_interacting": len(set(self_interacting)),
            "max_interaction": np.max(self.matrix),
            "min_interaction": min_interaction
        }
        stats = {k: [v] for k, v in stats.items()}
        stats = pd.DataFrame.from_dict(stats)
        return stats

    def get_interacting_species(self, idxs_interacting):
        return [idx for idx in idxs_interacting if len(set(idx)) > 1]

    def get_selfinteracting_species(self, idxs_interacting):
        return [idx[0] for idx in idxs_interacting if len(set(idx)) == 1]

    def get_unique_interacting_idxs(self):
        idxs_interacting = np.argwhere(self.matrix > 0)
        idxs_interacting = sorted([tuple(sorted(i)) for i in idxs_interacting])
        return list(set(idxs_interacting))
