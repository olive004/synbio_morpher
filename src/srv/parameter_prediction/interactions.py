

import logging
import numpy as np
import os
import pandas as pd
from typing import Tuple

from src.srv.parameter_prediction.simulator import SIMULATOR_UNITS, RawSimulationHandling
from src.utils.misc.scripts_io import get_root_experiment_folder, load_experiment_config
from src.srv.io.loaders.experiment import INTERACTION_FILE_ADDONS, load_param, load_units
from src.utils.misc.type_handling import flatten_listlike
from src.srv.io.loaders.misc import load_csv
from src.utils.data.data_format_tools.common import determine_file_format
from src.utils.misc.numerical import square_matrix_rand
from src.utils.misc.type_handling import flatten_listlike


class MolecularInteractions():

    def __init__(self, coupled_binding_rates,
                 binding_rates_association=None,
                 binding_rates_dissociation=None,
                 eqconstants=None, units=None) -> None:
        self.coupled_binding_rates = coupled_binding_rates
        self.binding_rates_association = binding_rates_association
        self.binding_rates_dissociation = binding_rates_dissociation
        self.eqconstants = eqconstants
        self.units = units


class InteractionMatrix():

    def __init__(self,  # config_args=None,
                 matrix_paths: dict = None,
                 experiment_dir: str = None,
                 num_nodes: int = None,
                 units=''):
        super().__init__()

        self.name = None
        self.units = units
        self.experiment_dir = experiment_dir

        random_matrices = np.random.rand(
            num_nodes, num_nodes, 4) * 0.000001
        self.interactions = MolecularInteractions(
            coupled_binding_rates=random_matrices[:, :, 0],
            binding_rates_association=random_matrices[:, :, 1],
            binding_rates_dissociation=random_matrices[:, :, 2],
            eqconstants=random_matrices[:, :, 3], units='test'
        )
        self.sample_names = None
        if matrix_paths is not None:
            self.interactions.binding_rates_association = load_param(matrix_path, 'creation_rate')
            for matrix_type, matrix_path in matrix_paths.items():
                loaded_matrix, self.units, self.sample_names = self.load(
                    matrix_path)
                self.interactions.__setattr__(matrix_type, loaded_matrix)
                self.interactions.units = self.units
        elif num_nodes is None:
            self.interactions.coupled_binding_rates = self.make_toy_matrix()

    def load(self, filepath):
        filetype = determine_file_format(filepath)

        self.name = self.isolate_circuit_name(filepath, filetype)
        if filetype == 'csv':
            matrix, sample_names = load_csv(
                filepath, load_as='numpy', return_header=True)
        else:
            raise TypeError(
                f'Unsupported filetype {filetype} for loading {filepath}')
        units = load_units(filepath)
        return matrix, units, sample_names

    def isolate_circuit_name(self, circuit_filepath, filetype):
        circuit_name = None
        for faddon in INTERACTION_FILE_ADDONS.keys():
            base_name = os.path.basename(circuit_filepath).replace('.'+filetype, '').replace(
                faddon+'_', '').replace('_'+faddon, '')
            circuit_name = base_name if type(
                base_name) == str else circuit_name
        return circuit_name

    def make_toy_matrix(self, num_nodes=None):
        if not num_nodes:
            min_nodes = 2
            max_nodes = 15
            num_nodes = np.random.randint(min_nodes, max_nodes)
        return square_matrix_rand(num_nodes)

    def get_stats(self):
        idxs_interacting = self.get_unique_interacting_idxs()
        interacting = self.get_interacting_species(idxs_interacting)
        self_interacting = self.get_selfinteracting_species(idxs_interacting)

        nonzero_matrix = self.interactions[np.where(self.interactions > 0)]
        if len(nonzero_matrix):
            min_interaction = np.min(nonzero_matrix)
        else:
            min_interaction = np.min(self.interactions)

        stats = {
            "name": self.name,
            "interacting": interacting,
            "self_interacting": self_interacting,
            "num_interacting": len(set(flatten_listlike(interacting))),
            "num_self_interacting": len(set(self_interacting)),
            "max_interaction": np.max(self.interactions),
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
        idxs_interacting = np.argwhere(self.interactions.eqconstants < 1)
        idxs_interacting = sorted([tuple(sorted(i)) for i in idxs_interacting])
        return list(set(idxs_interacting))


class InteractionData():

    def __init__(self, data: dict, simulation_handler: RawSimulationHandling,
                 test_mode=False):
        self.simulation_handler = simulation_handler
        self.simulation_protocol = simulation_handler.get_sim_interpretation_protocol()
        self.simulation_postproc = simulation_handler.get_postprocessing()
        if not test_mode:
            self.interactions = self.parse(data)
        else:
            self.interactions = MolecularInteractions(
                coupled_binding_rates=np.random.rand(len(data), len(data)),
                binding_rates_association=np.random.rand(len(data), len(data)),
                binding_rates_dissociation=np.random.rand(
                    len(data), len(data)),
                eqconstants=np.random.rand(len(data), len(data)),
            )
        self.interactions.units = simulation_handler.units

    def calculate_full_coupling_of_rates(self, eqconstants):
        self.coupled_binding_rates = self.simulation_handler.calculate_full_coupling_of_rates(
            k_d=self.interactions.binding_rates_dissociation, eqconstants=eqconstants
        )
        return self.coupled_binding_rates

    def parse(self, data: dict) -> MolecularInteractions:
        matrix, rates = self.make_matrix(data)
        return MolecularInteractions(
            coupled_binding_rates=data, binding_rates_association=rates[0],
            binding_rates_dissociation=rates[1], eqconstants=matrix)

    def make_matrix(self, data: dict) -> Tuple[np.ndarray, np.ndarray]:
        matrix = np.zeros((len(data), len(data)))
        for i, (name_i, sample) in enumerate(data.items()):
            for j, (name_j, raw_sample) in enumerate(sample.items()):
                matrix[i, j] = self.process_interaction(raw_sample)
        matrix, rates = self.simulation_postproc(matrix)
        return matrix, rates

    def process_interaction(self, sample):
        if sample == False:
            logging.warning('Interaction simulation went wrong.')
            return 0
        return self.simulation_protocol(sample)


class InteractionSimulator():
    def __init__(self, sim_args: dict = None):

        self.simulation_handler = RawSimulationHandling(sim_args)

    def run(self, batch: dict = None, allow_self_interaction=True):
        """ Makes nested dictionary for querying interactions as 
        {sample1: {sample2: interaction}} """

        simulator = self.simulation_handler.get_simulator(
            allow_self_interaction)
        data = simulator(batch)
        data = InteractionData(
            data, simulation_handler=self.simulation_handler)
        return data
