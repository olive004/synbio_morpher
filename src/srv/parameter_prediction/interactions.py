

import logging
import numpy as np
import os
import pandas as pd
from typing import Tuple, List

from src.srv.parameter_prediction.simulator import RawSimulationHandling
from src.srv.io.loaders.experiment_loading import INTERACTION_FILE_ADDONS, load_param, load_units
from src.utils.misc.type_handling import flatten_listlike
from src.srv.io.loaders.misc import load_csv
from src.utils.data.data_format_tools.common import determine_file_format
from src.utils.misc.type_handling import flatten_listlike


INTERACTION_TYPES = list(INTERACTION_FILE_ADDONS.keys())
# INTERACTION_TYPES.remove('coupled_binding_rates')


class MolecularInteractions():

    def __init__(self,
                 # coupled_binding_rates,
                 binding_rates_association=None,
                 binding_rates_dissociation=None,
                 eqconstants=None, units=None) -> None:
        # self.coupled_binding_rates = coupled_binding_rates
        self.binding_rates_association = binding_rates_association
        self.binding_rates_dissociation = binding_rates_dissociation
        self.eqconstants = eqconstants
        self.units = units


class InteractionMatrix():

    def __init__(self,  # config_args=None,
                 matrix_paths: dict = None,
                 experiment_dir: str = None,
                 num_nodes: int = None,
                 units: str = '',
                 experiment_config: dict = None,
                 interactions_kwargs: dict = None):

        self.name = None
        self.sample_names = None
        self.units = units
        self.experiment_dir = experiment_dir
        self.experiment_config = experiment_config

        init_nodes = num_nodes if num_nodes is not None else 3
        random_matrices = np.random.rand(
            init_nodes, init_nodes, 4) * 0.000001
        self.interactions = MolecularInteractions(
            # coupled_binding_rates=random_matrices[:, :, 0],
            binding_rates_association=random_matrices[:, :, 1],
            binding_rates_dissociation=random_matrices[:, :, 2],
            eqconstants=random_matrices[:, :, 3], units='test'
        )

        if interactions_kwargs is not None:
            self.interactions = MolecularInteractions(**interactions_kwargs)

        elif matrix_paths is not None:
            for matrix_type, matrix_path in matrix_paths.items():
                if isinstance(matrix_path, str):
                    loaded_matrix, self.units, self.sample_names = self.load(
                        matrix_path)
                    self.interactions.__setattr__(matrix_type, loaded_matrix)
                    self.interactions.units = self.units
            if 'binding_rates_association' in matrix_paths:
                self.interactions.binding_rates_association = matrix_paths['binding_rates_association'] * np.ones_like(
                    self.interactions.binding_rates_dissociation)
            else:
                self.interactions.binding_rates_association = load_param(
                    list(matrix_paths.values())[0], 'association_binding_rate', experiment_config=experiment_config
                ) * np.ones_like(self.interactions.binding_rates_dissociation)


    def load(self, filepath):
        filetype = determine_file_format(filepath)

        self.name = self.isolate_circuit_name(filepath, filetype)
        if filetype == 'csv':
            matrix, sample_names = load_csv(
                filepath, load_as='numpy', return_header=True)
        else:
            raise TypeError(
                f'Unsupported filetype {filetype} for loading {filepath}')
        units = load_units(filepath, experiment_config=self.experiment_config)
        return matrix, units, sample_names

    def isolate_circuit_name(self, circuit_filepath, filetype):
        circuit_name = None
        for faddon in INTERACTION_FILE_ADDONS.keys():
            if faddon in circuit_filepath:
                base_name = os.path.basename(circuit_filepath).replace('.'+filetype, '').replace(
                    faddon+'_', '').replace('_'+faddon, '')
                circuit_name = base_name if type(
                    base_name) == str and faddon not in base_name else circuit_name
        if circuit_name is None:
            logging.warning(
                f'Could not find circuit name in {circuit_filepath}')
        return circuit_name

    def get_stats(self, interaction_attr='eqconstants'):
        idxs_interacting = self.get_unique_interacting_idxs()
        interacting = self.get_interacting_species(idxs_interacting)
        self_interacting = self.get_selfinteracting_species(idxs_interacting)

        stats = {
            "name": self.name,
            "interacting": interacting,
            "self_interacting": self_interacting,
            "interacting_names": sorted([(self.sample_names[i[0]], self.sample_names[i[1]]) for i in interacting]),
            "self_interacting_names": sorted([(self.sample_names[i[0]], self.sample_names[i[1]]) for i in self_interacting]),
            "num_interacting": len(set(flatten_listlike(interacting))),
            "num_self_interacting": len(set(self_interacting)),
            "max_interaction": np.max(self.interactions.__getattribute__(interaction_attr)),
            "min_interaction": np.min(self.interactions.__getattribute__(interaction_attr))
        }
        stats = {k: [v] for k, v in stats.items()}
        stats = pd.DataFrame.from_dict(stats, dtype=object)
        return stats

    def get_interacting_species(self, idxs_interacting):
        return [idx for idx in idxs_interacting if len(set(idx)) > 1]

    def get_selfinteracting_species(self, idxs_interacting):
        return [idx for idx in idxs_interacting if len(set(idx)) == 1]

    def get_unique_interacting_idxs(self):
        idxs_interacting = np.argwhere(self.interactions.eqconstants != 1)
        # Assuming symmetry in interactions
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
                # coupled_binding_rates=np.random.rand(len(data), len(data)),
                binding_rates_association=np.random.rand(len(data), len(data)),
                binding_rates_dissociation=np.random.rand(
                    len(data), len(data)),
                eqconstants=np.random.rand(len(data), len(data)),
            )
        self.interactions.units = simulation_handler.units

    def calculate_full_coupling_of_rates(self, eqconstants, k_d):
        coupled_binding_rates = self.simulation_handler.calculate_full_coupling_of_rates(
            k_d=k_d, eqconstants=eqconstants
        )
        return coupled_binding_rates

    def parse(self, data: dict) -> MolecularInteractions:
        matrix, a_rates, d_rates = self.make_matrix(data)
        # coupled_binding_rates = self.calculate_full_coupling_of_rates(
        #     matrix, d_rates)
        return MolecularInteractions(
            binding_rates_association=a_rates,
            binding_rates_dissociation=d_rates, eqconstants=matrix)

    def make_matrix(self, data: dict) -> Tuple[np.ndarray, np.ndarray]:
        matrix = np.zeros((len(data), len(data)))
        for i, (name_i, sample) in enumerate(data.items()):
            for j, (name_j, raw_sample) in enumerate(sample.items()):
                matrix[i, j] = self.process_interaction(raw_sample)
        matrix, (a_rates, d_rates) = self.simulation_postproc(matrix)
        return matrix, a_rates, d_rates

    def process_interaction(self, sample):
        if sample == False:  # No interactions
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


def b_get_stats(interactions_mxs: List[InteractionMatrix]):
    b_interaction_attrs = {}
    for interaction_attr in INTERACTION_TYPES:
        b_interaction_attrs[interaction_attr] = np.concatenate([np.expand_dims(
            im.interactions.__getattribute__(interaction_attr), axis=0) for im in interactions_mxs], axis=0)
    # For 0-indexing
    batch_dim = b_interaction_attrs['eqconstants'].ndim - 2 - 1
    idxs_interacting = get_unique_interacting_idxs(
        b_interaction_attrs['eqconstants'], batch_dim)

    idxs_other_interacting = get_interacting_species(idxs_interacting)
    idxs_self_interacting = get_selfinteracting_species(idxs_interacting)

    idxs_other_interacting = [idxs_other_interacting[(
        idxs_other_interacting[:, :batch_dim+1] == i).flatten()][:, batch_dim+1:] for i in range(len(interactions_mxs))]
    idxs_self_interacting = [idxs_self_interacting[(idxs_self_interacting[:, :batch_dim+1] == i).flatten(
    )][:, batch_dim+1:] for i in range(len(interactions_mxs))]

    stats = {
        # "name": [i.name for i in interactions],
        "interacting": idxs_other_interacting,
        "self_interacting": idxs_self_interacting,
        "sample_names": [i.sample_names for i in interactions_mxs],
        "num_interacting": [len(i) for i in idxs_other_interacting],
        "num_self_interacting": [len(i) for i in idxs_self_interacting]
    }

    for interaction_attr in INTERACTION_TYPES:
        for i, s in enumerate(interactions_mxs[0].sample_names):
            for ii, s in enumerate(interactions_mxs[0].sample_names):
                stats[interaction_attr + '_' + str(i) + '-' + str(
                    ii)] = b_interaction_attrs[interaction_attr][:, i, ii]
        stats[interaction_attr + '_' + 'max_interaction'] = np.max(
            np.max(b_interaction_attrs[interaction_attr], axis=1), axis=1)
        stats[interaction_attr + '_' + 'min_interaction'] = np.min(
            np.min(b_interaction_attrs[interaction_attr], axis=1), axis=1)
    # stats = {k: [v] for k, v in stats.items()}
    stats = pd.DataFrame.from_dict(stats, dtype=object)
    return stats


def get_interacting_species(idxs_interacting: np.ndarray):
    assert idxs_interacting.ndim == 2, f'Array of indices should be 2-D: {idxs_interacting}'
    return idxs_interacting[idxs_interacting[:, -1] != idxs_interacting[:, -2], :]


def get_selfinteracting_species(idxs_interacting: np.ndarray):
    assert idxs_interacting.ndim == 2, f'Array of indices should be 2-D: {idxs_interacting}'
    return idxs_interacting[idxs_interacting[:, -1] == idxs_interacting[:, -2], :]


def get_unique_interacting_idxs(interaction_attr: np.ndarray, batch_dim: int):
    idxs_interacting = np.argwhere(interaction_attr != 1)
    samples = idxs_interacting[:, batch_dim+1:]
    samples.sort(axis=1)  # In-place sorting of idxs_interacting
    # idxs_interacting = np.concatenate(
    #     (idxs_interacting[:, :batch_dim], samples), axis=1)
    idxs_interacting = np.unique(idxs_interacting, axis=0)
    return idxs_interacting
    # Assuming symmetry in interactions
    # idxs_interacting = sorted([tuple(sorted(i)) for i in idxs_interacting])
    # return list(set(idxs_interacting))
