from typing import List, Union
import pandas as pd
import networkx as nx
import numpy as np
import logging

from src.srv.parameter_prediction.interactions import MolecularInteractions, InteractionMatrix
from src.srv.io.manage.data_manager import DataManager
from src.utils.circuit.common.system_setup import construct_bioreaction_model
from src.utils.misc.type_handling import flatten_listlike, get_unique
from src.utils.results.results import ResultCollector
from src.utils.signal.signals_new import Signal
from bioreaction.model.data_containers import BasicModel, QuantifiedReactions


FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
FORMAT = "%(filename)s:%(funcName)s():%(lineno)i: %(message)s %(levelname)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# def interactions_to_dict(interactions: np.ndarray, labels: list):
#     interactions_dict = {}
#     for i, sample in enumerate(labels):
#         interactions_dict[sample] = {s: interactions[i][j]
#                                      for j, s in enumerate(labels)}
#     return interactions_dict


def interactions_to_df(interactions: Union[np.ndarray, list], labels: list):
    return pd.DataFrame(data=interactions, columns=labels, index=labels)


class Circuit():

    species_axis = 0
    time_axis = 1

    def __init__(self, config: dict):

        self.name = config.get("name")
        self.subname = 'ref_circuit'

        self.result_collector = ResultCollector()
        self.model = construct_bioreaction_model(
            config.get('data'), config.get('molecular_params'))
        self.circuit_size = len(self.model.species)
        self.data: DataManager = config.get('data')
        self.qreactions = self.init_reactions(self.model, config)
        self.interactions_state: str = config.get(
            'interactions_state', 'uninitialised')
        self.interactions = self.init_interactions(config.get('interactions'), config.get('interactions_loaded'))
        self.signal: Signal = None
        self.mutations = {}
        self.mutations_args: dict = config.get('mutations', {})

        self = update_species_simulated_rates(self, self.interactions)

    def init_reactions(self, model: BasicModel, config: dict) -> QuantifiedReactions:
        qreactions = QuantifiedReactions()
        qreactions.init_properties(model, config)
        return qreactions

    def init_interactions(self, interaction_cfg: dict, interactions_loaded: dict) -> MolecularInteractions:
        if interaction_cfg is None and interactions_loaded is None:
            # matrix = np.zeros(
            #     (len(self.model.species), len(self.model.species)))
            # for r in self.model.reactions:
            #     if len(r.input) == 2:
            #         si = r.input[0]
            #         sj = r.input[1]
            #         matrix[self.model.species.index(si), self.model.species.index(
            #             sj)] = r.reverse_rate
            # return MolecularInteractions(binding_rates_dissociation=matrix)
            num_in_species = len(get_unique(flatten_listlike([r.input for r in self.model.reactions])))
            random_matrices = np.random.rand(
                num_in_species, num_in_species, 4) * 0.0001
            return MolecularInteractions(
                # coupled_binding_rates=random_matrices[:, :, 0],
                binding_rates_association=random_matrices[:, :, 1],
                binding_rates_dissociation=random_matrices[:, :, 2],
                eqconstants=random_matrices[:, :, 3], units='test'
            )
        assert self.interactions_state != 'uninitialised', f'The interactions should have been initialised from {interaction_cfg}'
        return InteractionMatrix(matrix_paths=interaction_cfg, interactions_kwargs=interactions_loaded).interactions

    def reset_to_initial_state(self):
        self.result_collector.reset()
        self.interactions_state = 'uninitialised'

    def strip_to_core(self):
        del self.data
        del self.mutations
        del self.mutations_args

    @property
    def signal(self):
        return self._signal

    @signal.getter
    def signal(self):
        if self._signal is None:
            logging.warning(
                f'Trying to retrieve None signal from circuit. Make sure signal specified in circuit config')
        return self._signal

    @signal.setter
    def signal(self, value):
        self._signal = value


def update_species_simulated_rates(circuit: Circuit,
                                   interactions: MolecularInteractions) -> Circuit:
    ordered_species = sorted(get_unique(flatten_listlike([r.input for r in circuit.model.reactions])))
    for i, r in enumerate(circuit.model.reactions):
        if len(r.input) == 2:
            si = r.input[0]
            sj = r.input[1]
            interaction_idx_si = ordered_species.index(si)
            interaction_idx_sj = ordered_species.index(sj)
            circuit.model.reactions[i].forward_rate = interactions.binding_rates_association[
                interaction_idx_si, interaction_idx_sj]
            circuit.model.reactions[i].reverse_rate = interactions.binding_rates_dissociation[
                interaction_idx_si, interaction_idx_sj]
    circuit.qreactions.reactions = circuit.qreactions.init_reactions(
        circuit.model)
    return circuit
