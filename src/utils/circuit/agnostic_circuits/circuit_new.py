from copy import deepcopy
from typing import List, Union
import pandas as pd
import networkx as nx
import numpy as np
import logging

from src.srv.parameter_prediction.interactions import MolecularInteractions
from src.srv.io.manage.data_manager import DataManager
from src.utils.circuit.common.system_setup import construct_bioreaction_model
from src.utils.results.results import ResultCollector
from src.utils.signal.signals_new import Signal
from bioreaction.model.data_containers import BasicModel, QuantifiedReactions


FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
FORMAT = "%(filename)s:%(funcName)s():%(lineno)i: %(message)s %(levelname)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def interactions_to_dict(interactions: np.ndarray, labels: list):
    interactions_dict = {}
    for i, sample in enumerate(labels):
        interactions_dict[sample] = {s: interactions[i, j]
                                     for j, s in enumerate(labels)}
    return interactions_dict


def interactions_to_df(interactions: np.ndarray, labels: list):
    interactions_df = pd.DataFrame.from_dict(
        interactions_to_dict(interactions, labels), dtype=object)
    return interactions_df


class Graph():

    def __init__(self, labels: List[str], source_matrix: np.ndarray = None) -> None:
        source_matrix = np.zeros(
            (len(labels), len(labels))) if source_matrix is None else source_matrix
        self._node_labels = labels
        self.build_graph(source_matrix)

    def build_graph(self, source_matrix: np.ndarray) -> nx.DiGraph:
        graph = nx.from_numpy_matrix(source_matrix, create_using=nx.DiGraph)
        if self.node_labels is not None:
            graph = nx.relabel_nodes(graph, self.node_labels)
        return graph

    def refresh_graph(self, source_matrix: np.ndarray):
        self.graph = self.build_graph(source_matrix)

    def get_graph_labels(self) -> dict:
        return sorted(self.graph)

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, new_graph):
        assert type(new_graph) == nx.DiGraph, 'Cannot set graph to' + \
            f' type {type(new_graph)}.'
        self._graph = new_graph

    @property
    def node_labels(self):
        if type(self._node_labels) == list:
            return {i: n for i, n in enumerate(self._node_labels)}
        return self._node_labels

    @node_labels.setter
    def node_labels(self, labels: Union[dict, list]):

        current_nodes = self.get_graph_labels()
        if type(labels) == list:
            self._node_labels = dict(zip(current_nodes, labels))
        else:
            labels = len(labels)
            self._node_labels = dict(zip(current_nodes, labels))
        self.graph = nx.relabel_nodes(self.graph, self.node_labels)


class Circuit():

    species_axis = 0
    time_axis = 1

    def __init__(self, config: dict):

        self.name = config.get("name")

        self.result_collector = ResultCollector()
        self.model = construct_bioreaction_model(
            config.get('data'), config.get('molecular_params'))
        self.data: DataManager = config.get('data')
        self.qreactions = self.init_reactions(self.model, config)
        self.interactions = self.init_interactions()
        self.species_state: str = config.get('species_state', 'uninitialised')
        self.signal: Signal = None
        self.mutations = {}
        self.mutations_args: dict = config.get('mutations', {})

        self.graph = Graph(source_matrix=self.interactions.eqconstants, labels=[
                           s.name for s in self.model.species])
        self.circuit_size = len(self.model.species)

    def init_reactions(self, model: BasicModel, config: dict) -> QuantifiedReactions:
        qreactions = QuantifiedReactions()
        qreactions.init_properties(model, config)
        return qreactions

    def init_interactions(self):
        matrix = np.zeros((len(self.model.species), len(self.model.species)))
        for r in self.model.reactions:
            if len(r.input) == 2:
                si = r.input[0]
                sj = r.input[1]
                matrix[self.model.species.index(si), self.model.species.index(
                    sj)] = r.forward_rate
        # for si in self.model.species:
        #     for sj in self.model.species:
        #         candidate_reactions = [
        #             r for r in self.model.reactions if si in r.input and sj in r.input]
        #         if len(candidate_reactions) == 0:
        #             continue
        #             # raise ValueError(
        #             #     f'The species {si} and {sj} were not in any reaction inputs')
        #         if len(candidate_reactions) > 1:
        #             raise ValueError(
        #                 f'Multiple reactions found in unique list {candidate_reactions}')
        #         matrix[self.model.species.index(si), self.model.species.index(
        #             sj)] = candidate_reactions[0].forward_rate
        return MolecularInteractions(
            coupled_binding_rates=matrix)

    def reset_to_initial_state(self):
        self.result_collector.reset()

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