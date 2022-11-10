from copy import deepcopy
from typing import List
import numpy as np
import pandas as pd
import networkx as nx
import logging

from src.srv.parameter_prediction.interactions import MolecularInteractions
from src.utils.results.results import ResultCollector
import bioreaction


FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
FORMAT = "%(filename)s:%(funcName)s():%(lineno)i: %(message)s %(levelname)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def interactions_to_dict(self, interactions: np.ndarray, labels: list):
    interactions_dict = {}
    for i, sample in enumerate(labels):
        interactions_dict[sample] = {s: interactions[i, j]
                                     for j, s in enumerate(labels)}
    return interactions_dict


def interactions_to_df(interactions: np.ndarray, labels: list):
    interactions_df = pd.DataFrame.from_dict(
        interactions_to_dict(interactions, labels))
    return interactions_df


class Graph():

    def __init__(self, source_matrix=None) -> None:
        self.build_graph()

    def build_graph(self, source_matrix=None) -> nx.DiGraph:
        if source_matrix is not None:
            inters = source_matrix
        else:
            inters = self.species.interactions
        graph = nx.from_numpy_matrix(inters, create_using=nx.DiGraph)
        if self.node_labels is not None:
            graph = nx.relabel_nodes(graph, self.node_labels)
        return graph

    def refresh_graph(self):
        self.graph = self.build_graph()

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
        return self._node_labels

    @node_labels.setter
    def node_labels(self, labels: dict):

        current_nodes = self.get_graph_labels()
        if type(labels) == list:
            self._node_labels = dict(zip(current_nodes, labels))
        else:
            labels = list(range(len(self.species.interactions)))
            self._node_labels = dict(zip(current_nodes, labels))
        self.graph = nx.relabel_nodes(self.graph, self.node_labels)


class Circuit():
    def __init__(self, config: dict, model: bioreaction.model.data_containers.BasicModel):

        self.name = config.get("name")

        self.result_collector = ResultCollector()
        self.species = model.species
        self.reactions = self.init_reactions(model, config)
        self.interactions = MolecularInteractions(
            interactions=self.reactions.rates)

        self.graph = Graph(self.species, self.interactions)
        self.circuit_size = len(self.species)

    def init_reactions(self, model, config):
        from bioreaction.model.data_containers import QuantifiedReactions
        qreactions = QuantifiedReactions()
        qreactions.init_properties(model, config)
        return qreactions

    def reset_to_initial_state(self):
        self.result_collector.reset()

    def make_subcircuit(self, mutation_name: str, mutation=None):
        subcircuit = deepcopy(self)
        subcircuit.reset_to_initial_state()
        subcircuit.species.are_interactions_loaded = False

        if mutation is None:
            mutation = self.species.mutations.get(mutation_name)

        subcircuit.species.mutate(mutation)
        return subcircuit

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
