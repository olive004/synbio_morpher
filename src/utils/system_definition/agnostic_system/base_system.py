from hashlib import new
from multiprocessing.sharedctypes import Value
import numpy as np
import networkx as nx
import logging

from src.utils.data.fake_data_generation.toy_graphs import square_matrix_rand


FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
FORMAT = "%(filename)s:%(funcName)s():%(lineno)i: %(message)s %(levelname)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BaseSpecies():
    def __init__(self, ):

        # Probability distribution for each interaction component?
        # Can also use different types of moments of prob distribution
        # for each

        self.data = None
        self.rates = {
            "generation": 1,
            "removal": 1
        }


class InteractionMatrix():
    def __init__(self, config_args=None,
                 num_nodes=None,
                 toy=False):
        super().__init__()

        self.toy = toy
        self.config_args = config_args

        if toy:
            self.matrix = self.make_toy_matrix(num_nodes)
        else:
            self.matrix = self.make_rand_matrix(num_nodes)

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


class BaseSystem():
    def __init__(self, config_args: dict = None):

        if config_args is None:
            config_args = {}

        self.data = config_args.get("data", None)
        self.interactions = self.init_interaction_matrix()
        self.init_graph()

    def init_graph(self):
        self._node_labels = None
        self.graph = self.build_graph()

    def build_graph(self, source_matrix=None) -> nx.DiGraph:
        if source_matrix is not None:
            inters = source_matrix
        else: 
            inters = self.interactions
        graph = nx.from_numpy_matrix(inters, create_using=nx.DiGraph)
        if self.node_labels is not None:
            graph = nx.relabel_nodes(graph, self.node_labels)
        return graph

    def refresh_graph(self):
        self.graph = self.build_graph()
    
    def get_graph_labels(self) -> dict:
        return sorted(self.graph)

    def determine_input_type(self) -> str:
        raise NotImplementedError

    def init_interaction_matrix(self) -> np.array:
        matrix_size = np.random.randint(5) if self.data is None \
            else self.data.size
        return InteractionMatrix(num_nodes=matrix_size).matrix

    def visualise(self, mode="pyvis"):
        self.refresh_graph()

        if mode == 'pyvis':
            from src.utils.visualisation.graph_drawer import visualise_graph_pyvis
            visualise_graph_pyvis(self.graph)
        else:
            from src.utils.visualisation.graph_drawer import visualise_graph_pyplot
            visualise_graph_pyplot(self.graph)

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, new_graph):
        assert type(new_graph) == nx.DiGraph, 'Cannot set graph to' + \
            f' type {type(new_graph)}.'
        self._graph = new_graph

    @property
    def interactions(self):
        return self._interactions

    @interactions.setter
    def interactions(self, new_interactions):
        if type(new_interactions) == np.ndarray:
            self._interactions = new_interactions
        else:
            raise ValueError('Cannot set interactions to' + \
            f' type {type(new_interactions)}.')

    @property
    def node_labels(self):
        return self._node_labels

    @node_labels.setter
    def node_labels(self, labels: dict):
        current_nodes = self.get_graph_labels()
        if type(labels) == list:
            self._node_labels = dict(zip(current_nodes, labels))
        else:
            labels = list(range(len(self.interactions)))
            self._node_labels = dict(zip(current_nodes, labels))
        self.graph = nx.relabel_nodes(self.graph, self.node_labels)

