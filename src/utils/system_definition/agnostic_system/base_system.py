import numpy as np
import networkx as nx

from src.utils.data.fake_data_generation.toy_graphs import square_matrix_rand


class BaseSystem():
    def __init__(self, config_args: dict = None):

        if config_args is None:
            config_args = {}

        self.data = config_args.get("data", None)
        self.interactions = self.init_interaction_matrix()

        self.graph = self.build_graph(self.interactions)

    def build_graph(self, source_matrix=None) -> nx.DiGraph:
        if source_matrix is not None:
            return nx.from_numpy_matrix(source_matrix, create_using=nx.DiGraph)
        return nx.from_numpy_matrix(self.interactions, create_using=nx.DiGraph)

    def determine_input_type(self) -> str:
        raise NotImplementedError

    def init_interaction_matrix(self) -> np.array:
        matrix_size = self.data.size
        return InteractionMatrix(num_nodes=matrix_size).matrix

    def visualise(self, mode="pyvis"):
        self.graph = self.build_graph(self.interactions)

        if self.node_labels:
            self.graph = nx.relabel_nodes(self.graph, self.node_labels)

        if mode == 'pyvis':
            from src.utils.visualisation.graph_drawer import visualise_graph_pyvis
            visualise_graph_pyvis(self.graph)
        else:
            from src.utils.visualisation.graph_drawer import visualise_graph_pyplot
            visualise_graph_pyplot(self.graph)

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, nodes: dict):
        if type(nodes) == dict:
            self.label_nodes(list(nodes.keys()))
        self._nodes = list(nodes.values())

    def label_nodes(self, labels):
        if type(labels) == list:
            current_nodes = self.get_graph_labels()
            self.node_labels = dict(zip(current_nodes, labels))
        elif type(labels) == dict:
            self.node_labels = labels

        self.graph = nx.relabel_nodes(self.graph, self.node_labels)

    def get_graph_labels(self) -> dict:
        return sorted(self.graph)


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
