import numpy as np
import networkx as nx

from src.utils.data.fake_data_generation.toy_graphs import square_matrix_rand


class BaseSystem():
    def __init__(self, config_args: dict = None):
        super().__init__()

        self.nodes = self.make_inputs(config_args)
        self.interactions = self.init_interaction_matrix()

        self.node_labels = None
        self.graph = self.build_graph(self.interactions)

    def build_graph(self, source_matrix=None) -> nx.DiGraph:
        if source_matrix is not None:
            return nx.from_numpy_matrix(source_matrix, create_using=nx.DiGraph)
        return nx.from_numpy_matrix(self.interactions, create_using=nx.DiGraph)

    def determine_input_type(self) -> str:
        raise NotImplementedError

    def make_inputs(self, configs=None):
        return list(np.range(np.random.randint(1, 10)))

    def init_interaction_matrix(self, config_args=None) -> np.array:
        return InteractionMatrix(toy=False).matrix

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
                 toy=False,
                 num_nodes=None):
        super().__init__()

        self.toy = toy
        self.config_args = config_args

        if toy:
            self.matrix = self.make_toy_matrix(num_nodes)
        else:
            self.matrix = self.make_matrix()

    def make_matrix(self):
        num_nodes = self.config_args
        return square_matrix_rand(num_nodes)

    def make_toy_matrix(self, num_nodes=None):
        if not num_nodes:
            min_nodes = 2
            max_nodes = 15
            num_nodes = np.random.randint(min_nodes, max_nodes)
        return self.make_matrix(num_nodes)
