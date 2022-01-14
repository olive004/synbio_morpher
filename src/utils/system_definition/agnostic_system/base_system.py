import numpy as np
import networkx as nx

from src.utils.data.fake_data_generation.toy_graphs import square_matrix


class BaseSystem():
    def __init__(self, config_args: dict = None):
        super().__init__()

        self.interactions = self.init_interaction_matrix()

        self.node_labels = None
        self.graph = self.build_graph(self.interactions)

    def build_graph(self, source_matrix=None) -> nx.DiGraph:
        if source_matrix is not None:
            return nx.from_numpy_matrix(source_matrix)
        return nx.from_numpy_matrix(self.interactions)

    def init_interaction_matrix(self, config_args=None):
        return InteractionMatrix(toy=False).matrix

    def visualise(self, mode="pyvis"):
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

    def get_graph_labels(self):
        return sorted(self.graph)


class Component():
    def __init__(self):
        super().__init__()

        # Probability distribution for each interaction component?
        # Can also use different types of moments of prob distribution
        # for each


class InteractionMatrix():
    def __init__(self, config_args=None,
                 toy=False):
        super().__init__()
        
        self.toy = toy

        self.matrix = self.make_toy_matrix()

    def make_toy_matrix(self):
        if self.toy:
            min_nodes = 2
            max_nodes = 15
            num_nodes = np.random.randint(min_nodes, max_nodes)
            return square_matrix(num_nodes)
        return square_matrix()
