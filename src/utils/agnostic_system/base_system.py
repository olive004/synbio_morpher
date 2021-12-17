import numpy as np
import networkx as nx


class BaseSystem():
    def __init__(self, config_args):
        super().__init__()

        self.interactions = self.init_interaction_matrix()

    def build_graph(self, source_matrix=None):
        if source_matrix is not None:
            return nx.from_numpy_matrix(source_matrix)
        return nx.from_numpy_matrix(self.interactions)

    def init_interaction_matrix(self, config_args=None):
        return InteractionMatrix().matrix

    def visualise(self):
        import matplotlib.pyplot as plt
        graph = self.build_graph(self.interactions)
        
        ax1 = plt.subplot(111)
        nx.draw(graph)


class Component():
    def __init__(self):
        super().__init__()

        # Probability distribution for each interaction component?
        # Can also use different types of moments of prob distribution 
        # for each 


class InteractionMatrix():
    def __init__(self, config_args=None):
        super().__init__()

        self.dims = config_args['dimensions'] if config_args else (2, 2)
        self.matrix = np.random.rand(*self.dims)
