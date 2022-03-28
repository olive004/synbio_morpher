from typing import List, Tuple
import numpy as np
import networkx as nx
import logging

from src.srv.parameter_prediction.interactions import InteractionMatrix
from src.srv.results.result_writer import ResultWriter


FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
FORMAT = "%(filename)s:%(funcName)s():%(lineno)i: %(message)s %(levelname)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseSpecies():
    def __init__(self, config_args: dict) -> None:

        # Probability distribution for each interaction component?
        # Can also use different types of moments of prob distribution
        # for each

        self.data = config_args.get("data", None)
        self.identities = {
            "input": 0,
            "output": 1
        }

        self.interactions = self.init_matrix(ndims=2, init_type="randint")
        self.complexes = self.init_matrix(ndims=2, init_type="zeros")
        self.degradation_rates = self.init_matrix(ndims=1, init_type="uniform",
                                                  uniform_val=20)
        self.creation_rates = self.init_matrix(ndims=1, init_type="uniform",
                                               uniform_val=50)
        self.all_copynumbers = None
        self.copynumbers = None
        self.steady_state_copynums = self.init_matrix(ndims=1, init_type="zeros")

        self.params = {
            "creation_rates": self.creation_rates,
            "copynumbers": self.copynumbers,
            "complexes": self.complexes,
            "degradation_rates": self.degradation_rates,
            "interactions": self.interactions
        }

    def init_matrices(self, uniform_vals, ndims=2, init_type="rand") -> List[np.array]:
        matrices = (self.init_matrix(ndims, init_type, val) for val in uniform_vals)
        return tuple(matrices)

    def init_matrix(self, ndims=2, init_type="rand", uniform_val=1) -> np.array:
        matrix_size = np.random.randint(5) if self.data is None \
            else self.data.size
        if ndims > 1:
            matrix_shape = tuple([matrix_size]*ndims)
        else:
            matrix_shape = (1, matrix_size)

        if init_type == "rand":
            return InteractionMatrix(num_nodes=matrix_size).matrix
        elif init_type == "randint":
            return np.random.randint(10, 1000, matrix_shape).astype(np.float64)
        elif init_type == "uniform":
            return np.ones(matrix_shape) * uniform_val
        elif init_type == "zeros":
            return np.zeros(matrix_shape)
        raise ValueError(f"Matrix init type {init_type} not recognised.")

    @property
    def interactions(self):
        return self._interactions

    @interactions.setter
    def interactions(self, new_interactions):
        if type(new_interactions) == np.ndarray:
            self._interactions = new_interactions
        else:
            raise ValueError('Cannot set interactions to' +
                             f' type {type(new_interactions)}.')

    @property
    def copynumbers(self):
        if self.all_copynumbers is not None:
            self._copynumbers = self.all_copynumbers[:, -1]
        else:
            self._copynumbers = self._all_copynumbers
        return self._copynumbers

    @copynumbers.setter
    def copynumbers(self, value):
        # This may not be as fast as a[a < 0] = 0
        if value is not None:
            value[value < 0] = 0
            self._copynumbers = value

    @property
    def all_copynumbers(self):
        return self._all_copynumbers

    @all_copynumbers.setter
    def all_copynumbers(self, value):
        """ Careful: does not kick in for setting slices """
        # This may not be as fast as a[a < 0] = 0
        if value is not None:
            value[value < 0] = 0
        self._all_copynumbers = value


class BaseSystem():
    def __init__(self, config_args: dict = None):

        if config_args is None:
            config_args = {}

        self.species = BaseSpecies(config_args)
        self.result_writer = ResultWriter()

        self.init_graph()

    def init_graph(self):
        self._node_labels = None
        self.graph = self.build_graph()

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

    def determine_input_type(self) -> str:
        raise NotImplementedError

    def simulate_signal(self, signal):
        pass

    def visualise(self, mode="pyvis", new_vis=False):
        self.refresh_graph()

        if mode == 'pyvis':
            from src.srv.results.visualisation import visualise_graph_pyvis
            visualise_graph_pyvis(self.graph, new_vis=new_vis)
        else:
            from src.srv.results.visualisation import visualise_graph_pyplot
            visualise_graph_pyplot(self.graph, new_vis=new_vis)

        self.result_writer.write_all()

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
